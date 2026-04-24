#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse an OpenFOAM log file in streaming mode, extract diagnostic quantities,
save them to CSV, and generate diagnostic figures.

The script is designed for large log files and avoids loading the whole file
into memory. It works line by line and stores one record per physical time step.

Main features
-------------
- automatic phase detection from the log
- automatic detection of the energy variable used by the solver ('h' or 'e')
- dynamic generation of plots for an arbitrary number of phases
- PNG and PDF export for all figures

Author
------
M. de' Michieli Vitturi

Affiliation
-----------
INGV Pisa
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


AUTHOR = "M. de' Michieli Vitturi"
AFFILIATION = "INGV Pisa"
LINE_WIDTH = 1.0


RE_TIME = re.compile(r"^Time = ([0-9Ee+\-\.]+)s$")
RE_EXEC_CLOCK = re.compile(
    r"^ExecutionTime = ([0-9Ee+\-\.]+) s\s+ClockTime = ([0-9Ee+\-\.]+) s$"
)
RE_COURANT = re.compile(
    r"^Courant Number mean: ([0-9Ee+\-\.]+) max: ([0-9Ee+\-\.]+)$"
)
RE_DELTA_T = re.compile(r"^deltaT = ([0-9Ee+\-\.]+)$")
RE_PIMPLE_ITER = re.compile(r"^PIMPLE: Iteration ([0-9]+)$")
RE_PIMPLE_NOT_CONVERGED = re.compile(
    r"^PIMPLE: Not converged within ([0-9]+) iterations$"
)
RE_PIMPLE_INITIAL_RESIDUAL = re.compile(
    r"^PIMPLE iter: ([0-9]+), Initial p_rgh residual: ([0-9Ee+\-\.]+)$"
)
RE_PIMPLE_RESIDUAL_RATIO = re.compile(r"^residual ratio ([0-9Ee+\-\.]+)$")
RE_PIMPLE_BYPASS = re.compile(r"^PIMPLE iter: ([0-9]+) -> Bypassing")
RE_P_RGH_SOLVE = re.compile(
    r"^.*Solving for p_rgh, Initial residual = ([0-9Ee+\-\.]+), "
    r"Final residual = ([0-9Ee+\-\.]+), No Iterations ([0-9]+)$"
)

RE_SCALAR_SOLVE = re.compile(
    r"^.*Solving for ([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_]+))?, "
    r"Initial residual = ([0-9Ee+\-\.]+), "
    r"Final residual = ([0-9Ee+\-\.]+), No Iterations ([0-9]+)$"
)

RE_SCALAR_INITIAL = re.compile(
    r"^([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_]+))? initial residual ([0-9Ee+\-\.]+)$"
)

RE_P_MIN_MAX = re.compile(r"^p, min, max = ([0-9Ee+\-\.]+) ([0-9Ee+\-\.]+)$")
RE_PHASE_T = re.compile(
    r"^([A-Za-z0-9_]+) min/max T ([0-9Ee+\-\.]+) - ([0-9Ee+\-\.]+)$"
)
RE_PHASE_FRACTION = re.compile(
    r"^([A-Za-z0-9_]+) fraction: avg, min, max = "
    r"([0-9Ee+\-\.]+) ([0-9Ee+\-\.]+) ([0-9Ee+\-\.]+)$"
)
RE_PHASE_AFTER_CLIP = re.compile(
    r"^([A-Za-z0-9_]+) after clip fraction: avg, min, max = "
    r"([0-9Ee+\-\.]+) ([0-9Ee+\-\.]+) ([0-9Ee+\-\.]+)$"
)
RE_PACKING_PROXIMITY = re.compile(
    r"^Packing proximity \(sum\(alpha\) - alfasMax\): avg, min, max = "
    r"([0-9Ee+\-\.]+), ([0-9Ee+\-\.]+), ([0-9Ee+\-\.]+)$"
)
RE_ALPHAS_MAX = re.compile(
    r"^alphasMax, min, max = ([0-9Ee+\-\.]+) ([0-9Ee+\-\.]+)$"
)
RE_PHASE_THETA = re.compile(
    r"^([A-Za-z0-9_]+) theta: avg, min, max = "
    r"([0-9Ee+\-\.]+) ([0-9Ee+\-\.]+) ([0-9Ee+\-\.]+)$"
)
RE_AITKEN = re.compile(r"^\s*Continuous Aitken omega: ([0-9Ee+\-\.]+)$")
RE_ENERGY_CHECK = re.compile(
    r"^Iteration ([0-9]+) Check for initial Energy Residual (true|false)$"
)
RE_CONVERGENCE_FLAG = re.compile(r"^convergenceFlag = (true|false)$")


def to_float(text: str) -> float:
    return float(text)


def safe_last(values: List[float]) -> Optional[float]:
    return values[-1] if values else None


def safe_min(values: List[float]) -> Optional[float]:
    return min(values) if values else None


def safe_max(values: List[float]) -> Optional[float]:
    return max(values) if values else None


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()


def initialize_step(time_value: float, pending: Dict[str, Optional[float]]) -> Dict[str, object]:
    return {
        "time": time_value,
        "execution_time_s": pending.get("execution_time_s"),
        "clock_time_s": pending.get("clock_time_s"),
        "courant_max": pending.get("courant_max"),
        "delta_t": pending.get("delta_t"),
        "p_min": None,
        "p_max": None,
        "packing_proximity_avg": None,
        "packing_proximity_min": None,
        "packing_proximity_max": None,
        "alphas_max": None,
        "_phase_theta": {},
        "max_pimple_iteration": 0,
        "pimple_not_converged_limit": None,
        "last_pimple_initial_residual": None,
        "last_pimple_residual_ratio": None,
        "bypass_count": 0,
        "last_aitken_omega": None,
        "last_energy_check_iteration": None,
        "last_energy_check_success": None,
        "last_convergence_flag": None,
        "_p_rgh_initial_residuals": [],
        "_p_rgh_final_residuals": [],
        "_phase_temperature": {},
        "_phase_fraction": {},
        "_phase_fraction_after_clip": {},
        "_scalar_initial_residuals": {},
        "_scalar_linear_iterations": {},
    }


def ensure_nested(
    container: Dict[str, Dict[str, object]],
    key: str,
    default: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if key not in container:
        container[key] = {} if default is None else default.copy()
    return container[key]


def ensure_list_nested(
    container: Dict[str, Dict[str, List[float]]],
    group: str,
    item: str,
) -> List[float]:
    if group not in container:
        container[group] = {}
    if item not in container[group]:
        container[group][item] = []
    return container[group][item]


def finalize_step(step: Dict[str, object]) -> Dict[str, object]:
    flat = dict(step)

    p_rgh_initial = flat.pop("_p_rgh_initial_residuals")
    p_rgh_final = flat.pop("_p_rgh_final_residuals")
    phase_temperature = flat.pop("_phase_temperature")
    phase_fraction = flat.pop("_phase_fraction")
    phase_fraction_after_clip = flat.pop("_phase_fraction_after_clip")
    phase_theta = flat.pop("_phase_theta")
    scalar_initial_residuals = flat.pop("_scalar_initial_residuals")
    scalar_linear_iterations = flat.pop("_scalar_linear_iterations")

    flat["p_rgh_initial_residual_last"] = safe_last(p_rgh_initial)
    flat["p_rgh_initial_residual_max"] = safe_max(p_rgh_initial)
    flat["p_rgh_initial_residual_min"] = safe_min(p_rgh_initial)
    flat["p_rgh_final_residual_last"] = safe_last(p_rgh_final)
    flat["p_rgh_solve_count"] = len(p_rgh_initial)

    for phase, values in phase_temperature.items():
        token = sanitize(phase)
        flat[f"{token}_t_min"] = values.get("min")
        flat[f"{token}_t_max"] = values.get("max")

    for phase, values in phase_fraction.items():
        token = sanitize(phase)
        flat[f"{token}_fraction_avg"] = values.get("avg")
        flat[f"{token}_fraction_min"] = values.get("min")
        flat[f"{token}_fraction_max"] = values.get("max")

    for phase, values in phase_fraction_after_clip.items():
        token = sanitize(phase)
        flat[f"{token}_after_clip_avg"] = values.get("avg")
        flat[f"{token}_after_clip_min"] = values.get("min")
        flat[f"{token}_after_clip_max"] = values.get("max")

    for phase, values in phase_theta.items():
        token = sanitize(phase)
        flat[f"{token}_theta_avg"] = values.get("avg")
        flat[f"{token}_theta_min"] = values.get("min")
        flat[f"{token}_theta_max"] = values.get("max")

    for variable, phase_map in scalar_linear_iterations.items():
        var_token = sanitize(variable)
        for phase, values in phase_map.items():
            phase_token = sanitize(phase)
            prefix = f"{var_token}_{phase_token}"
            flat[f"{prefix}_linear_iterations_last"] = safe_last(values)
            flat[f"{prefix}_linear_iterations_max"] = safe_max(values)
            flat[f"{prefix}_linear_iterations_min"] = safe_min(values)

    for variable, phase_map in scalar_initial_residuals.items():
        var_token = sanitize(variable)
        for phase, values in phase_map.items():
            phase_token = sanitize(phase)
            prefix = f"{var_token}_{phase_token}"
            flat[f"{prefix}_initial_residual_last"] = safe_last(values)
            flat[f"{prefix}_initial_residual_max"] = safe_max(values)
            flat[f"{prefix}_initial_residual_min"] = safe_min(values)
            flat[f"{prefix}_solve_count"] = len(values)

    return flat


def parse_log(log_path: Path) -> Tuple[pd.DataFrame, List[str], Optional[str]]:
    pending = {
        "execution_time_s": None,
        "clock_time_s": None,
        "courant_max": None,
        "delta_t": None,
    }

    steps: List[Dict[str, object]] = []
    current_step: Optional[Dict[str, object]] = None
    phases = set()
    energy_variable = None

    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()

            if not line:
                continue

            match = RE_EXEC_CLOCK.match(line)
            if match:
                pending["execution_time_s"] = to_float(match.group(1))
                pending["clock_time_s"] = to_float(match.group(2))
                continue

            match = RE_COURANT.match(line)
            if match:
                pending["courant_max"] = to_float(match.group(2))
                continue

            match = RE_DELTA_T.match(line)
            if match:
                pending["delta_t"] = to_float(match.group(1))
                continue

            match = RE_TIME.match(line)
            if match:
                if current_step is not None:
                    steps.append(finalize_step(current_step))
                current_step = initialize_step(to_float(match.group(1)), pending)
                continue

            if current_step is None:
                continue

            match = RE_PIMPLE_ITER.match(line)
            if match:
                current_step["max_pimple_iteration"] = max(
                    int(current_step["max_pimple_iteration"]),
                    int(match.group(1)),
                )
                continue

            match = RE_PIMPLE_NOT_CONVERGED.match(line)
            if match:
                current_step["pimple_not_converged_limit"] = int(match.group(1))
                continue

            match = RE_PIMPLE_INITIAL_RESIDUAL.match(line)
            if match:
                current_step["last_pimple_initial_residual"] = to_float(match.group(2))
                continue

            match = RE_PIMPLE_RESIDUAL_RATIO.match(line)
            if match:
                current_step["last_pimple_residual_ratio"] = to_float(match.group(1))
                continue

            match = RE_PIMPLE_BYPASS.match(line)
            if match:
                current_step["bypass_count"] = int(current_step["bypass_count"]) + 1
                continue

            match = RE_P_RGH_SOLVE.match(line)
            if match:
                current_step["_p_rgh_initial_residuals"].append(to_float(match.group(1)))
                current_step["_p_rgh_final_residuals"].append(to_float(match.group(2)))
                continue

            match = RE_P_MIN_MAX.match(line)
            if match:
                current_step["p_min"] = to_float(match.group(1))
                current_step["p_max"] = to_float(match.group(2))
                continue

            match = RE_PHASE_T.match(line)
            if match:
                phase = match.group(1)
                phases.add(phase)
                temp_map = ensure_nested(current_step["_phase_temperature"], phase)
                temp_map["min"] = to_float(match.group(2))
                temp_map["max"] = to_float(match.group(3))
                continue

            match = RE_PHASE_FRACTION.match(line)
            if match:
                phase = match.group(1)
                phases.add(phase)
                frac_map = ensure_nested(current_step["_phase_fraction"], phase)
                frac_map["avg"] = to_float(match.group(2))
                frac_map["min"] = to_float(match.group(3))
                frac_map["max"] = to_float(match.group(4))
                continue

            match = RE_PHASE_AFTER_CLIP.match(line)
            if match:
                phase = match.group(1)
                phases.add(phase)
                frac_map = ensure_nested(current_step["_phase_fraction_after_clip"], phase)
                frac_map["avg"] = to_float(match.group(2))
                frac_map["min"] = to_float(match.group(3))
                frac_map["max"] = to_float(match.group(4))
                continue

            match = RE_PACKING_PROXIMITY.match(line)
            if match:
                current_step["packing_proximity_avg"] = to_float(match.group(1))
                current_step["packing_proximity_min"] = to_float(match.group(2))
                current_step["packing_proximity_max"] = to_float(match.group(3))
                continue

            match = RE_ALPHAS_MAX.match(line)
            if match:
                current_step["alphas_max"] = to_float(match.group(1))
                continue

            match = RE_PHASE_THETA.match(line)
            if match:
                phase = match.group(1)
                phases.add(phase)
                theta_map = ensure_nested(current_step["_phase_theta"], phase)
                theta_map["avg"] = to_float(match.group(2))
                theta_map["min"] = to_float(match.group(3))
                theta_map["max"] = to_float(match.group(4))
                continue

            match = RE_AITKEN.match(line)
            if match:
                current_step["last_aitken_omega"] = to_float(match.group(1))
                continue

            match = RE_ENERGY_CHECK.match(line)
            if match:
                current_step["last_energy_check_iteration"] = int(match.group(1))
                current_step["last_energy_check_success"] = match.group(2) == "true"
                continue

            match = RE_CONVERGENCE_FLAG.match(line)
            if match:
                current_step["last_convergence_flag"] = match.group(1) == "true"
                continue

            match = RE_SCALAR_INITIAL.match(line)
            if match:
                variable = match.group(1)
                phase = match.group(2)
                value = to_float(match.group(3))
                if phase is not None:
                    phases.add(phase)
                phase_key = phase if phase is not None else "_global"
                series = ensure_list_nested(
                    current_step["_scalar_initial_residuals"], variable, phase_key
                )
                series.append(value)
                if variable in {"h", "e"}:
                    energy_variable = variable
                continue

            match = RE_SCALAR_SOLVE.match(line)
            if match:
                variable = match.group(1)
                phase = match.group(2)
                value = to_float(match.group(3))
                if variable == "p_rgh":
                    continue
                if phase is not None:
                    phases.add(phase)
                    phase_key = phase
                else:
                    phase_key = "_global"

                series = ensure_list_nested(
                    current_step["_scalar_initial_residuals"], variable, phase_key
                )
                series.append(value)

                iter_series = ensure_list_nested(
                    current_step["_scalar_linear_iterations"], variable, phase_key
                )
                iter_series.append(int(match.group(5)))

                if variable in {"h", "e"}:
                    energy_variable = variable
                continue

    if current_step is not None:
        steps.append(finalize_step(current_step))

    if not steps:
        raise RuntimeError("No time steps were detected in the log file.")

    dataframe = pd.DataFrame(steps)
    dataframe = dataframe.sort_values("time").reset_index(drop=True)
    dataframe["execution_time_step_s"] = dataframe["execution_time_s"].diff()
    dataframe["clock_time_step_s"] = dataframe["clock_time_s"].diff()
    dataframe["effective_pimple_iterations"] = (
        dataframe["max_pimple_iteration"] - dataframe["bypass_count"]
    ).clip(lower=0)

    return dataframe, sorted(phases), energy_variable


def apply_common_style(ax, xlabel: str = "Physical time [s]") -> None:
    ax.set_xlabel(xlabel)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)


def save_figure(fig, output_path: Path) -> None:
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def merge_legends(ax1, ax2) -> None:
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=True)


def apply_dual_axis_style(ax1, ax2, left_color: str = "tab:blue", right_color: str = "tab:red") -> None:
    ax1.tick_params(axis="y", colors=left_color)
    ax2.tick_params(axis="y", colors=right_color)
    ax1.yaxis.label.set_color(left_color)
    ax2.yaxis.label.set_color(right_color)
    ax1.spines["left"].set_color(left_color)
    ax2.spines["right"].set_color(right_color)


def axis_color_sequence(cmap_name: str, n: int, start: float, stop: float) -> List:
    cmap = plt.get_cmap(cmap_name)
    if n <= 1:
        return [cmap((start + stop) / 2.0)]
    return [cmap(start + i * (stop - start) / (n - 1)) for i in range(n)]


def phase_styles(phases: List[str], cmap_name: str, start: float, stop: float) -> Dict[str, Dict[str, object]]:
    styles = ["-", "--", ":", "-."]
    colors = axis_color_sequence(cmap_name, max(len(phases), 1), start, stop)
    mapping = {}
    for index, phase in enumerate(phases):
        mapping[phase] = {
            "color": colors[index],
            "linestyle": styles[index % len(styles)],
        }
    return mapping


def plot_performance(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time"], df["execution_time_s"], label="Execution time", linewidth=LINE_WIDTH)
    ax.set_ylabel("Execution time [s]")
    ax.set_title("Cumulative execution time")
    apply_common_style(ax)
    ax.legend(frameon=True)
    save_figure(fig, output_dir / "performance_execution_time")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time"], df["execution_time_step_s"], label="Time per step", linewidth=LINE_WIDTH)
    ax.plot(
        df["time"],
        rolling_mean(df["execution_time_step_s"], window=25),
        label="Rolling mean (25 points)",
        linewidth=LINE_WIDTH,
        linestyle="--",
    )
    ax.set_ylabel("Execution time increment [s]")
    ax.set_title("Execution time per time step")
    apply_common_style(ax)
    ax.legend(frameon=True)
    save_figure(fig, output_dir / "performance_time_per_step")


def plot_stability(df: pd.DataFrame, output_dir: Path) -> None:
    if not df["delta_t"].notna().any() and not df["courant_max"].notna().any():
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    if df["delta_t"].notna().any():
        ax1.plot(
            df["time"],
            df["delta_t"],
            color="tab:blue",
            linestyle="-",
            linewidth=LINE_WIDTH,
            label="deltaT",
        )
    ax1.set_ylabel("deltaT [s]")

    if df["courant_max"].notna().any():
        ax2.plot(
            df["time"],
            df["courant_max"],
            color="tab:red",
            linestyle="--",
            linewidth=LINE_WIDTH,
            label="Courant max",
        )
    ax2.set_ylabel("Courant max [-]")

    ax1.set_title("Time step and Courant diagnostics")
    apply_common_style(ax1)
    apply_dual_axis_style(ax1, ax2)
    merge_legends(ax1, ax2)
    save_figure(fig, output_dir / "stability_deltaT_courant")


def plot_pressure(df: pd.DataFrame, output_dir: Path) -> None:
    if df["p_rgh_initial_residual_last"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            df["time"],
            df["p_rgh_initial_residual_last"],
            label="Last p_rgh initial residual",
            linewidth=LINE_WIDTH,
        )
        if df["p_rgh_initial_residual_max"].notna().any():
            ax.plot(
                df["time"],
                df["p_rgh_initial_residual_max"],
                label="Max p_rgh initial residual",
                linewidth=LINE_WIDTH,
                linestyle="--",
            )
        ax.set_yscale("log")
        ax.set_ylabel("Residual [-]")
        ax.set_title("Pressure residual diagnostics")
        apply_common_style(ax)
        ax.legend(frameon=True)
        save_figure(fig, output_dir / "pressure_residuals")

    if df["p_min"].notna().any() or df["p_max"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        if df["p_min"].notna().any():
            ax.plot(df["time"], df["p_min"], label="p min", linewidth=LINE_WIDTH)
        if df["p_max"].notna().any():
            ax.plot(
                df["time"],
                df["p_max"],
                label="p max",
                linewidth=LINE_WIDTH,
                linestyle="--",
            )
        ax.set_yscale("log")
        ax.set_ylabel("Pressure")
        ax.set_title("Pressure extrema")
        apply_common_style(ax)
        ax.legend(frameon=True)
        save_figure(fig, output_dir / "pressure_extrema")


def plot_energy(df: pd.DataFrame, output_dir: Path, phases: List[str], energy_variable: Optional[str]) -> None:
    if energy_variable is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        plotted = False
        styles = phase_styles(phases, "tab10", 0.0, 1.0)

        for phase in phases:
            token = sanitize(phase)
            column = f"{sanitize(energy_variable)}_{token}_initial_residual_last"
            if column in df.columns and df[column].notna().any():
                style = styles[phase]
                ax.plot(
                    df["time"],
                    df[column],
                    label=f"{energy_variable}.{phase} initial residual",
                    linewidth=LINE_WIDTH,
                    color=style["color"],
                    linestyle=style["linestyle"],
                )
                plotted = True

        if plotted:
            ax.set_yscale("log")
            ax.set_ylabel("Residual [-]")
            ax.set_title(f"{energy_variable} residual diagnostics")
            apply_common_style(ax)
            ax.legend(frameon=True)
            save_figure(fig, output_dir / "energy_residuals")
        else:
            plt.close(fig)

    min_columns = [f"{sanitize(phase)}_t_min" for phase in phases]
    max_columns = [f"{sanitize(phase)}_t_max" for phase in phases]
    has_temperature = any(
        column in df.columns and df[column].notna().any()
        for column in min_columns + max_columns
    )

    if has_temperature:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        min_styles = phase_styles(phases, "Blues", 0.55, 0.95)
        max_styles = phase_styles(phases, "Reds", 0.55, 0.95)

        for phase in phases:
            token = sanitize(phase)
            column = f"{token}_t_min"
            if column in df.columns and df[column].notna().any():
                style = min_styles[phase]
                ax1.plot(
                    df["time"],
                    df[column],
                    label=f"{phase} T min",
                    linewidth=LINE_WIDTH,
                    color=style["color"],
                    linestyle=style["linestyle"],
                )

        for phase in phases:
            token = sanitize(phase)
            column = f"{token}_t_max"
            if column in df.columns and df[column].notna().any():
                style = max_styles[phase]
                ax2.plot(
                    df["time"],
                    df[column],
                    label=f"{phase} T max",
                    linewidth=LINE_WIDTH,
                    color=style["color"],
                    linestyle=style["linestyle"],
                )

        ax1.set_ylabel("Temperature minima [K]")
        ax2.set_ylabel("Temperature maxima [K]")
        ax1.set_title("Temperature extrema")
        apply_common_style(ax1)
        apply_dual_axis_style(ax1, ax2)
        merge_legends(ax1, ax2)
        save_figure(fig, output_dir / "energy_temperature_extrema")


def plot_volume_fractions(df: pd.DataFrame, output_dir: Path, phases: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    styles = phase_styles(phases, "tab20", 0.0, 1.0)

    for phase in phases:
        token = sanitize(phase)
        style = styles[phase]

        min_column = f"{token}_fraction_min"
        if min_column in df.columns and df[min_column].notna().any():
            ax.plot(
                df["time"],
                df[min_column],
                label=f"{phase} fraction min",
                linewidth=LINE_WIDTH,
                color=style["color"],
                linestyle=style["linestyle"],
            )
            plotted = True

        max_column = f"{token}_fraction_max"
        if max_column in df.columns and df[max_column].notna().any():
            ax.plot(
                df["time"],
                df[max_column],
                label=f"{phase} fraction max",
                linewidth=LINE_WIDTH,
                color=style["color"],
                linestyle="--",
            )
            plotted = True

    if "packing_proximity_max" in df.columns and df["packing_proximity_max"].notna().any():
        ax.plot(
            df["time"],
            df["packing_proximity_max"],
            label="Packing proximity max",
            linewidth=LINE_WIDTH,
            color="black",
            linestyle="-.",
        )
        plotted = True

    if plotted:
        ax.set_ylabel("Value [-]")
        ax.set_title("Volume-fraction diagnostics")
        apply_common_style(ax)
        ax.legend(frameon=True, ncol=2)
        save_figure(fig, output_dir / "fractions_diagnostics")
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    styles = phase_styles(phases, "tab10", 0.0, 1.0)

    for phase in phases:
        token = sanitize(phase)
        for suffix, line_style, label_suffix in [
            ("theta_avg", None, "theta avg"),
            ("theta_min", "--", "theta min"),
            ("theta_max", ":", "theta max"),
        ]:
            column = f"{token}_{suffix}"
            if column in df.columns and df[column].notna().any():
                style = styles[phase]
                ax.plot(
                    df["time"],
                    df[column],
                    label=f"{phase} {label_suffix}",
                    linewidth=LINE_WIDTH,
                    color=style["color"],
                    linestyle=line_style if line_style is not None else style["linestyle"],
                )
                plotted = True

    if plotted:
        ax.set_yscale("log")
        ax.set_ylabel("Theta [-]")
        ax.set_title("Phase theta diagnostics")
        apply_common_style(ax)
        ax.legend(frameon=True, ncol=2)
        save_figure(fig, output_dir / "fractions_theta")
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    styles = phase_styles(phases, "tab10", 0.0, 1.0)

    for phase in phases:
        token = sanitize(phase)
        column = f"theta_{token}_linear_iterations_last"
        if column in df.columns and df[column].notna().any():
            style = styles[phase]
            ax.plot(
                df["time"],
                df[column],
                label=f"Theta.{phase} solver iterations",
                linewidth=LINE_WIDTH,
                color=style["color"],
                linestyle=style["linestyle"],
            )
            plotted = True

    if plotted:
        ax.set_ylabel("Linear solver iterations [-]")
        ax.set_title("Theta equation solver iterations")
        apply_common_style(ax)
        ax.legend(frameon=True, ncol=2)
        save_figure(fig, output_dir / "theta_solver_iterations")
    else:
        plt.close(fig)


def plot_pimple(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    if "max_pimple_iteration" in df.columns and df["max_pimple_iteration"].notna().any():
        ax.plot(
            df["time"],
            df["max_pimple_iteration"],
            label="Max PIMPLE iteration",
            linewidth=LINE_WIDTH,
        )
        plotted = True

    if "effective_pimple_iterations" in df.columns and df["effective_pimple_iterations"].notna().any():
        ax.plot(
            df["time"],
            df["effective_pimple_iterations"],
            label="Effective PIMPLE iterations",
            linewidth=LINE_WIDTH,
            linestyle="--",
        )
        plotted = True

    if plotted:
        ax.set_ylabel("Count [-]")
        ax.set_title("PIMPLE iteration diagnostics")
        apply_common_style(ax)
        ax.legend(frameon=True)
        save_figure(fig, output_dir / "pimple_iterations")
    else:
        plt.close(fig)

    if "last_pimple_residual_ratio" in df.columns and df["last_pimple_residual_ratio"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            df["time"],
            df["last_pimple_residual_ratio"],
            label="Residual ratio",
            linewidth=LINE_WIDTH,
        )
        ax.set_ylabel("Residual ratio [-]")
        ax.set_title("PIMPLE residual ratio")
        apply_common_style(ax)
        ax.legend(frameon=True)
        save_figure(fig, output_dir / "pimple_residual_ratio")


def make_all_plots(df: pd.DataFrame, output_dir: Path, phases: List[str], energy_variable: Optional[str]) -> None:
    plot_performance(df, output_dir)
    plot_stability(df, output_dir)
    plot_pressure(df, output_dir)
    plot_energy(df, output_dir, phases, energy_variable)
    plot_volume_fractions(df, output_dir, phases)
    plot_pimple(df, output_dir)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse an OpenFOAM log file, extract diagnostics, save CSV, "
            "and create PNG and PDF figures."
        )
    )
    parser.add_argument("logfile", type=Path, help="Path to the OpenFOAM log file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("log_analysis"),
        help="Directory where CSV and figures will be saved.",
    )
    parser.add_argument(
        "--csv-name",
        default="log_diagnostics.csv",
        help="Name of the output CSV file.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    log_path = args.logfile
    output_dir = args.output_dir

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Author: {AUTHOR} ({AFFILIATION})")
    print(f"Reading log file: {log_path}")
    print("Parsing log in streaming mode...")

    dataframe, phases, energy_variable = parse_log(log_path)

    csv_path = output_dir / args.csv_name
    dataframe.to_csv(csv_path, index=False)

    print(f"Parsed {len(dataframe)} physical time steps.")
    print(f"Detected phases: {', '.join(phases) if phases else 'none'}")
    print(f"Detected energy variable: {energy_variable if energy_variable else 'none'}")
    print(f"CSV file written to: {csv_path}")
    print("Generating figures...")

    make_all_plots(dataframe, output_dir, phases, energy_variable)

    print(f"Analysis completed. Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
