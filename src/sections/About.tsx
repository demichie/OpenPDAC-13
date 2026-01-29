import { Layers, Wind, Box, Zap } from 'lucide-react';

const features = [
  {
    icon: Layers,
    title: 'Multiphase Modeling',
    description:
      'Advanced kinetic theory equations modified to handle multiple dispersed solid phases in gas-solid mixtures.',
  },
  {
    icon: Wind,
    title: 'Lagrangian Tracking',
    description:
      'One-way coupled Lagrangian library for tracking particles within the gas-solid mixture flow field.',
  },
  {
    icon: Box,
    title: 'Hydrostatic Initialization',
    description:
      'Automatic initialization of hydrostatic pressure profiles for large domain simulations with proper inflow/outflow boundary conditions.',
  },
  {
    icon: Zap,
    title: 'OpenFOAM Integration',
    description:
      'Built on the robust multiphaseEuler module, ensuring compatibility with OpenFOAM 13 and its extensive toolchain.',
  },
];

export function About() {
  return (
    <section id="about" className="relative py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <span className="text-orange-500 text-sm font-semibold tracking-wider uppercase">
            About OpenPDAC
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mt-4 mb-6">
            Powerful CFD for Complex
            <span className="text-gradient"> Multiphase Flows</span>
          </h2>
          <p className="text-gray-400 text-lg max-w-3xl mx-auto">
            OpenPDAC is an open-source computational fluid dynamics solver designed specifically 
            for simulating pyroclastic density currents and granular flows. Built as an extension 
            to OpenFOAM's multiphaseEuler module, it provides researchers and engineers with 
            advanced capabilities for modeling complex gas-solid interactions.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group p-6 rounded-2xl glass hover:bg-white/10 transition-all duration-300 hover:-translate-y-1"
            >
              <div className="w-12 h-12 rounded-xl bg-orange-500/10 flex items-center justify-center mb-4 group-hover:bg-orange-500/20 transition-colors">
                <feature.icon className="w-6 h-6 text-orange-500" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Code Block */}
        <div className="mt-16 max-w-4xl mx-auto">
          <div className="rounded-2xl overflow-hidden glass">
            <div className="flex items-center gap-2 px-4 py-3 bg-white/5 border-b border-white/10">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span className="ml-4 text-sm text-gray-500">OpenPDAC Solver</span>
            </div>
            <div className="p-6 overflow-x-auto">
              <pre className="text-sm text-gray-300 font-mono">
                <code>{`// OpenPDAC: Multiphase granular flow solver
// Based on OpenFOAM multiphaseEuler module

Solver features:
  - Multiple dispersed solid phases
  - Kinetic theory for granular flows
  - Lagrangian particle tracking (one-way coupling)
  - Hydrostatic pressure initialization
  - Large domain simulation support

Applications:
  - Pyroclastic density currents
  - Fluidized beds
  - Particle-laden flows
  - Granular material transport`}</code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
