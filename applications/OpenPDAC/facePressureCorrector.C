/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2025 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "OpenPDAC.H"
#include "constrainHbyA.H"
#include "constrainPressure.H"
#include "findRefCell.H"
#include "fvcDdt.H"
#include "fvcDiv.H"
#include "fvcSup.H"
#include "fvcSnGrad.H"
#include "fvmDdt.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "fvmSup.H"
#include "fvcFlux.H"
#include "fvcMeshPhi.H"
#include "fvcReconstruct.H"

// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

void Foam::solvers::OpenPDAC::facePressureCorrector()
{
    volScalarField& p(p_);
    volScalarField& p_rgh = p_rgh_;

    // Face volume fractions
    PtrList<surfaceScalarField> alphafs(phases.size());
    forAll(phases, phasei)
    {
        const phaseModel& phase = phases[phasei];
        const volScalarField& alpha = phase;

        alphafs.set(phasei, fvc::interpolate(alpha).ptr());
        alphafs[phasei].rename("pEqn" + alphafs[phasei].name());
    }

    // Diagonal coefficients
    rAs.clear();
    if (fluid.implicitPhasePressure())
    {
        rAs.setSize(phases.size());
    }

    PtrList<surfaceScalarField> HVmfs(movingPhases.size());
    PtrList<PtrList<surfaceScalarField>> invADVfs;
    {
        PtrList<surfaceScalarField> Afs(movingPhases.size());

        forAll(movingPhases, movingPhasei)
        {
            const phaseModel& phase = movingPhases[movingPhasei];
            const volScalarField& alpha = phase;

            const volScalarField A
            (
                byDt
                (
                    max(alpha.oldTime(), phase.residualAlpha())
                   *phase.rho().oldTime()
                )
              + UEqns[phase.index()].A()
            );

            if (fluid.implicitPhasePressure())
            {
                rAs.set
                (
                    phase.index(),
                    new volScalarField
                    (
                        IOobject::groupName("rA", phase.name()),
                        1/A
                    )
                );
            }

            Afs.set
            (
                movingPhasei,
                new surfaceScalarField
                (
                    IOobject::groupName("rAf", phase.name()),
                    fvc::interpolate(A)
                )
            );
        }

        invADVfs = momentumTransferSystem_.invADVfs(Afs, HVmfs);
    }

    volScalarField rho("rho", fluid.rho());

    // Phase diagonal coefficients
    PtrList<surfaceScalarField> alphaByADfs;
    PtrList<surfaceScalarField> FgByADfs;
    {
        // Explicit force fluxes
        PtrList<surfaceScalarField> Ffs(momentumTransferSystem_.Ffs());

        const surfaceScalarField ghSnGradRho
        (
            "ghSnGradRho",
            buoyancy.ghf*fvc::snGrad(rho)*mesh.magSf()
        );

        UPtrList<surfaceScalarField> movingAlphafs(movingPhases.size());
        PtrList<surfaceScalarField> Fgfs(movingPhases.size());

        forAll(movingPhases, movingPhasei)
        {
            const phaseModel& phase = movingPhases[movingPhasei];

            movingAlphafs.set(movingPhasei, &alphafs[phase.index()]);

            Fgfs.set
            (
                movingPhasei,
                Ffs[phase.index()]
              + alphafs[phase.index()]
               *(
                   ghSnGradRho
                 - fluid.surfaceTension(phase)*mesh.magSf()
                )
              - max(alphafs[phase.index()], phase.residualAlpha())
               *fvc::interpolate(phase.rho() - rho)*(buoyancy.g & mesh.Sf())
            );
        }

        alphaByADfs = invADVfs & movingAlphafs;
        FgByADfs = invADVfs & Fgfs;
    }


    // Mass transfer rates
    PtrList<volScalarField::Internal> dmdts(populationBalanceSystem_.dmdts());

    bool checkInnerResidual(false);
    scalar r0(0.0);
    scalar r0Inner(0.0);
    // --- Pressure corrector loop
    while (pimple.correct())
    {
        // Correct fixed-flux BCs to be consistent with the velocity BCs
        fluid_.correctBoundaryFlux();

        // Predicted fluxes for each phase
        PtrList<surfaceScalarField> phiHbyADs;
        {
            PtrList<surfaceScalarField> phiHs(movingPhases.size());

            forAll(movingPhases, movingPhasei)
            {
                const phaseModel& phase = movingPhases[movingPhasei];
                const volScalarField& alpha = phase;

                phiHs.set
                (
                    movingPhasei,
                    (
                       fvc::interpolate
                       (
                           max(alpha.oldTime(), phase.residualAlpha())
                          *phase.rho().oldTime()
                       )
                      *byDt
                       (
                           phase.Uf().valid()
                         ? (mesh.Sf() & phase.Uf()().oldTime())
                         : MRF.absolute(phase.phi()().oldTime())
                       )
                     + fvc::flux(UEqns[phase.index()].H())
                    )
                );

                if (HVmfs.set(movingPhasei))
                {
                    phiHs[movingPhasei] += HVmfs[movingPhasei];
                }
            }

            phiHbyADs = invADVfs & phiHs;
        }

        forAll(movingPhases, movingPhasei)
        {
            const phaseModel& phase = movingPhases[movingPhasei];

            constrainPhiHbyA(phiHbyADs[movingPhasei], phase.U(), p_rgh);

            phiHbyADs[movingPhasei] -= FgByADfs[movingPhasei];
        }

        // Total predicted flux
        surfaceScalarField phiHbyA
        (
            IOobject
            (
                "phiHbyA",
                runTime.name(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(dimVolumetricFlux, 0)
        );

        forAll(movingPhases, movingPhasei)
        {
            const phaseModel& phase = movingPhases[movingPhasei];

            phiHbyA += alphafs[phase.index()]*phiHbyADs[movingPhasei];
        }

        MRF.makeRelative(phiHbyA);
        fvc::makeRelative(phiHbyA, movingPhases[0].U());

        // Construct pressure "diffusivity"
        surfaceScalarField rAf
        (
            IOobject
            (
                "rAf",
                runTime.name(),
                mesh
            ),
            mesh,
            dimensionedScalar(dimensionSet(-1, 3, 1, 0, 0), 0)
        );

        forAll(movingPhases, movingPhasei)
        {
            const phaseModel& phase = movingPhases[movingPhasei];

            rAf += alphafs[phase.index()]*alphaByADfs[movingPhasei];
        }

        
        // Update the fixedFluxPressure BCs to ensure flux consistency
        {
            surfaceScalarField::Boundary phib
            (
                surfaceScalarField::Internal::null(),
                phi.boundaryField()
            );
            phib = 0;

            forAll(movingPhases, movingPhasei)
            {
                phaseModel& phase = movingPhases_[movingPhasei];

                phib +=
                    alphafs[phase.index()].boundaryField()
                   *phase.phi()().boundaryField();
            }

            setSnGrad<fixedFluxPressureFvPatchScalarField>
            (
                p_rgh.boundaryFieldRef(),
                (
                    phiHbyA.boundaryField() - phib
                )/(mesh.magSf().boundaryField()*rAf.boundaryField())
            );
        }
        

        // Compressible pressure equations
        PtrList<fvScalarMatrix> pEqnComps(compressibilityEqns(dmdts));

        // Cache p prior to solve for density update
        volScalarField p_rgh_0(p_rgh);

        bool checkResidual(false);
        // Iterate over the pressure equation to correct for non-orthogonality
        while (pimple.correctNonOrthogonal())
        {
            // Update the fixedFluxPressure BCs to ensure flux consistency
            {
                surfaceScalarField::Boundary phib
                (
                    surfaceScalarField::Internal::null(),
                    phi.boundaryField()
                );
                phib = 0;

                forAll(movingPhases, movingPhasei)
                {
                    phaseModel& phase = movingPhases_[movingPhasei];

                    phib +=
                        alphafs[phase.index()].boundaryField()
                       *phase.phi()().boundaryField();
                }

                setSnGrad<fixedFluxPressureFvPatchScalarField>
                (
                    p_rgh.boundaryFieldRef(),
                    (
                        phiHbyA.boundaryField() - phib
                    )/(mesh.magSf().boundaryField()*rAf.boundaryField())
                );
            }
        
            // Construct the transport part of the pressure equation
            fvScalarMatrix pEqnIncomp
            (
                fvc::div(phiHbyA)
              - fvm::laplacian(rAf, p_rgh)
            );

            if ((!checkResidual) && (!checkInnerResidual))
            {      
                // Solve
                {
                    fvScalarMatrix pEqn(pEqnIncomp);

                    forAll(phases, phasei)
                    {
                        pEqn += pEqnComps[phasei];
                    }

                    if (fluid.incompressible())
                    {
                        pEqn.setReference
                        (
                            pressureReference.refCell(),
                            pressureReference.refValue()
                        );
                    }

                    fvConstraints().constrain(pEqn);

                    pEqn.solve();

                    const DynamicList<SolverPerformance<scalar>> sp
                    (
                        Residuals<scalar>::field(mesh, "p_rgh")
                    );
                    
                    label n = sp.size();
                    r0 = cmptMax(sp[n-1].initialResidual());
                    
                    if (pimple.firstNonOrthogonalIter() && pimple.firstPisoIter()  && !pimple.finalIter())
                    {
                        
                        Info << "PIMPLE iter: " << pimpleIter
                             << ", Initial p_rgh residual: " << r0 << endl;

                        // Resetta lo stato alla prima iterazione PIMPLE (corr=1)
                        if (pimpleIter == 1)
                        {
                           prevPimpleInitialResidual_ = r0;
                        }
                        // Applica la logica di controllo nelle iterazioni successive
                        else if (
                            pimpleIter > pimple.nCorr()/2
                         && !pimple.finalIter()
                        )
                        {
                            Info << "residual ratio " << r0/prevPimpleInitialResidual_ << endl;
                            // Se il residuo non è diminuito, imposta il flag
                            if (r0 > prevPimpleInitialResidual_ * residualRatio)
                            {
                                if (ratioFirstCheck)
                                {
                                    Info << "  --> PIMPLE: Initial residual increased. "
                                         << "Scheduling jump to final iter." << endl;
                                    forceFinalPimpleIter_ = true;
                                }
                                else
                                {
                                    ratioFirstCheck = true;
                                }
                            }
                            else
                            {
                                ratioFirstCheck = false;
                            }


                            prevPimpleInitialResidual_ = r0;
                        }
                        else
                        {
                            // Aggiorna comunque il residuo per le iterazioni iniziali
                            prevPimpleInitialResidual_ = r0;
                        }
                    }
                    
                    // Info << " p_rgh initial residual " << r0 << endl;
                    // Info << checkResidual << endl;  
                    if ( r0 <= nonOrthogonalResidual ) 
                    {
                        checkResidual = true;
                        Info << "NonOrthogonal convergence "
                             << checkResidual << endl;
                    
                    }  
                }              
            }

            if (pimple.firstNonOrthogonalIter())
            {
                r0Inner = r0;
            }

            // Correct fluxes and velocities on last non-orthogonal iteration
            if (pimple.finalNonOrthogonalIter())
            {

                phi_ = phiHbyA + pEqnIncomp.flux();

                surfaceScalarField mSfGradp("mSfGradp", pEqnIncomp.flux()/rAf);

                forAll(movingPhases, movingPhasei)
                {
                    phaseModel& phase = movingPhases_[movingPhasei];
                    const label phasei = phase.index();

                    phase.phiRef() =
                        phiHbyADs[movingPhasei]
                      + alphaByADfs[movingPhasei]*mSfGradp;

                    // Set the phase dilatation rate
                    phase.divU(-pEqnComps[phasei] & p_rgh);
                }

                forAll(movingPhases, movingPhasei)
                {
                    phaseModel& phase = movingPhases_[movingPhasei];

                    MRF.makeRelative(phase.phiRef());
                    fvc::makeRelative(phase.phiRef(), phase.U());

                    phase.URef() = fvc::reconstruct
                    (
                        fvc::absolute(MRF.absolute(phase.phi()), phase.U())
                    );

                    phase.URef().correctBoundaryConditions();
                    phase.correctUf();
                    fvConstraints().constrain(phase.URef());
                }
            }
        }

	if ( !checkInnerResidual )
        { 

            // Update and limit the static pressure
            p = p_rgh + rho*buoyancy.gh;
            // p = p_rgh + rho*buoyancy.gh + buoyancy.pRef;
            fvConstraints().constrain(p);

            // Account for static pressure reference
            if (p_rgh.needReference() && fluid.incompressible())
            {
                p += dimensionedScalar
                (
                    "p",
                    p.dimensions(),
                    pressureReference.refValue()
                  - getRefCellValue(p, pressureReference.refCell())
                );
            }

            Info<< "p, min, max = " << min(p).value() << " " << max(p).value() << endl;
        
        
            if (lowPressureTimestepCorrection)
            {
                p_ratio = max(0.01,min(p).value() /p.weightedAverage(mesh_.V()).value());
                Info<< "p_ratio = " << p_ratio << endl;
            }
                
            // Limit p_rgh
            p_rgh = p - rho*buoyancy.gh;
            // p_rgh = p - rho*buoyancy.gh - buoyancy.pRef;

            // Update densities from change in p_rgh
            forAll(phases, phasei)
            {
                phaseModel& phase = phases_[phasei];
                if (!phase.incompressible())
                {
                    phase.rho() += phase.fluidThermo().psi()*(p_rgh - p_rgh_0);
                }
            }

            // Correct p_rgh for consistency with p and the updated densities
            rho = fluid.rho();
            p_rgh = p - rho*buoyancy.gh;
            // p_rgh = p - rho*buoyancy.gh - buoyancy.pRef;

            surfaceScalarField::Boundary phib
            (
                surfaceScalarField::Internal::null(),
                phi.boundaryField()
            );
            phib = 0;

            forAll(movingPhases, movingPhasei)
            {
                phaseModel& phase = movingPhases_[movingPhasei];

                phib +=
                    alphafs[phase.index()].boundaryField()
                   *phase.phi()().boundaryField();
            }
            
            setSnGrad<fixedFluxPressureFvPatchScalarField>
            (
                p_rgh.boundaryFieldRef(),
                (
                    phiHbyA.boundaryField() - phib
                )/(mesh.magSf().boundaryField()*rAf.boundaryField())
            );            
            p_rgh.correctBoundaryConditions();
        }

        if ( ( r0Inner <= innerResidual ) && ( !checkInnerResidual ) )
        {
            checkInnerResidual = true;
            Info << "Innerloop convergence "
            << checkInnerResidual << endl;
        } 

    }

    UEqns.clear();

}


// ************************************************************************* //
