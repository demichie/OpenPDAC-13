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
#include "fvmSup.H"

// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

void Foam::solvers::OpenPDAC::cellMomentumPredictor()
{
    Info<< "Constructing momentum equations" << endl;

    autoPtr<HashPtrTable<fvVectorMatrix>> popBalMomentumTransferPtr =
        populationBalanceSystem_.momentumTransfer();
    HashPtrTable<fvVectorMatrix>& popBalMomentumTransfer =
        popBalMomentumTransferPtr();

    forAll(movingPhases, movingPhasei)
    {
        phaseModel& phase = movingPhases_[movingPhasei];

        const volScalarField& alpha = phase;
        const volScalarField& rho = phase.rho();
        volVectorField& U = phase.URef();

        UEqns.set
        (
            phase.index(),
            new fvVectorMatrix
            (
                phase.UEqn()
             ==
                *popBalMomentumTransfer[phase.name()]
              + fvModels().source(alpha, rho, U)
            )
        );

        UEqns[phase.index()].relax();
        fvConstraints().constrain(UEqns[phase.index()]);
        U.correctBoundaryConditions();
        fvConstraints().constrain(U);
    }
}


void Foam::solvers::OpenPDAC::faceMomentumPredictor()
{
    Info<< "Constructing face momentum equations" << endl;

    autoPtr<HashPtrTable<fvVectorMatrix>> popBalMomentumTransferPtr =
        populationBalanceSystem_.momentumTransferf();
    HashPtrTable<fvVectorMatrix>& popBalMomentumTransfer =
        popBalMomentumTransferPtr();

    forAll(movingPhases, movingPhasei)
    {
        phaseModel& phase = movingPhases_[movingPhasei];

        const volScalarField& alpha = phase;
        const volScalarField& rho = phase.rho();
        volVectorField& U = phase.URef();

        UEqns.set
        (
            phase.index(),
            new fvVectorMatrix
            (
                phase.UfEqn()
             ==
                *popBalMomentumTransfer[phase.name()]
              + fvModels().source(alpha, rho, U)
            )
        );

        UEqns[phase.index()].relax();
        fvConstraints().constrain(UEqns[phase.index()]);
        U.correctBoundaryConditions();
        fvConstraints().constrain(U);
    }
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::OpenPDAC::momentumPredictor()
{

    if (forceFinalPimpleIter_ && !pimple.finalIter())
    {
        // Non fare nulla, la funzione è stata svuotata in questa iterazione.
        return;
    }

    UEqns.setSize(phases.size());

    if (faceMomentum)
    {
        faceMomentumPredictor();
    }
    else
    {
        cellMomentumPredictor();
    }
}


// ************************************************************************* //
