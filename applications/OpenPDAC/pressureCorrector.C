/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2025 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenPDAC.
    This file was derived from the multiphaseEuler solver in OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
    Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program. If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "OpenPDAC.H"

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::OpenPDAC::pressureCorrector()
{

    if (forceFinalPimpleIter_ && !pimple.finalIter())
    {
        const DynamicList<SolverPerformance<scalar>>& spConst =
            Residuals<scalar>::field(mesh, "p_rgh");

        DynamicList<SolverPerformance<scalar>>& sp =
            const_cast<DynamicList<SolverPerformance<scalar>>&>(spConst);

        Info << "Bypass p_rgh residual history size before append = "
             << sp.size() << endl;

        sp.append(SolverPerformance<scalar>("bypass",
                                            "p_rgh",
                                            prevPimpleInitialResidual_,
                                            prevPimpleInitialResidual_,
                                            0,
                                            false,
                                            false));

        Info << "Bypass p_rgh residual history size after append = "
             << sp.size()
             << ", copied initial residual = " << prevPimpleInitialResidual_
             << endl;

        return;
    }

    if (faceMomentum)
    {
        facePressureCorrector();
    }
    else
    {
        cellPressureCorrector();
    }

    fluid_.correctKinematics();
}


// ************************************************************************* //
