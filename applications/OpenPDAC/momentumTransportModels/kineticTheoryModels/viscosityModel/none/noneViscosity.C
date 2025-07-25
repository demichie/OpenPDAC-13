/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2023 OpenFOAM Foundation
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

#include "noneViscosity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace kineticTheoryModels
{
namespace viscosityModels
{
    defineTypeNameAndDebug(none, 0);
    addToRunTimeSelectionTable(viscosityModel, none, dictionary);
}
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::kineticTheoryModels::viscosityModels::none::none
(
    const dictionary& coeffDict
)
:
    viscosityModel(coeffDict)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::kineticTheoryModels::viscosityModels::none::~none()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField>
Foam::kineticTheoryModels::viscosityModels::none::nu
(
    const volScalarField& alpha1,
    const volScalarField& Theta,
    const dimensionedScalar& ThetaSmall,
    const volScalarField& g0,
    const volScalarField& beta,
    const volScalarField& rho1,
    const volScalarField& da,
    const dimensionedScalar& e
) const
{
    return volScalarField::New
    (
        IOobject::groupName
        (
            Foam::typedName<viscosityModel>("nu"),
            Theta.group()
        ),
        alpha1.mesh(),
        dimensionedScalar(dimArea/dimTime, 0)
    );
}

Foam::tmp<Foam::volScalarField>
Foam::kineticTheoryModels::viscosityModels::none::nu
(
    const volScalarField& alpha1,
    const volScalarField& Theta,
    const dimensionedScalar& ThetaSmall,
    const volScalarField& g0,
    const volScalarField& sumAlphaGs0,
    const volScalarField& beta,
    const volScalarField& rho1,
    const volScalarField& da,
    const dimensionedScalar& e
) const
{
    return volScalarField::New
    (
        IOobject::groupName
        (
            Foam::typedName<viscosityModel>("nu"),
            Theta.group()
        ),
        alpha1.mesh(),
        dimensionedScalar(dimArea/dimTime, 0)
    );
}

// ************************************************************************* //
