/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2025 OpenFOAM Foundation
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

#include "WenYu.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace dragModels
{
    defineTypeNameAndDebug(WenYu, 0);
    addToRunTimeSelectionTable(dragModel, WenYu, dictionary);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::dragModels::WenYu::WenYu
(
    const dictionary& dict,
    const phaseInterface& interface,
    const bool registerObject
)
:
    dispersedDragModel(dict, interface, registerObject)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::dragModels::WenYu::~WenYu()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField> Foam::dragModels::WenYu::CdRe() const
{
    const volScalarField alpha2
    (
        max(1.0 - interface_.dispersed(), interface_.continuous().residualAlpha())
    );

    const volScalarField Res(alpha2*interface_.Re());

    /*
    Info << "Re min, max = " << min(interface_.Re()).value()  << " " << max(interface_.Re()).value() << endl;

    Info << "nu min, max = " << min(interface_.continuous().fluidThermo().nu()).value()  << " " << max(interface_.continuous().fluidThermo().nu()).value() << endl;

    Info << "rho min, max = " << min(interface_.continuous().fluidThermo().rho()).value()  << " " << max(interface_.continuous().fluidThermo().rho()).value() << endl;

    Info << "Res min, max = " << min(Res).value()  << " " << max(Res).value() << endl;
    */

    const volScalarField CdsRes
    (
        neg(Res - 1000)*24*(1.0 + 0.15*pow(Res, 0.687))
      + pos0(Res - 1000)*0.44*Res
    );

    return CdsRes*pow(alpha2, -2.65);
}


// ************************************************************************* //
