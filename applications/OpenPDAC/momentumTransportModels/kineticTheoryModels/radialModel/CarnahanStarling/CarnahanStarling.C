/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2024 OpenFOAM Foundation
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

#include "CarnahanStarling.H"
#include "addToRunTimeSelectionTable.H"
#include "phaseSystem.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace kineticTheoryModels
{
namespace radialModels
{
    defineTypeNameAndDebug(CarnahanStarling, 0);

    addToRunTimeSelectionTable
    (
        radialModel,
        CarnahanStarling,
        dictionary
    );
}
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::kineticTheoryModels::radialModels::CarnahanStarling::CarnahanStarling
(
    const dictionary& coeffDict
)
:
    radialModel(coeffDict)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::kineticTheoryModels::radialModels::CarnahanStarling::~CarnahanStarling()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::PtrList<Foam::volScalarField>
Foam::kineticTheoryModels::radialModels::CarnahanStarling::g0
(
    const phaseModel& phasei,
    const phaseModel& continuousPhase,
    const dimensionedScalar& alphaMinFriction,
    const volScalarField& alphasMax
) const
{
    const volScalarField& alphai = phasei;
    const label& indexi = phasei.index();
    const phaseSystem& fluid = phasei.fluid();

    PtrList<volScalarField> g0_im(fluid.phases().size());

    volScalarField alphas = alphai;
    volScalarField eta2 = alphai / phasei.d();

    forAll(fluid.phases(), phaseIdx)
    {
        const phaseModel& phase = fluid.phases()[phaseIdx];
        if ((&phase != &continuousPhase) and !(phaseIdx==indexi))
        {
            alphas += phase;
            eta2 += phase/phase.d();
        }
    }

    const volScalarField denominatorTerm = 1.0 - alphas;

    forAll(g0_im, iter)
    {
        const phaseModel& phasem = fluid.phases()[iter];

        if (&phasem != &continuousPhase)
        {
            const volScalarField di = phasei.d();
            const volScalarField dm = phasem.d();
            volScalarField term_d = di*dm / (di + dm);

            g0_im.set
            (
                iter,
                volScalarField
                (
                    "g0_im" + phasei.name() + "_" + phasem.name(),
                    1.0/denominatorTerm
                  + 3.0*term_d*eta2/sqr(denominatorTerm)
                  + 2.0*sqr(term_d)*sqr(eta2)/pow3(denominatorTerm)
                )
            );
        }
    }

    return g0_im;
}


Foam::PtrList<Foam::volScalarField>
Foam::kineticTheoryModels::radialModels::CarnahanStarling::g0prime
(
    const phaseModel& phasei,
    const phaseModel& continuousPhase,
    const dimensionedScalar& alphaMinFriction,
    const volScalarField& alphasMax
) const
{
    const volScalarField& alphai = phasei;
    const label& indexi = phasei.index();
    const phaseSystem& fluid = phasei.fluid();

    PtrList<volScalarField> g0prime_im(fluid.phases().size());

    volScalarField alphas = alphai;
    volScalarField eta2 = alphai / phasei.d();

    forAll(fluid.phases(), phaseIdx)
    {
        const phaseModel& phase = fluid.phases()[phaseIdx];

        if ((&phase != &continuousPhase) and !(phaseIdx==indexi))
        {
            alphas += phase;
            eta2 += phase / phase.d();
        }
    }

    volScalarField mask = Foam::sign(Foam::pos(alphas - SMALL));
    volScalarField dEta2dAlphas = mask * (eta2 / (alphas + ROOTVSMALL));
    const volScalarField denominatorTerm = 1.0 - alphas;

    forAll(g0prime_im, iter)
    {
        const phaseModel& phasem = fluid.phases()[iter];

        if (&phasem != &continuousPhase)
        {
            const volScalarField di = phasei.d();
            const volScalarField dm = phasem.d();
            // *** FINAL FIX: Materialize tmp<> into a persistent volScalarField ***
            volScalarField term_d = di*dm / (di + dm);

            tmp<volScalarField> dT1 = 1.0/sqr(denominatorTerm);
            tmp<volScalarField> dT2 = 3.0*term_d*
            (
                dEta2dAlphas/sqr(denominatorTerm)
              + 2.0*eta2/pow3(denominatorTerm)
            );
            tmp<volScalarField> dT3 = 2.0*sqr(term_d)*
            (
                2.0*eta2*dEta2dAlphas/pow3(denominatorTerm)
              + 3.0*sqr(eta2)/pow(denominatorTerm, 4.0)
            );

            g0prime_im.set
            (
                iter,
                volScalarField
                (
                    "g0prime_im" + phasei.name() + "_" + phasem.name(),
                    dT1 + dT2 + dT3
                )
            );
        }
    }

    return g0prime_im;
}


// ************************************************************************* //
