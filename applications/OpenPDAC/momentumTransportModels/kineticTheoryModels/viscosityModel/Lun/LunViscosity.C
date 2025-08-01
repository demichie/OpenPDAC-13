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

#include "LunViscosity.H"
#include "mathematicalConstants.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace kineticTheoryModels
{
namespace viscosityModels
{
    defineTypeNameAndDebug(Lun, 0);
    addToRunTimeSelectionTable(viscosityModel, Lun, dictionary);
}
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::kineticTheoryModels::viscosityModels::Lun::Lun
(
    const dictionary& coeffDict
)
:
    viscosityModel(coeffDict),
    alfa_
    (
        "alfa",
        dimless,
        coeffDict.lookupOrDefault<scalar>("alfa", 1.6)
    )
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::kineticTheoryModels::viscosityModels::Lun::~Lun()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField>
Foam::kineticTheoryModels::viscosityModels::Lun::nu
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
    const scalar sqrtPi = sqrt(constant::mathematical::pi);
    const scalar Pi = constant::mathematical::pi;

    const dimensionedScalar eta = 0.5*(1.0 + e);
    
    const volScalarField mu = 5.0/96.0*rho1*da*sqrt(Theta)*sqrtPi; 
    
    const volScalarField mu_b = 256.0/(5.0*Pi)*mu*alpha1*alpha1*g0;
    
    
    // Added correction 
    const volScalarField muStar = ( rho1*alpha1*g0*Theta*mu ) /
                                  ( rho1*alpha1*g0*(Theta+ThetaSmall) + 
                                    (2*beta*mu)/(rho1*alpha1) );  
    
                                        
    const volScalarField mu_i = (2+alfa_)/3.0*( muStar / (g0*eta*(2-eta))*
                                (1+8/5*eta*alpha1*g0)*(1+8/5*eta*(3*eta-2)*alpha1*g0)+
                                3/5*eta*mu_b );

    return volScalarField::New
    (
        IOobject::groupName
        (
            Foam::typedName<viscosityModel>("nu"),
            Theta.group()
        ),
        mu_i/rho1
    );
}

Foam::tmp<Foam::volScalarField>
Foam::kineticTheoryModels::viscosityModels::Lun::nu
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
    const scalar sqrtPi = sqrt(constant::mathematical::pi);
    const scalar Pi = constant::mathematical::pi;

    // Eq. B12 MFIX2012
    const dimensionedScalar eta = 0.5*(1.0 + e);
    // Eq. B6 MFIX2012    
    const volScalarField mu = 5.0/96.0*rho1*da*sqrt(Theta)*sqrtPi; 
    // Eq. B7 MFIX2012
    const volScalarField mu_b = 256.0/(5.0*Pi)*mu*alpha1*sumAlphaGs0;
    // Eq. B5 MFIX2012
    const volScalarField muStar = ( rho1*alpha1*g0*Theta*mu ) /
                                  ( rho1*sumAlphaGs0*(Theta+ThetaSmall) + 
                                    (2*beta*mu)/(rho1*alpha1) );  
    // Eq. B4 MFIX2012
    const volScalarField mu_i = (2+alfa_)/3.0*( muStar / (g0*eta*(2-eta))*
                                (1+8/5*eta*sumAlphaGs0)*(1+8/5*eta*(3*eta-2)*sumAlphaGs0)+
                                3/5*eta*mu_b );

    return volScalarField::New
    (
        IOobject::groupName
        (
            Foam::typedName<viscosityModel>("nu"),
            Theta.group()
        ),
        mu_i/rho1
    );
}

// ************************************************************************* //
