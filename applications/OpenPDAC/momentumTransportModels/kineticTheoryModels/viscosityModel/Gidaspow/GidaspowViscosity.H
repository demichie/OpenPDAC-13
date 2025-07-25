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

Class
    Foam::kineticTheoryModels::viscosityModels::Gidaspow

Description

SourceFiles
    GidaspowViscosity.C

\*---------------------------------------------------------------------------*/

#ifndef GidaspowViscosity_H
#define GidaspowViscosity_H

#include "kineticTheoryViscosityModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace kineticTheoryModels
{
namespace viscosityModels
{

/*---------------------------------------------------------------------------*\
                          Class Gidaspow Declaration
\*---------------------------------------------------------------------------*/

class Gidaspow
:
    public viscosityModel
{
    // Private Data

        dictionary coeffDict_;

        //- Angle of internal friction
        dimensionedScalar alfa_;
        
public:

    //- Runtime type information
    TypeName("Gidaspow");

    // Constructors

        //- Construct from the coefficients dictionary
        Gidaspow(const dictionary& coeffDict);


    //- Destructor
    virtual ~Gidaspow();


    // Member Functions

        tmp<volScalarField> nu
        (
            const volScalarField& alpha1,
            const volScalarField& Theta,
            const dimensionedScalar& ThetaSmall,
            const volScalarField& g0,
            const volScalarField& beta,
            const volScalarField& rho1,
            const volScalarField& da,
            const dimensionedScalar& e
        ) const;
        
        tmp<volScalarField> nu
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
        ) const;        
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace viscosityModels
} // End namespace kineticTheoryModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
