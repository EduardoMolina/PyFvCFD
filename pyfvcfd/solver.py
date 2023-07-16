import taichi as ti
import numpy as np
import pickle

from pyfvcfd import linInterp, greenGaussGrad, integrateFluxes3D
from pyfvcfd import Fluid, idealGasP, idealGasRho, calPerfectEnergy, mag, calPerfectT
from pyfvcfd import centralJST

@ti.func
def decodePrimitives_(state:ti.template(), fluid: Fluid, prim: ti.template()):

    ## Velocity ##
    # Ux = xMom/rho
    prim[2] = state[1]/state[0]
    # Uy = yMom/rho
    prim[3] = state[2]/state[0]
    # Uz = zMom/rho
    prim[4] = state[3]/state[0]

    ## Energy ##
    # e = (eV2/rho) - (mag((Ux, Uy, Uz))^2)/2
    e = (state[4]/state[0]) - (mag(prim[2],prim[3],prim[4])**2)/2

    ## Temperature ##
    prim[1] = calPerfectT(e, fluid)

    ## Pressure ##
    # P = idealGasP(rho, P, R)
    prim[0] = idealGasP(state[0], prim[1], fluid.R)

@ti.func
def calculateFluxes_(prim:ti.template(), state:ti.template(), fluxes:ti.template()):
    """
        Calculates fluxes of transported variables at cell center, from
        Arguments:
            prim:   (input) vector of cell center primitives
            state:  (input) vector of cell center state

        Returns:
            fluxes: (output) vector of cell center fluxes (to be calculated/populated)

        Notes:
            See dataStructuresDefinitions.md for definitions of state, primitives, etc...

    """

    #### Mass Fluxes ####
    fluxes[0] = state[1]
    fluxes[1] = state[2]
    fluxes[2] = state[3]

    #### Momentum Fluxes ####
    ## x-direction momentum fluxes ##
    # xMomxFlux = xMom*Ux + P
    fluxes[3] = state[1]*prim[2] + prim[0]
    # yMomxFlux = xMom*Uy
    fluxes[6] = state[1]*prim[3]
    # zMomxFlux = xMom*Uz
    fluxes[9] = state[1]*prim[4]

    ## y-direction momentum fluxes ##
    # xMomyFlux = yMomxFlux
    fluxes[4] = fluxes[6]
    # yMomyFlux = yMom*Uy + P
    fluxes[7] = state[2]*prim[3] + prim[0]
    # zMomyFlux = yMom*Uz
    fluxes[10] = state[2]*prim[4]

    ## z-direction momentum fluxes ##
    # xMomzFlux = zMomxFlux
    fluxes[5] = fluxes[9]
    # yMomzFlux = zMomyFlux
    fluxes[8] = fluxes[10]
    # zMomzFlux = zMom*Uz + P
    fluxes[11] = state[3]*prim[4] + prim[0]

    #### Energy Fluxes ###
    # eV2xFlux = Ux*eV2 + P*Ux
    fluxes[12] = prim[2]*state[4] + prim[0]*prim[2]
    # eV2yFlux = Uy*eV2 + P*Uy
    fluxes[13] = prim[3]*state[4] + prim[0]*prim[3]
    # eV2zFlux = Uz*eV2 + P*Uz
    fluxes[14] = prim[4]*state[4] + prim[0]*prim[4]


@ti.kernel
def encodePrimitives(cellPrimitives:ti.template(), fluid:Fluid, cellState:ti.template()):
    """
        Calculate cellState from cellPrimitives
    """
    for i in cellPrimitives:
        P  = cellPrimitives[i][0]
        T  = cellPrimitives[i][1]
        Ux = cellPrimitives[i][2]
        Uy = cellPrimitives[i][3]
        Uz = cellPrimitives[i][4]
        Umag = mag(Ux,Uy,Uz)
        Rho = idealGasRho(T=T, P=P,R=fluid.R)

        cellState[i][0] = Rho
        cellState[i][1] = Ux * Rho
        cellState[i][2] = Uy * Rho
        cellState[i][3] = Uz * Rho
        e = calPerfectEnergy(T=T, fluid=fluid)
        cellState[i][4] = Rho * (e + 0.5*Umag**2)
        
@ti.kernel
def decodeSolution(cellState:ti.template(), fluid:Fluid, cellPrimitives:ti.template(), cellFluxes:ti.template()):

    for i in cellState:
        # decodePrimitives_(cellState[i], fluid, cellPrimitives[i])
        calculateFluxes_(cellPrimitives[i], cellState[i], cellFluxes[i] )

@ti.data_oriented
class Mesh:
    def __init__(self,  ofmesh) -> None:

        self.nBCs   = len(ofmesh['boundaryFaces'])
        self.nFaces = len(ofmesh['faces'])
        self.nCells = len(ofmesh['cells'])
        self.nBdryFaces = 0

        for i in range(self.nBCs):
            self.nBdryFaces +=  len(ofmesh['boundaryFaces'][i])
        
        self.cVols = ti.field(float, shape=(self.nCells,))
        self.cVols.from_numpy(ofmesh['cVols'])

        self.cCenters = ti.field(ti.math.vec3, shape=(self.nCells,))
        self.cCenters.from_numpy(np.array(ofmesh['cCenters']))

        self.cellSizes = ti.field(ti.math.vec3, shape=(self.nCells,))
        self.cellSizes.from_numpy(ofmesh['cellSizes'])

        self.faces = ti.field(ti.math.ivec2, shape=(self.nFaces,))
        #Convert to 0-based indexing
        self.faces.from_numpy(np.array(ofmesh['faces'])-1)

        self.fAVecs = ti.field(ti.math.vec3, shape=(self.nFaces,))
        self.fAVecs.from_numpy(np.array(ofmesh['fAVecs']))

        self.fCenters = ti.field(ti.math.vec3, shape=(self.nFaces,))
        self.fCenters.from_numpy(np.array(ofmesh['fCenters']))

        self.boundaryFaces = []
        for i in range(self.nBCs):
            nBdryFaces = len(ofmesh['boundaryFaces'][i])
            BdryFaces  = ti.ndarray(int, shape=(nBdryFaces,))
            # Convert to 0-based indexing
            BdryFaces.from_numpy(ofmesh['boundaryFaces'][i]-1)
            self.boundaryFaces.append(BdryFaces)

@ti.data_oriented
class fvCFD_Solver:
    def __init__(self, 
                 ofmesh:str,
                 fluid: Fluid) -> None:

        nDims   = 3
        nVars   = 5
        nFluxes = nVars * nDims

        self.mesh = Mesh(ofmesh=ofmesh)
        self.fluid = fluid
        self.cellPrimitives = ti.Vector.field(n=nVars, dtype=float, shape=(self.mesh.nCells))
        self.cellState      = ti.Vector.field(n=nVars, dtype=float, shape=(self.mesh.nCells))
        self.fluxResiduals  = ti.Vector.field(n=nFluxes, dtype=float, shape=(self.mesh.nCells))
        self.cellFluxes     = ti.Vector.field(n=nFluxes, dtype=float, shape=(self.mesh.nCells))
        self.faceFluxes     = ti.Vector.field(n=nFluxes, dtype=float, shape=(self.mesh.nFaces))

        self.faceVals       = ti.Vector.field(n=nVars, dtype=float, shape=(self.mesh.nFaces,))
        self.primGrad       = ti.Matrix.field(n=5, m=3, dtype=ti.f32, shape=(self.mesh.nCells,))

    @ti.kernel
    def initializeUniformSolution(self,
                                  P: float,
                                  T: float,
                                  Ux: float,
                                  Uy: float,
                                  Uz: float,):
        
        for i in self.cellPrimitives:
            self.cellPrimitives[i][0] = P
            self.cellPrimitives[i][1] = T
            self.cellPrimitives[i][2] = Ux
            self.cellPrimitives[i][3] = Uy
            self.cellPrimitives[i][4] = Uz

