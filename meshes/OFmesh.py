
# import taichi as ti
import numpy as np

# Import Julia to read the openfoam mesh
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
import pickle

if __name__=='__main__':
    
    Main.include("mesh.jl")

    ### Choose a mesh ###
    meshPath = "../../FvCFD.jl/test/OFairfoilMesh"
    # meshPath = "../../FvCFD.jl/test/OFforwardStepMesh"
    # meshPath = "../FvCFD.jl/test/OFdeltaMesh"
    mesh = Main.OpenFOAMMesh(meshPath)
    # points, OFfaces, owner, neighbour, boundaryNames, boundaryNumFaces, boundaryStartFaces = Main.readOpenFOAMMesh(meshPath)
    meshout = {
        "cells": mesh.cells,
        "cVols": mesh.cVols,
        "cCenters": mesh.cCenters,
        "cellSizes": mesh.cellSizes,

        "faces": mesh.faces,
        "fAVecs": mesh.fAVecs,
        "fCenters": mesh.fCenters,
        "boundaryFaces": mesh.boundaryFaces
    }

    with open('meshfile.pl', 'wb') as fp:
        pickle.dump(meshout, fp)

    
    breakpoint()
