import pickle
import pyfvcfd
import taichi as ti

ti.init()

with open ('../meshes/naca0012.pl', 'rb') as fp:
    ofmesh = pickle.load(fp)

### Define the fluid ###
gamma = 1.4
R = 287.05
Cp = 1005
fluid = pyfvcfd.Fluid(Cp, R, gamma)


solver = pyfvcfd.fvCFD_Solver(ofmesh=ofmesh, fluid = fluid)
solver.initializeUniformSolution(P = 101325.0, T = 325.0, Ux = 100.0, Uy = 0.0, Uz = 0.0)
pyfvcfd.encodePrimitives(solver.cellPrimitives, solver.fluid, solver.cellState)
pyfvcfd.decodeSolution(solver.cellState, solver.fluid, solver.cellPrimitives,solver.cellFluxes)
pyfvcfd.linInterp(solver.mesh.nFaces, solver.mesh.nBdryFaces, solver.mesh.faces, 
        solver.mesh.fCenters, solver.mesh.cCenters,
        solver.cellPrimitives, solver.faceVals)

pyfvcfd.linInterp(solver.mesh.nFaces, solver.mesh.nBdryFaces, solver.mesh.faces, 
        solver.mesh.fCenters, solver.mesh.cCenters, 
        solver.cellFluxes, solver.faceFluxes)

breakpoint()
pyfvcfd.integrateFluxes3D(solver.mesh.faces, solver.mesh.fAVecs, solver.mesh.cVols,
                solver.faceFluxes, solver.fluxResiduals)

pyfvcfd.greenGaussGrad(solver.mesh.faces, solver.mesh.fAVecs, solver.mesh.cVols,
            solver.faceVals, solver.primGrad)

fluxfunc = pyfvcfd.centralJST(nCells=solver.mesh.nCells, nFaces=solver.mesh.nFaces, 
                    nBdryFaces=solver.mesh.nBdryFaces)
fluxfunc.calculate_eps(solver.mesh.faces, solver.mesh.cCenters, solver.cellPrimitives,
                    solver.primGrad)
fluxfunc.calculate_fDeltas(solver.mesh.nFaces, solver.mesh.nBdryFaces,
                        solver.mesh.faces, solver.cellState)



# ti.profiler.print_scoped_profiler_info()
breakpoint()