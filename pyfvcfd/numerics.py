import taichi as ti

@ti.dataclass
class Fluid:
    Cp: ti.f32
    R: ti.f32
    gamma: ti.f32

@ti.func
def idealGasRho(T, P, R=287.05):
    return P/(R*T)

@ti.func
def idealGasP(rho, T, R=287.05):
    return rho*R*T

@ti.func
def calPerfectEnergy(T, fluid):
    return T * (fluid.Cp - fluid.R)

@ti.func
def calPerfectT(e, fluid):
    return e / (fluid.Cp - fluid.R)

@ti.func
def mag(v1,v2,v3):
    sqrSum = v1*v1 + v2*v2 + v3*v3
    return ti.math.sqrt(sqrSum)

####################### Face value interpolation ######################
"""
    Interpolates to all INTERIOR faces
    Arbitrary value matrix interpolation
    Example Input Matrix:
    Cell      x1       x2       x3
    Cell 1    x1_c1    x2_c1    x3_c1
    Cell 2    x1_c2    x2_c2    x3_c2
    ...
    Outputs a matrix of the following form
    Cell      x1       x2       x3
    Face 1    x1_f1    x2_f1    x3_f1
    Face 2    x1_f2    x2_f2    x3_f2
    ...
"""
@ti.kernel
def linInterp(nFaces:int,
             nBdryFaces:int,
             faces:ti.template(),
             fCenters:ti.template(),
             cCenters:ti.template(),
             cellVals:ti.template(),
             faceVals:ti.template()):
    
    nVars = cellVals.n

    # Boundary face fluxes must be set separately
    for f in range(nFaces-nBdryFaces):
        # Find value at face using linear interpolation
        c1 = faces[f][0]
        c2 = faces[f][1]

        #TODO:
        c1Dist = 0.0
        c2Dist = 0.0
        for i in ti.static(range(3)):
            c1Dist += (cCenters[c1][i] - fCenters[f][i])**2
            c2Dist += (cCenters[c2][i] - fCenters[f][i])**2
        totalDist = c1Dist + c2Dist

        for v in ti.static(range(cellVals.n)):
            faceVals[f][v] = cellVals[c1][v]*(c2Dist/totalDist) + cellVals[c2][v]*(c1Dist/totalDist)
        
# Similar to linInterp_3D. Instead of linearly interpolating, calculates the change (delta) of each variable across each face (required for JST method)
@ti.kernel
def faceDeltas(nFaces:int,
             nBdryFaces:int,
             faces:ti.template(),
             cellVals:ti.template(),
             faceVals:ti.template()):
    
    # Boundary face fluxes must be set separately (faceDelta is zero at all possible boundary conditions right now)
    for f in range(nFaces-nBdryFaces):
        ownerCell = faces[f][0]
        neighbourCell = faces[f][1]
        for v in ti.static(range(5)):
            # print(cellVals[neighbourCell][v]-cellVals[ownerCell][v], ownerCell, neighbourCell)
            faceVals[f][v] = cellVals[neighbourCell][v] - cellVals[ownerCell][v]


"""
    Takes the gradient of (scalar) data provided in matrix form (passed into arugment 'matrix'):
    Cell      x1      x2      x3
    Cell 1    x1_1    x2_1    x3_1
    Cell 2    x1_2    x2_2    x3_2
    ...
    (where x1, x2, and x3 are arbitrary scalars)
    and output a THREE-DIMENSIONAL gradient arrays of the following form
    Cell      x1          x2          x3
    Cell 1    grad(x1)_1  grad(x2)_1  grad(x3)_1
    Cell 2    grad(x1)_2  grad(x2)_2  grad(x3)_2
    ...
    Where each grad(xa)_b is made up of THREE elements for the (x,y,z) directions

    Ex. Gradient @ cell 1 of P would be: greenGaussGrad(mesh, P)[1, 1, :]

"""
@ti.kernel
def greenGaussGrad(faces:ti.template(), fAVecs:ti.template(), cVols:ti.template(),
                   faceVals:ti.template(), primGrad:ti.template()):
    
    # Reset primitive gradients
    primGrad.fill(0.0)

    # Integrate fluxes from each face
    for f in faces:
        ownerCell     = faces[f][0]
        neighbourCell = faces[f][1]

        for v in ti.static(range(5)):
            for d in ti.static(range(3)):

                # Every face has an owner
                primGrad[ownerCell][v, d] += fAVecs[f][d] * faceVals[f][v]

                # Boundary faces don't - could split into two loops
                if neighbourCell > -1:
                    primGrad[neighbourCell][v, d] -= fAVecs[f][d] * faceVals[f][v]
    
    # Divide integral by cell volume to obtain gradients
    for c in cVols:
        for v in ti.static(range(5)):
            for d in ti.static(range(3)):
                primGrad[c][v, d] /= cVols[c]


@ti.kernel
def integrateFluxes3D(faces:ti.template(), fAVecs:ti.template(), 
                    cVols:ti.template(), faceFluxes:ti.template(),
                    fluxResiduals:ti.template()):

    # Recomputing flux balances, so wipe existing values
    fluxResiduals.fill(0.0)

    # Flux Integration
    for f in faces:
        ownerCell     = faces[f][0]
        neighbourCell = faces[f][1]

        #for v in range(faceFluxes.n):
        for v in range(5):
            i1 = v * 3
            flow = ti.math.dot(
                        ti.Vector([faceFluxes[f][i1], 
                                  faceFluxes[f][i1+1],
                                  faceFluxes[f][i1+2]]),
                        fAVecs[f])
            
            # Subtract from owner cell
            fluxResiduals[ownerCell][v] -= flow
            
            # Add to neighbour cell
            if neighbourCell > -1:
                fluxResiduals[neighbourCell][v] -= flow
    
    # Divide by cell volume
    for c in cVols:
        for v in ti.static(range(5)):
            fluxResiduals[c][v] /= cVols[c]
        