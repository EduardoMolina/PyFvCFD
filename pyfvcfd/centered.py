import taichi as ti
import taichi.math as tm

from pyfvcfd import faceDeltas

@ti.data_oriented
class centralJST:

    def __init__(self,
                 nCells : int,
                 nFaces : int,
                 nBdryFaces: int,
                 k2 : float = 0.5,
                 k4 : float = (1.0/32.0),
                 ):
        
        self.nCells     = nCells
        self.nFaces     = nFaces
        self.nBdryFaces = nBdryFaces
        self.k2       = k2
        self.k4       = k4
        self.eps      = ti.Vector.field(n=2, dtype=float, shape = (nCells,))
        self.fDeltas  = ti.Vector.field(n=5, dtype=float, shape = (nFaces,))
        self.fDGrads  = ti.Matrix.field(n=5, m=3, dtype=ti.f32, shape=(nFaces,))
        
        # 'Sensor' used to detect shock waves and apply second-order artificial diffusion to stabilize solution in their vicinity
        self.sj = ti.field(float, shape=(nCells,))
        
        # Store the number of sj's calculated for each cell, cell-center value will be the average of all of them
        self.sjCount = ti.field(ti.i32, shape=(nCells,))

    @ti.kernel
    def calculate_eps(self, 
                      faces:ti.template(), cCenters:ti.template(),
                      prim:ti.template(), primGrad:ti.template()):

        self.sj.fill(0.0)
        self.sjCount.fill(0)
        for f in range(self.nFaces - self.nBdryFaces):
            
            # At each internal face, calculate sj, rj, eps2, eps4
            ownerCell = faces[f][0]
            neighbourCell = faces[f][1]
            d = cCenters[neighbourCell] - cCenters[ownerCell]
            
            # 1. Find pressure at owner/neighbour cells
            oP = prim[ownerCell][0]
            nP = prim[neighbourCell][0]

            # 2. Calculate pressures at 'virtual' far-owner and far-neighbour cells using the pressure gradient (2nd-order)
            oPGrad = ti.Vector([primGrad[ownerCell][0,0], primGrad[ownerCell][0,1], primGrad[ownerCell][0,2]])
            nPGrad = ti.Vector([primGrad[neighbourCell][0,0], primGrad[neighbourCell][0,1], primGrad[neighbourCell][0,2]])
            farOwnerP = nP - 2*ti.math.dot(d, oPGrad)
            farNeighbourP = oP + 2*ti.math.dot(d, nPGrad)

            # 3. With the known and virtual values, can calculate sj at each cell center.
            self.sj[ownerCell] += (ti.abs( nP - 2.0*oP + farOwnerP )/ ti.max( ti.abs(nP - oP) + ti.abs(oP - farOwnerP), 0.0000000001))**2
            self.sjCount[ownerCell] += 1
            self.sj[neighbourCell] += (ti.abs( oP - 2.0*nP + farNeighbourP )/ ti.max( ti.abs(farNeighbourP - nP) + ti.abs(nP - oP), 0.0000000001))**2
            self.sjCount[neighbourCell] += 1

    def calculate_fDeltas(self, nFaces, nBdryFaces, faces, cellState):
        faceDeltas(nFaces,nBdryFaces,faces, cellState, self.fDeltas)


    @ti.kernel
    def calculate_JSTFlux(self, 
                      faces:ti.template(), cCenters:ti.template(),
                      prim:ti.template(), primGrad:ti.template()):
        
        pass