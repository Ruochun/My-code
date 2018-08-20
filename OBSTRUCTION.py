from dolfin import *
from matplotlib import pyplot

from fenapack import PCDKrylovSolver
from fenapack import PCDAssembler
from fenapack import PCDNewtonSolver, PCDNonlinearProblem
from fenapack import StabilizationParameterSD

import argparse, sys, os


# Parse input arguments
parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", type=int, dest="level", default=1,
                    help="level of mesh refinement")

parser.add_argument("--nu", type=float, dest="viscosity", default=0.1,
                    help="kinematic viscosity")

parser.add_argument("--pcd", type=str, dest="pcd_variant", default="BRM2",
                    choices=["BRM1", "BRM2"], help="PCD variant")

parser.add_argument("--nls", type=str, dest="nls", default="newton",
                    choices=["picard", "newton"], help="nonlinear solver")
args = parser.parse_args(sys.argv[1:])

# create mesh
from mshr import *
channel = Box(Point(-0.75, -1.5, -2.5), Point(0.75, 1.5, 2.5))
box     = Box(Point(-0.5,  -0.5, -0.5), Point(0.5,  0.5, 0.5))
g3d     = channel - box
mesh    = generate_mesh(g3d, 10)

for i in range(args.level):
    mesh = refine(mesh)

class Gamma0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
# Inlet bc       
class Gamma1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], -2.5)
# Oultet bc
class Gamma2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2],  2.5)
        
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_markers.set_all(3)        # interior facets
Gamma0().mark(boundary_markers, 0) # no-slip facets
Gamma1().mark(boundary_markers, 1) # inlet facets
Gamma2().mark(boundary_markers, 2) # outlet facets

# Build Taylor-Hood function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P2*P1)

# Provide some info about the current problem
info("Dimension of the function space: %g" % W.dim())
info("DOF of velocity: %g" % W.sub(0).dim())
info("DOF of pressure: %g" % W.sub(1).dim())
