from dolfin import *
from matplotlib import pyplot

from fenapack import PCDKrylovSolver
from fenapack import PCDAssembler
from fenapack import PCDNewtonSolver, PCDNonlinearProblem
from fenapack import StabilizationParameterSD

import argparse, sys, os
from mpi4py import MPI

parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--nu", type=float, dest="viscosity", default=0.02,
                    help="kinematic viscosity")
parser.add_argument("--pcd", type=str, dest="pcd_variant", default="BRM2",
                    choices=["BRM1", "BRM2"], help="PCD variant")
parser.add_argument("--nls", type=str, dest="nls", default="picard",
                    choices=["picard", "newton"], help="nonlinear solver")
parser.add_argument("--ls", type=str, dest="ls", default="iterative",
                    choices=["direct", "iterative"], help="linear solvers")
args = parser.parse_args(sys.argv[1:])


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = 0


Da = 1e-4
nu = 1.#Constant(args.viscosity)
Re = 20.#180.
l = 0.2#
rho = 1.#
u0 = nu*Re/l/rho
alphamax = nu/Da/l**2#
alphamin = 0.
alpha_q = 0.1
volfrac = 0.31

parameters["form_compiler"]["quadrature_degree"] = 3
mr = 10
mesh = BoxMesh(Point(0,0,0), Point(1,1,1), mr,mr,mr)




def alpha(gamma):
	return alphamin+(alphamax-alphamin)*alpha_q*(1.0-gamma)/(alpha_q+gamma)
eps = 1e-6
class Gamma0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
class Gamma1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[2]<eps and (x[0]-0.5)**2+(x[1]-0.8)**2<=0.1**2
class Gamma2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[2]<eps and (x[0]-0.5)**2+(x[1]-0.2)**2<=0.1**2
class Gamma3(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[2]>1.-eps and (x[0]-0.2)**2+(x[1]-0.5)**2<=0.1**2
class Gamma4(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[2]>1.-eps and (x[0]-0.8)**2+(x[1]-0.5)**2<=0.1**2

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_markers.set_all(5)        # interior facets
Gamma0().mark(boundary_markers, 0) # no-slip facets
Gamma1().mark(boundary_markers, 1) # inlet facets01
Gamma2().mark(boundary_markers, 2) # inlet facets02
Gamma3().mark(boundary_markers, 3) # outlet facets01
Gamma4().mark(boundary_markers, 4) # outlet facets02

# Inlet velocity
u_in1 = Expression(("0.0","0.0","4.*u0*(pow(0.1,2)-pow(x[0]-0.5,2)-pow(x[1]-0.8,2))"),u0=u0,degree=2)
u_in2 = Expression(("0.0","0.0","4.*u0*(pow(0.1,2)-pow(x[0]-0.5,2)-pow(x[1]-0.2,2))"),u0=u0,degree=2)


# Function Space
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
V_ele = VectorElement('CG', mesh.ufl_cell(), 2)
Q_ele = FiniteElement('CG', mesh.ufl_cell(), 1)
W_ele = V_ele*Q_ele
W = FunctionSpace(mesh, W_ele)


gamma = Function(Q)
gamma.vector()[:] = volfrac

# Apply bc
bc0 = DirichletBC(W.sub(0), (0.0, 0.0, 0.0), boundary_markers, 0)
bc1 = DirichletBC(W.sub(0), u_in1, boundary_markers, 1)
bc2 = DirichletBC(W.sub(0), u_in2, boundary_markers, 2)
bc3 = DirichletBC(W.sub(1), 0.0, boundary_markers, 3)
bc4 = DirichletBC(W.sub(1), 0.0, boundary_markers, 4)

# Artificial BC for PCD preconditioner
if args.pcd_variant == "BRM1":
    bc_pcd1 = DirichletBC(W.sub(1), 0.0, boundary_markers, 1)
    bc_pcd2 = DirichletBC(W.sub(1), 0.0, boundary_markers, 2)
elif args.pcd_variant == "BRM2":
    bc_pcd1 = DirichletBC(W.sub(1), 0.0, boundary_markers, 3)
    bc_pcd2 = DirichletBC(W.sub(1), 0.0, boundary_markers, 4)

u, p = TrialFunctions(W)
v, q = TestFunctions(W)
w = Function(W)
# FIXME: Which split is correct? Both work but one might use
# restrict_as_ufc_function
u_, p_ = split(w)


F = (
      alpha(gamma)*inner(u_, v)
    + nu*inner(grad(u_), grad(v))
    + inner(dot(grad(u_), u_), v)
    - p_*div(v)
    - q*div(u_)
)*dx

J = (
          alpha(gamma)*inner(u, v)
        + nu*inner(grad(u), grad(v))
        + inner(dot(grad(u), u_), v)
        - p*div(v)
        - q*div(u)
)*dx

#mu = alpha(gamma)*inner(u, v)*dx
mp = Constant(1.0/nu)*p*q*dx
kp = Constant(1.0/nu)*(alpha(gamma)*p + dot(grad(p), u_))*q*dx
ap = inner(grad(p), grad(q))*dx

if args.pcd_variant == "BRM2":
    n = FacetNormal(mesh)
    ds = Measure("ds", subdomain_data=boundary_markers)
    # TODO: What about the reaction term? Does it appear here?
    kp -= Constant(1.0/nu)*dot(u_, n)*p*q*ds(1)+Constant(1.0/nu)*dot(u_, n)*p*q*ds(2)
    #kp -= Constant(1.0/nu)*dot(u_, n)*p*q*ds(0)  # TODO: Is this beneficial?

pcd_assembler = PCDAssembler(J, F, [bc0, bc1, bc2, bc3, bc4],
                             ap=ap, kp=kp, mp=mp, bcs_pcd=[bc_pcd1, bc_pcd2])
problem = PCDNonlinearProblem(pcd_assembler)

linear_solver = PCDKrylovSolver(comm=mesh.mpi_comm())
linear_solver.parameters["relative_tolerance"] = 1e-6
PETScOptions.set("ksp_monitor")
PETScOptions.set("ksp_gmres_restart", 150)


# Set up subsolvers
PETScOptions.set("fieldsplit_p_pc_python_type", "fenapack.PCDPC_" + args.pcd_variant)


PETScOptions.set("fieldsplit_u_ksp_type", "richardson")
PETScOptions.set("fieldsplit_u_ksp_max_it", 1)
PETScOptions.set("fieldsplit_u_pc_type", "hypre")
PETScOptions.set("fieldsplit_u_pc_hypre_type", "boomeramg")
PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_type", "richardson")
PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_max_it", 2)
PETScOptions.set("fieldsplit_p_PCD_Ap_pc_type", "hypre")
PETScOptions.set("fieldsplit_p_PCD_Ap_pc_hypre_type", "boomeramg")
PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_type", "chebyshev")
PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_max_it", 5)
PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_chebyshev_eigenvalues", "0.5, 2.5")
PETScOptions.set("fieldsplit_p_PCD_Mp_pc_type", "jacobi")

# Apply options
linear_solver.set_from_options()

# Set up nonlinear solver
solver = PCDNewtonSolver(linear_solver)
solver.parameters["relative_tolerance"] = 1e-6

# Solve problem
solver.solve(problem, w.vector())
