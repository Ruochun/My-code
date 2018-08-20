from dolfin import *
from matplotlib import pyplot

from fenapack import PCDKrylovSolver
from fenapack import PCDAssembler
from fenapack import PCDNewtonSolver, PCDNonlinearProblem
from fenapack import StabilizationParameterSD

import argparse, sys, os

parameters["form_compiler"]["quadrature_degree"] = 3
# Parse input arguments
parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", type=int, dest="level", default=1,
                    help="level of mesh refinement")

#parser.add_argument("--nu", type=float, dest="viscosity", default=0.1,
#                    help="kinematic viscosity")

parser.add_argument("--pcd", type=str, dest="pcd_variant", default="BRM2",
                    choices=["BRM1", "BRM2"], help="PCD variant")

parser.add_argument("--nls", type=str, dest="nls", default="picard",
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
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P2*P1)

Re = 50
nu = 0.1#Constant(args.viscosity)
Dh = 4*0.75*2*1.5*2/(0.75*2*2+1.5*2*2)
U0 = Re*nu/Dh

# Provide some info about the current problem
info("Dimension of the function space: %g" % W.dim())
info("DOF of velocity: %g" % W.sub(0).dim())
info("DOF of pressure: %g" % W.sub(1).dim())
info("Reynolds number: %g" % Re)

# No-slip BC
bc0 = DirichletBC(W.sub(0), (0.0, 0.0, 0.0), boundary_markers, 0)

# Parabolic inflow BC
inflow = Expression(("0.0", "0.0", "2.0*U0*(0.75-x[0])*(0.75+x[0])*(1.5-x[1])*(1.5+x[1])/(0.75*0.75*1.5*1.5)"), U0=U0,degree=4)
bc1 = DirichletBC(W.sub(0), inflow, boundary_markers, 1)

# Artificial BC for PCD preconditioner
if args.pcd_variant == "BRM1":
    bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 1)
elif args.pcd_variant == "BRM2":
    bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 2)
    
# Arguments and coefficients of the form
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
w = Function(W)
u_, p_ = split(w)

h = CellDiameter(mesh)
vnorm = sqrt(dot(u_,u_))
tau_supg = ( (2.0*vnorm/h)**2 + 9*(4.0*nu/h**2)**2 )**(-0.5)
tau_pspg = h**2/2#tau_supg#

# Nonlinear equation

F = (
      nu*inner(grad(u_), grad(v))
    + inner(dot(grad(u_), u_), v)
    - p_*div(v)
    - q*div(u_)
    + tau_supg*inner(grad(v)*u_,grad(u_)*u_+grad(p_)-div(nu*grad(u_)))
    - tau_pspg*inner(grad(q),grad(u_)*u_+grad(p_)-div(nu*grad(u_)))
)*dx
"""
F = (
      nu*inner(grad(u_), grad(v))
    + inner(dot(grad(u_), u_), v)
    - p_*div(v)
    - q*div(u_)
)*dx
"""
# Jacobian
if args.nls == "picard":
    J = (
          nu*inner(grad(u), grad(v))
        + inner(dot(grad(u), u_), v)
        - p*div(v)
        - q*div(u)
    )*dx
elif args.nls == "newton":
    J = derivative(F, w)
J = J + tau_supg*inner(grad(v)*u_,grad(u)*u_+grad(p)-div(nu*grad(u)))*dx\
      - tau_pspg*inner(grad(q),grad(u)*u_+grad(p)-div(nu*grad(u)))*dx


# PCD operators
mp = Constant(1.0/nu)*p*q*dx
kp = Constant(1.0/nu)*dot(grad(p), u_)*q*dx
ap = inner(grad(p), grad(q))*dx
if args.pcd_variant == "BRM2":
    n = FacetNormal(mesh)
    ds = Measure("ds", subdomain_data=boundary_markers)
    kp -= Constant(1.0/nu)*dot(u_, n)*p*q*ds(1)
    
pcd_assembler = PCDAssembler(J, F, [bc0, bc1],
                             ap=ap, kp=kp, mp=mp, bcs_pcd=bc_pcd)
                             
problem = PCDNonlinearProblem(pcd_assembler)

# Set up linear solver (GMRES with right preconditioning using Schur fact)
linear_solver = PCDKrylovSolver(comm=mesh.mpi_comm())
linear_solver.parameters["relative_tolerance"] = 1e-6
PETScOptions.set("ksp_monitor")
PETScOptions.set("ksp_gmres_restart", 150)

# Set up subsolvers
PETScOptions.set("fieldsplit_p_pc_python_type", "fenapack.PCDPC_" + args.pcd_variant)

#PETScOptions.set("fieldsplit_u_ksp_type", "richardson")
#PETScOptions.set("fieldsplit_u_ksp_max_it", 1)
#PETScOptions.set("fieldsplit_u_pc_type", "hypre")
#PETScOptions.set("fieldsplit_u_pc_hypre_type", "boomeramg")
PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_type", "gmres")
PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_max_it", 2)
PETScOptions.set("fieldsplit_p_PCD_Ap_pc_type", "hypre")
PETScOptions.set("fieldsplit_p_PCD_Ap_pc_hypre_type", "boomeramg")
#PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_type", "chebyshev")
PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_type", "cg")
PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_max_it", 5)
#PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_chebyshev_eigenvalues", "0.5, 2.5")
PETScOptions.set("fieldsplit_p_PCD_Mp_pc_type", "jacobi")

#PETScOptions.set("fieldsplit_u_mat_mumps_icntl_4", 2)
#PETScOptions.set("fieldsplit_p_PCD_Ap_mat_mumps_icntl_4", 2)
#PETScOptions.set("fieldsplit_p_PCD_Mp_mat_mumps_icntl_4", 2)

# Apply options
linear_solver.set_from_options()

# Set up nonlinear solver
solver = PCDNewtonSolver(linear_solver)
solver.parameters["relative_tolerance"] = 1e-5

# Solve problem
solver.solve(problem, w.vector())

# Report timings
list_timings(TimingClear.clear, [TimingType.wall, TimingType.user])

# Plot solution
u, p = w.split()
size = MPI.size(mesh.mpi_comm())
rank = MPI.rank(mesh.mpi_comm())
File('result/velocity.pvd')<<u
