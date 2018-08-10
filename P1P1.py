
"""Flow over a backward-facing step. Incompressible Navier-Stokes equations are
solved using Newton/Picard iterative method. Linear solver is based on field
split PCD preconditioning."""

# Copyright (C) 2015-2017 Martin Rehor, Jan Blechta
#
# This file is part of FENaPack.
#
# FENaPack is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FENaPack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FENaPack.  If not, see <http://www.gnu.org/licenses/>.

# Begin demo

from dolfin import *
from matplotlib import pyplot

from fenapack import PCDKrylovSolver
from fenapack import PCDAssembler
from fenapack import PCDNewtonSolver, PCDNonlinearProblem
from fenapack import StabilizationParameterSD
import numpy as np
import argparse, sys, os

# Parse input arguments
parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--nu", type=float, dest="viscosity", default=0.02,
                    help="kinematic viscosity")
parser.add_argument("--pcd", type=str, dest="pcd_variant", default="BRM2",
                    choices=["BRM1", "BRM2"], help="PCD variant")


args = parser.parse_args(sys.argv[1:])
parameters["form_compiler"]["quadrature_degree"]=3
# Load mesh from file and refine uniformly
#mesh = Mesh(os.path.join(os.path.pardir, "data", "mesh_lshape.xml"))
mesh = UnitCubeMesh(15,15,15)

print (mesh.num_cells())
# Define and mark boundaries
class Gamma0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
class Gamma1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 1.0)
class Gamma2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 5.0)
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

# No-slip BC
bc0 = DirichletBC(W.sub(0), (0.0, 0.0, 0.0), boundary_markers, 0)

# Parabolic inflow BC
#inflow = Expression(("4.0*x[1]*(1.0 - x[1])", "0.0"), degree=2)
inflow = Expression(("1.0", "0.0", "0.0"), degree=1)
bc1 = DirichletBC(W.sub(0), inflow, boundary_markers, 1)

# Artificial BC for PCD preconditioner
if args.pcd_variant == "BRM1":
    bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 1)
elif args.pcd_variant == "BRM2":
    bc_pcd = DirichletBC(W.sub(1), 0.0, boundary_markers, 2)

# Provide some info about the current problem
info("Reynolds number: Re = %g" % (2.0/args.viscosity))
info("Dimension of the function space: %g" % W.dim())

# Arguments and coefficients of the form

#w1 = Function(W)
#w2 = Function(W)
# FIXME: Which split is correct? Both work but one might use
# restrict_as_ufc_function
u_ = Function(V)
u_c= Function(V)
#u_, p_ = w.split()
nu = Constant(args.viscosity)

h = CellDiameter(mesh)
picard_error = 1
picard_iter = 0
# Nonlinear equation
while picard_error>1e-8:
	u, p = TrialFunctions(W)
	v, q = TestFunctions(W)
	w = Function(W)
	picard_iter += 1
# Linearized equation
	F = (
      		nu*inner(grad(u), grad(v))
    		+ inner(dot(grad(u), u_), v)
    		- p*div(v)
    		- q*div(u)
	)*dx

	vnorm = sqrt(dot(u_, u_))
	tau_supg = ( (2.0*vnorm/h)**2 + 9*(4.0*nu/h**2)**2 )**(-0.5)
	tau_pspg = h**2/2#tau_supg#
	tau_lsic = vnorm*h/2
	res = grad(u)*u_+grad(p)-div(nu*grad(u))
	#res = grad(u)*u_k+grad(p)-div(mu*2.0*sym(grad(u)))+a(gamma)*u
	F += tau_supg*inner(grad(v)*u_,res)*dx
	F += -tau_pspg*inner(grad(q),res)*dx
	#F += tau_lsic*inner(grad(v),grad(u))*dx

#a, L = lhs(F), rhs(F)
	f = Constant((0.0,0.0,0.0))
	L = inner(f,v)*dx

# PCD operators
	mp = Constant(1.0/nu)*p*q*dx
	kp = Constant(1.0/nu)*dot(grad(p), u_)*q*dx
	ap = inner(grad(p), grad(q))*dx
	if args.pcd_variant == "BRM2":
    		n = FacetNormal(mesh)
    		ds = Measure("ds", subdomain_data=boundary_markers)
    		kp -= Constant(1.0/nu)*dot(u_, n)*p*q*ds(1)
    #kp -= Constant(1.0/nu)*dot(u_, n)*p*q*ds(0)  # TODO: Is this beneficial?
#SystemAssembler(F, L, [bc0, bc1])
# Collect forms to define nonlinear problem
	pcd_assembler = PCDAssembler(F, L, [bc0, bc1], ap=ap, kp=kp, mp=mp, bcs_pcd=bc_pcd)
	problem = PCDNonlinearProblem(pcd_assembler)
#problem = PCDLinearProblem(pcd_assembler)

# Set up linear solver (GMRES with right preconditioning using Schur fact)
	linear_solver = PCDKrylovSolver(comm=mesh.mpi_comm())
	linear_solver.parameters["relative_tolerance"] = 1e-10
	PETScOptions.set("ksp_monitor")
	PETScOptions.set("ksp_gmres_restart", 150)

# Set up subsolvers
	PETScOptions.set("fieldsplit_p_pc_python_type", "fenapack.PCDPC_" + args.pcd_variant)

	PETScOptions.set("fieldsplit_u_ksp_type", "richardson")
	PETScOptions.set("fieldsplit_u_ksp_max_it", 1)
	PETScOptions.set("fieldsplit_u_pc_type", "hypre")
	PETScOptions.set("fieldsplit_u_pc_hypre_type", "boomeramg")
	PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_type", "gmres")
	PETScOptions.set("fieldsplit_p_PCD_Ap_ksp_max_it", 2)
	PETScOptions.set("fieldsplit_p_PCD_Ap_pc_type", "hypre")
	PETScOptions.set("fieldsplit_p_PCD_Ap_pc_hypre_type", "boomeramg")
	PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_type", "gmres")
	PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_max_it", 5)
	#PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_chebyshev_eigenvalues", "0.5, 2.0")
#PETScOptions.set("fieldsplit_p_PCD_Mp_ksp_chebyshev_esteig", "1,0,0,1")  # FIXME: What does it do?
	PETScOptions.set("fieldsplit_p_PCD_Mp_pc_type", "hypre")
	PETScOptions.set("fieldsplit_p_PCD_Mp_pc_hypre_type", "boomeramg")

# Apply options
	linear_solver.set_from_options()

# Set up nonlinear solver
	solver = PCDNewtonSolver(linear_solver)
#solver.parameters["relative_tolerance"] = 1e-7

# Solve problem
	solver.solve(problem, w.vector())
	
	assign(u_c,w.sub(0))
	diff = (u_c.vector().get_local() - u_.vector().get_local())
	picard_error = np.linalg.norm(diff, ord = np.Inf)
	assign(u_,w.sub(0))
	print ('error', picard_error, ' iter = ', picard_iter)
# Report timings
#list_timings(TimingClear.clear, [TimingType.wall, TimingType.user])

# Plot solution
u, p = w.split()
size = MPI.size(mesh.mpi_comm())
rank = MPI.rank(mesh.mpi_comm())
File("result/v.pvd")<<u
