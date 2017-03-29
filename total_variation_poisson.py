from fenics import *
from mshr import *
import numpy as np
import scipy.io as sio
from dolfin_adjoint import *
import moola
np.random.seed(1)
set_log_level(ERROR)




R_ves = 6
p_ves = 80.
#C = 1.14e-3

center1 = [-100, -100]
center2 = [100, 100]
mesh = Mesh("Mesh.xml")
V = FunctionSpace(mesh, 'CG', 1)
W = FunctionSpace(mesh, 'DG', 0)

def boundary1(x, on_boundary):
    r = np.sqrt((x[0]-center1[0])**2 + (x[1]-center1[1])**2)
    b = ((r < R_ves+DOLFIN_EPS) and on_boundary)
    return b

def boundary2(x, on_boundary):
    r = np.sqrt((x[0]-center2[0])**2 + (x[1]-center2[1])**2)
    b = ((r < R_ves+DOLFIN_EPS) and on_boundary)
    return b

bc1 = DirichletBC(V, p_ves, boundary1)
bc2 = DirichletBC(V, p_ves, boundary2)
bcs = [bc1, bc2]




C = 3.54e-4
p = Function(V)
# M = Function(V)
M = Constant(C)
v = TestFunction(V)
form = (inner(nabla_grad(p), nabla_grad(v)) + M*v )*dx
solve(form==0, p, bcs)

p_solution = p.copy(deepcopy=True)
noise = 5*np.random.randn(np.size(p.vector().array()))
p_noisy = p.copy(deepcopy=True)
p_noisy.vector()[:] = p.vector().array() + noise

e1 = errornorm(p,p_noisy)

# p_noisy = Function(V)
p = Function(V, name='State')
M = Function(W, name='Control')
form = (inner(nabla_grad(p), nabla_grad(v)) + M*v )*dx
solve(form==0, p, bcs)

l = 1
eps = 1e-8

control = Control(M)

def func(p, M, l, eps):
    return (0.5*inner(p_noisy-p,p_noisy-p) + l*sqrt(inner(nabla_grad(M), nabla_grad(M))+eps))*dx

J = Functional(func(p,M,l,eps))
rf = ReducedFunctional(J, control)
problem = MoolaOptimizationProblem(rf)
M_moola = moola.DolfinPrimalVector(M, inner_product="L2")
solver = moola.BFGS(problem, M_moola, options={'jtol': 0,
                                               'rjtol': 1e-12,
                                               'gtol': 1e-9,
                                               'Hinit': "default",
                                               'maxiter': 100,
                                               'mem_lim': 10})
sol = solver.solve()


M_opt = sol['control'].data
# me2 = errornorm(M, M_opt)
# M_noisy.assign(M_opt)

p_opt = Function(V)
form_opt = (inner(nabla_grad(p_opt), nabla_grad(v)) + M_opt*v )*dx
solve(form_opt==0, p_opt,bcs)

file1 = File("p_opt.pvd")
file1 << p_opt

e2 = errornorm(p_opt, p_solution)


print "Error in noisy signal: ", e1
print "Error in restored signal: ", e2

file1 = File("p_noisy.pvd")
file1 << p_noisy
file2 = File("p_exact.pvd")
file2 << p_solution
file3 = File("p_optimal.pvd")
file3 << p_opt
