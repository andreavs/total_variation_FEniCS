from scipy.misc import imread
import numpy as np
from dolfin import *
from dolfin_adjoint import *
import moola
set_log_level(ERROR)
np.random.seed(1)

### This first part is a hacky way of importing an image to FEniCS.
N = 219
mesh = UnitSquareMesh(N,N)
X = imread('phantom.png', mode='L')
Y = np.reshape(X, (220*220,1))

import matplotlib.pyplot as plt
V = FunctionSpace(mesh, 'CG', 1)
W = FunctionSpace(mesh, 'DG', 0)
v = TestFunction(V)

d2v = dof_to_vertex_map(V)
u = Function(V)
u.vector()[:] = Y[d2v]


### Adding noise:
u_noisy = u.copy(deepcopy=True)
u_true = u.copy(deepcopy=True)
u_noisy.vector()[:] = u.vector().array() + (5*np.random.randn(220*220))


### solving the basic forward problem
u = Function(V, name='State')
f = Function(W, name='Control')
f.assign(u_true - u_noisy)
f_true = f.copy(deepcopy=True)
f_noisy = Function(W)

form = (u - u_noisy - f)*v*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(form==0, u, bc)
u_sol1 = u.copy(deepcopy=True)

### denoising
l = 1e-2
epsilon = 1e-3
def func(u, l, epsilon, f):
    return (l*(sqrt(inner(nabla_grad(u), nabla_grad(u)) + epsilon)) + 0.5*f*f)*dx

J = Functional(func(u,l,epsilon,f))
control = Control(f)
rf = ReducedFunctional(J, control)
problem = MoolaOptimizationProblem(rf)
f_moola = moola.DolfinPrimalVector(f, inner_product="L2")
solver = moola.NewtonCG(problem, f_moola, options={'gtol': 1e-9,
                                                   'maxiter': 200,
                                                   'display': 3,
                                                   'ncg_hesstol': 0,
                                                   'line_search': 'strong_wolfe'})
sol = solver.solve()


print sol

#### Saving results
f_opt = sol['control'].data
# f.assign(f_opt)
# solve(form==0, u, bc)




### Print out a bunch of stuff:
# def tv(u,l,epsilon):
#     return l*sqrt(inner(nabla_grad(u), nabla_grad(u)) + epsilon)



# print assemble((u_noisy-u_true)*(u_noisy-u_true)*dx)
# print assemble((u-u_sol1)*(u-u_sol1)*dx)
# print assemble((u_noisy-u_sol1)*(u_noisy-u_sol1)*dx)
# print assemble((u_true-u)*(u_true-u)*dx)
# print assemble((u-u_noisy)*(u-u_noisy)*dx)
#
# print assemble(tv(u,l,epsilon)*dx)
# print assemble(tv(u_true,l,epsilon)*dx)
# print assemble(tv(u_noisy,l,epsilon)*dx)

def calculate_functional(f, l, epsilon):
    u = Function(V)
    form = (u - u_noisy - f)*v*dx
    solve(form==0, u, bc)
    return func(u, l, epsilon, f)


print "Functional of original signal: ", assemble(calculate_functional(f_true, l, epsilon))
print "Functional of noisy signal: ",assemble(calculate_functional(f_noisy, l, epsilon))
print "Functional of optimal signal: ", assemble(calculate_functional(f_opt, l, epsilon))
print "Functional of initial guess signal: ", assemble(calculate_functional(f, l, epsilon))

file1 = File("u_opt.pvd")
file1 << u
file2 = File("u_true.pvd")
file2 << u_true
file3 = File("u_noisy.pvd")
file3 << u_noisy
