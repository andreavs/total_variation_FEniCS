from scipy.misc import imread
import numpy as np
from dolfin import *
from dolfin_adjoint import *
import moola
np.seed(1)

### This first part is a hacky way of importing an image to FEniCS.
N = 219
mesh = UnitSquareMesh(N,N)
X = imread('phantom.png', mode='L')

print np.size(X)

Y = np.reshape(X, (220*220,1))
print np.size(X)

import matplotlib.pyplot as plt
V = FunctionSpace(mesh, 'CG', 1)
v = TestFunction(V)

d2v = dof_to_vertex_map(V)
u = Function(V)
print np.size(u.vector().array())
u.vector()[:] = Y[d2v]


### Adding noise:
u_noisy = u.copy(deepcopy=True)
u_true = u.copy(deepcopy=True)
u_noisy.vector()[:] = u.vector().array() + (5*np.random.randn(220*220))


### solving the basic forward problem
u = Function(V)
# f = interpolate(Expression('100*x[0]', degree=3), V, name='Control')
f = Function(V)
f.assign(u_true - u_noisy)
# f.vector()[:] = 1.0
epsilon = 1e-8
form = (u - u_noisy - f)*v*dx
a,L = system(form)
bc = DirichletBC(V, 0.0, "on_boundary")
solve(form==0, u, bc)
u_sol1 = u.copy(deepcopy=True)

### denoising
l = 1.0
def func(u, l, epsilon, f):
    return (l*(inner(nabla_grad(u), nabla_grad(u))) + 0.5*f*f)*dx

# fn = l*sqrt(inner(nabla_grad(u), nabla_grad(u)) + 1e-5)*dx + 0.5*f*f*dx
# J = Functional(l*sqrt(inner(nabla_grad(u), nabla_grad(u)) + epsilon)*dx + 0.5*f*f*dx)
J = Functional(func(u,l,epsilon,f))
control = Control(f)
rf = ReducedFunctional(J, control)
problem = MoolaOptimizationProblem(rf)
f_moola = moola.DolfinPrimalVector(f, inner_product="L2")
solver = moola.BFGS(problem, f_moola)
sol = solver.solve()


print sol

#### Saving results
f_opt = sol['control'].data
f.assign(f_opt)
solve(form==0, u, bc)



### Print out a bunch of stuff:
def tv(u,l,epsilon):
    return l*sqrt(inner(nabla_grad(u), nabla_grad(u)) + epsilon)



# print assemble((u_noisy-u_true)*(u_noisy-u_true)*dx)
# print assemble((u-u_sol1)*(u-u_sol1)*dx)
# print assemble((u_noisy-u_sol1)*(u_noisy-u_sol1)*dx)
# print assemble((u_true-u)*(u_true-u)*dx)
# print assemble((u-u_noisy)*(u-u_noisy)*dx)
#
# print assemble(tv(u,l,epsilon)*dx)
# print assemble(tv(u_true,l,epsilon)*dx)
# print assemble(tv(u_noisy,l,epsilon)*dx)


print "Functional of original signal: ", assemble(func(u_true,l,epsilon,f)*dx)
print "Functional of noisy signal: ",assemble(func(u_noisy,l,epsilon,f)*dx)
print "Functional of optimal signal: ", assemble(func(u,l,epsilon,f)*dx)
print "Functional of initial guess signal: ", assemble(func(u_sol1,l,epsilon,f)*dx)

# file1 = File("u_opt.pvd")
# file1 << u
# file2 = File("u_true.pvd")
# file2 << u_true
# file3 = File("u_noisy.pvd")
# file3 << u_noisy
# filename << u_true
# filename << u_noisy
# filename << mesh
