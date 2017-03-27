from scipy.misc import imread
import numpy as np
from dolfin import *
from dolfin_adjoint import *
import moola


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
u_noisy.vector()[:] = u.vector().array() + 30*np.random.random(220*220)


### solving the basic forward problem
u = Function(V, name='Control')
form = (u-u_noisy)*v*dx
solve(form==0, u)


### Total variation denoising
l = 1.
epsilon = 1e-6
control = Control(u)
J = Functional(0.5*inner(u-u_noisy, u-u_noisy)*dx + (l)*sqrt(inner(nabla_grad(u), nabla_grad(u))+epsilon)*dx)
rf = ReducedFunctional(J, control)
problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u, inner_product="L2")
solver = moola.NewtonCG(problem, u_moola)
sol = solver.solve()


#### Saving results
u_opt = sol['control'].data
u = Function(V)
u.assign(u_opt)

file1 = File("u_opt.pvd")
file1 << u
file2 = File("u_true.pvd")
file2 << u_true
file3 = File("u_noisy.pvd")
file3 << u_noisy
# filename << u_true
# filename << u_noisy
# filename << mesh
