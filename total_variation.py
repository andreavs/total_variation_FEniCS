from dolfin import *

def tikhonov_regularization(u, k=Constant(1)):
    R_TN = assemble(0.5*k*inner(nabla_grad(u), nabla_grad(u))*dx)
    return R_TN

def total_variation_regularization(u, k=Constant(1), eps=1e-4):
    R_TV = assemble(k*sqrt(inner(nabla_grad(u), nabla_grad(u)) + eps)*dx)
    return R_TV


if __name__=='__main__':
    from scipy.misc import imread
    import numpy as np
    N = 219
    mesh = UnitSquareMesh(N,N)
    X = imread('phantom.png', mode='L')

    print np.size(X)

    Y = np.reshape(X, (220*220,1))
    print np.size(X)

    import matplotlib.pyplot as plt
    V = FunctionSpace(mesh, 'CG', 1)

    v2d = vertex_to_dof_map(V)
    d2v = dof_to_vertex_map(V)
    u = Function(V)
    print np.size(u.vector().array())
    u.vector()[:] = Y[d2v]

    tn = tikhonov_regularization(u)
    tv = total_variation_regularization(u)

    print tn
    print tv


    plot(u)
    interactive()
