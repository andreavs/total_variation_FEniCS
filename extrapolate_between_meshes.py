from fenics import *
from mshr import *
import numpy as np

### first we set up the system:
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



### Solve the noiseless system to find the true p
C = 3.54e-4
p = Function(V)
# M = Function(V)
M = Constant(C)
v = TestFunction(V)
form = (inner(nabla_grad(p), nabla_grad(v)) + M*v )*dx
solve(form==0, p, bcs)


### Map to mesh without hole
N = 40
point_1 = Point(-400,-400)
point_2 = Point(400, 400)
mesh_without_holes = RectangleMesh(point_1, point_2, N, N)
V_without_holes = FunctionSpace(mesh_without_holes, 'CG', 1)


parameters['allow_extrapolation'] = True
u = Function(V)
u.assign(p)
# parameters['allow_extrapolation'] = False
u = project(u, V_without_holes)

plot(u, interactive=True)
