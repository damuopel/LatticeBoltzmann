from numpy import *
import matplotlib.pyplot as plt
import gif
from os import getcwd, path

# DEFINE CONSTANTS
maxIter = 100000
Re = 200
nx, ny = 420, 180 # MESH SIZE
lx, ly = nx-1, ny-1 # SIZE IN LATTICE UNITS 
cx, cy, r = nx//4, ny//2, ny//9 # BC: CYLINDER
uLB = 0.04 # VELOCITY
nulb = uLB*r/Re # VISCOSITY
omega = 1/(3*nulb+0.5) # RELAXATION

# LATTICE CONSTANTS
v = array([ [ 1,  1], [ 1,  0], [ 1, -1], [ 0,  1], [ 0,  0],
	            [ 0, -1], [-1,  1], [-1,  0], [-1, -1] ])
t = array([ 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])

col1 = array([0, 1, 2])
col2 = array([3, 4, 5])
col3 = array([6, 7, 8])

@gif.frame
def plot(i):
	plt.figure()
	plt.imshow(sqrt(u[0]**2+u[1]**2).transpose(),cmap='Reds')

def macroscopic(fin):
	rho = sum(fin, axis=0)
	u = zeros((2,nx,ny))
	for i in range(0,9):
		u[0,:,:] += v[i,0]*fin[i,:,:]
		u[1,:,:] += v[i,1]*fin[i,:,:]
	u /= rho
	return rho, u

def equilibrium(rho,u):
	usqr = 3/2*(u[0,:,:]**2+u[1,:,:]**2)
	eq = zeros((9,nx,ny))
	for i in range(0,9):
		vu = 3*(v[i,0]*u[0,:,:]+v[i,1]*u[1,:,:])
		eq[i,:,:] = rho*t[i]*(1+vu+0.5*vu**2-usqr)
	return eq

def obstacle(x,y):
	return (x-cx)**2+(y-cy)**2<r**2

obstable_mask = fromfunction(obstacle,(nx,ny))

def inivel(d, x, y):
    return (1-d) * uLB * (1 + 1e-4*sin(y/ly*2*pi))

v0 = fromfunction(inivel, (2,nx,ny))

fin = equilibrium(1,v0)

frames = []
for time in range(0,maxIter):
	# PERIODIC RIGHT WALL CONDITIONS
	fin[col3,-1,:] = fin[col3,-2,:] 

	rho, u = macroscopic(fin)

	# INFLOW CONDITION
	u[:,0,:] = v0[:,0,:]
	rho[0,:] = (sum(fin[col2,0,:], axis=0)+2*sum(fin[col3,0,:], axis=0))/(1-u[0,0,:])

	# EQUILIBRIUM
	feq = equilibrium(rho,u)
	fin[[0,1,2],0,:] = feq[[0,1,2],0,:] + fin[[8,7,6],0,:] - feq[[8,7,6],0,:]

    # COLLISION
	fout = fin - omega * (fin - feq)

    # OBSTACLE
	for i in range(0,9):
		fout[i, obstable_mask] = fin[8-i, obstable_mask]

    # STREAMING
	for i in range(0,9):
		fin[i,:,:] = roll(roll(fout[i,:,:], v[i,0], axis=0),v[i,1], axis=1)

	if time%100==0:
		frame = plot(time)
		frames.append(frame)

savePath = path.join(getcwd(),'Re{:n}.gif'.format(Re))
gif.save(frames,savePath,duration=50)