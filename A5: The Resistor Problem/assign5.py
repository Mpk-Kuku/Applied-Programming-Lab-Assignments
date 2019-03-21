#Assignment 5
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.linalg import lstsq
import scipy.integrate as spint
import mpl_toolkits.mplot3d.axes3d as p3
from scipy import ndimage

Nx = 25 #Size along x
Ny = 25 #Size along y
radius = 0.35 #Radius of central lead
Niter = 1500 #Number of iterations

phi = np.full((Nx,Ny),0.0)
# print(Ny//2)
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
Y, X = np.meshgrid(y,x)
# print(Y.shape, X.shape)
# print(Y)
ii = np.where(X*X + Y*Y <= radius*radius) #Where the wire is present
# print(origin)
# print(ii)
phi[ii] = 1.0 #This is true always.
# print(phi)

#Plotting a contour plot:
plt.figure()
cp = plt.contour(X,Y,phi.T)#,10)
plt.plot(x[ii[0]], y[ii[1]], 'ro')
plt.clabel(cp,inline=True, fontsize=7)
plt.title(r'Contour plot of phi')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# print(phi)
# k=np.array([[0,1,0],[1,0,1],[0,1,0]])
# print(k.shape)
errors=np.full(Niter,0.0)
for m in range(Niter):
	oldphi=phi.copy()
	# phi=(ndimage.convolve(oldphi,k,mode='constant',cval=0.0))/4.0 #This works too
	phi[1:-1,1:-1]=0.25*(phi[1:-1,0:-2] + phi[1:-1,2:] + phi[0:-2,1:-1] + phi[2:,1:-1])
	#Boundary conditions
	phi[1:-1,0]=phi[1:-1,1]
	phi[1:-1,-1]=phi[1:-1,-2]
	phi[0,1:-1]=0.0 #Because this part is grounded.
	phi[-1,1:-1]=phi[-2,1:-1]
	phi[ii]=1.0
	errors[m]=(np.abs(np.subtract(phi,oldphi))).max()
	# if(m%50==0):
	# 	print(errors[m])
#Now phi has almost steady values.

#Plotting a contour plot:
plt.figure()
cp = plt.contour(X,Y,phi.T,10)
plt.plot(x[ii[0]], y[ii[1]], 'ro')
plt.clabel(cp,inline=True, fontsize=7)
plt.title(r'Contour plot of phi after iterations')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# #Error plot:
# plt.plot(np.arange(30),errors[::50])
# plt.xlabel('Iteration x 50',size=20)
# plt.ylabel('errors',size=20)
# plt.title(r'Error plot for every 50th iteration')
# plt.show()

#Error plot in semilogy:
plt.semilogy(np.arange(30),errors[::50])
plt.xlabel('Iteration x 50',size=20)
plt.ylabel('errors',size=20)
plt.title(r'Error plot for every 50th iteration')
plt.show()

#Curve fitting:
#The error for each iteration is to be fitted by an exponential curve as described in the problem statement.
#all represents for all iterations, 500 represents iterations 500 till the end
error_all=np.log(errors)[:,None]
error_500=np.log(errors[500:])[:,None]
# print(error_500.shape,error_all.shape)

x_all=np.concatenate((np.ones((1500,1)),np.arange(1500)[:,None]),axis=1)
x_500=np.concatenate((np.ones((1000,1)),(np.arange(1000)[:,None] + 500)),axis=1)
# print(x_all.shape)

fit_all=lstsq(x_all,error_all)[0]
fit_500=lstsq(x_500,error_500)[0]
# print(fit_all.shape)
# print(fit_all)
# print(fit_500)

predicted_all=np.exp(fit_all[0])*np.exp(fit_all[1]*np.arange(1500))
predicted_500=np.exp(fit_500[0])*np.exp(fit_500[1]*np.arange(1500))

# #Error plot:
# plt.plot(np.arange(30),errors[::50],label='original error')
# plt.plot(np.arange(30),predicted_all[::50],label='Fit 1')
# plt.plot(np.arange(30),predicted_500[::50],label='Fit 2')
# plt.legend(loc='upper right')
# plt.xlabel('Iteration x 50',size=20)
# plt.ylabel('errors',size=20)
# plt.title(r'Error plot for every 50th iteration')
# plt.show()

#Error plot in semilogy:
plt.semilogy(np.arange(30),errors[::50],label='original error')
plt.semilogy(np.arange(30),predicted_all[::50],label='Predicted error from the fitting 1')
plt.semilogy(np.arange(30),predicted_500[::50],label='Predicted error from the fitting 2')
plt.legend(loc='upper right')
plt.xlabel('Iteration x 50',size=20)
plt.ylabel('errors',size=20)
plt.title(r'Error plot for every 50th iteration')
plt.show()


#3D plot of potential:
fig1 = plt.figure()#figure(4)
ax=p3.Axes3D(fig1)
plt.title('3D surface plot of the potential')
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap='viridis',lightsource=None)
fig1.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

#Calculating the current density by convolutiion:
# k=np.array([[0,1,0],[0,0,0],[0,-1,0]])
# Jx=(ndimage.convolve(phi,k,mode='constant',cval=0.0))/2.0
# k=np.array([[0,0,0],[1,0,-1],[0,0,0]])
# Jy=(ndimage.convolve(phi,k,mode='constant',cval=0.0))/2.0

#Calculating the current density
Jx = np.full((Nx, Ny), 0.0)
Jy = np.full((Nx, Ny), 0.0)
Jx[1:-1, 1:-1] = 0.5*(phi[:-2,1:-1] - phi[2:,1:-1])
Jy[1:-1, 1:-1] = 0.5*(phi[1:-1,:-2] - phi[1:-1,2:])


# print(Jx[::-1,:].shape,Jx.shape)
#Quiver plot of the current density:
plt.quiver(Y, X, Jy[0:-1,:], Jx[0:-1,:])
plt.plot(x[ii[0]], y[ii[1]], 'ro',label = "Points with V = 1.0 volt")
plt.title("Vector plot of the Current flow", size = 16)
plt.xlabel("Grounded side of the plate")
plt.legend(loc = "upper right")
plt.show()
