# Assignment 4 
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.linalg import lstsq
import scipy.integrate as spint

def f1(x): #To calculate exp(x)
	return np.exp(x)

def f2(x): #To calculate cos(cos (x))
	return np.cos(np.cos(x))

def u1(x,k):
	return f1(x)*np.cos(k*x)

def v1(x,k):
	return f1(x)*np.sin(k*x)

def u2(x,k):
	return f2(x)*np.cos(k*x)

def v2(x,k):
	return f2(x)*np.sin(k*x)

#Taking x to be 600 points in the given range
x = np.linspace(-2*np.pi,4*np.pi,600)
f1_y = f1(x)
f2_y = f2(x)

#Plotting the functions:
plt.semilogy(x,f1_y)
plt.grid(True)
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$exp(x)$',size=20)
plt.title(r'The first function in semi log scale along y axis')
plt.show()

plt.plot(x,f1_y)
plt.grid(True)
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$exp(x)$',size=20)
plt.title(r'The first function')
plt.show()

plt.plot(x,f2_y)
plt.grid(True)
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$cos(cos(x))$',size=20)
plt.title(r'The second function')
plt.show()

#Finding the first fourier coefficients of both functions by integration
a0_1=(spint.quad(f1,0,2*np.pi)[0])/(2*np.pi)
a0_2=(spint.quad(f2,0,2*np.pi)[0])/(2*np.pi)
# print(a0_1,a0_2)
# a_1=np.full(25,0.0)
# b_1=np.full(25,0.0)
ab_1=np.full(51,0.0)
# a_2=np.full(25,0.0)
# b_2=np.full(25,0.0)
ab_2=np.full(51,0.0)
ab_1[0]=a0_1
ab_2[0]=a0_2
# print(ab_2[0])

#Finding the next 25 fourier coefficients of both functions by integration:
for i in range (25):
	# a_1[i]=(spint.quad(u1,0,2*np.pi,args=(i+1))[0])/(np.pi)
	# b_1[i]=(spint.quad(v1,0,2*np.pi,args=(i+1))[0])/(np.pi)
	# a_2[i]=(spint.quad(u2,0,2*np.pi,args=(i+1))[0])/(np.pi)
	# b_2[i]=(spint.quad(v2,0,2*np.pi,args=(i+1))[0])/(np.pi)
	ab_1[2*i+1] = (spint.quad(u1,0,2*np.pi,args=(i+1))[0])/(np.pi)
	ab_1[2*i+2] = (spint.quad(v1,0,2*np.pi,args=(i+1))[0])/(np.pi)
	ab_2[2*i+1] = (spint.quad(u2,0,2*np.pi,args=(i+1))[0])/(np.pi)
	ab_2[2*i+2] = (spint.quad(v2,0,2*np.pi,args=(i+1))[0])/(np.pi)
	# print(a_2[i],b_2[i])
# print(a_1)
# print(b_1)
# print(a_2)
# print(b_2)
# print(ab_1)
ab_x = np.arange(51)
ab_1 = np.abs(ab_1)
ab_2 = np.abs(ab_2)
# print(ab_2)

# plt.semilogy(ab_x,ab_1,'ro')
# plt.grid(True)
# plt.ylabel(r'Fourier coefficients',size=20)
# plt.title(r'Fourier coefficients of the first function')
# plt.show()

# plt.loglog(ab_x,ab_1,'ro')
# plt.grid(True)
# plt.ylabel(r'Fourier coefficients',size=20)
# plt.title(r'Fourier coefficients of the first function')
# plt.show()

# plt.semilogy(ab_x,ab_2,'ro')
# plt.grid(True)
# plt.ylabel(r'Fourier coefficients',size=20)
# plt.title(r'Fourier coefficients of the second function')
# plt.show()

# plt.loglog(ab_x,ab_2,'ro')
# plt.grid(True)
# plt.ylabel(r'Fourier coefficients',size=20)
# plt.title(r'Fourier coefficients of the second function')
# plt.show()

#Finding the fourier coefficients by least squares fitting:
x_ls = np.linspace(0,2*np.pi,401)
x_ls = x_ls[:-1]
b_1 = f1(x_ls)
b_2 = f2(x_ls)
A = np.full((400,51),1.0)
for k in range(1,26):
	A[:,2*k-1] = np.cos(k*x_ls)
	A[:,2*k] = np.sin(k*x_ls)
c_1 = lstsq(A,b_1)[0]
c_2 = lstsq(A,b_2)[0]
# We expect that the accuracy of this method will be less since least squares work best with gaussian noise.
# print(c_1)
# print(c_2)

#Plotting the obtained coefficients to compare both methods:
plt.semilogy(ab_x,ab_1,'ro')
plt.semilogy(ab_x,np.abs(c_1),'go')
plt.grid(True)
plt.xlabel(r'Index',size=20)
plt.ylabel(r'Fourier coefficients',size=20)
plt.title(r'Fourier coefficients of the first function')
plt.show()

plt.loglog(ab_x,ab_1,'ro')
plt.loglog(ab_x,np.abs(c_1),'go')
plt.grid(True)
plt.xlabel(r'Index',size=20)
plt.ylabel(r'Fourier coefficients',size=20)
plt.title(r'Fourier coefficients of the first function')
plt.show()

plt.semilogy(ab_x,ab_2,'ro')
plt.semilogy(ab_x,np.abs(c_2),'go')
plt.grid(True)
plt.xlabel(r'Index',size=20)
plt.ylabel(r'Fourier coefficients',size=20)
plt.title(r'Fourier coefficients of the second function')
plt.show()

plt.loglog(ab_x,ab_2,'ro')
plt.loglog(ab_x,np.abs(c_2),'go')
plt.grid(True)
plt.xlabel(r'Index',size=20)
plt.ylabel(r'Fourier coefficients',size=20)
plt.title(r'Fourier coefficients of the second function')
plt.show()

Ac1 = np.dot(A,c_1)
Ac2 = np.dot(A,c_2)

md1 = np.max(np.abs(c_1 - ab_1))
md2 = np.max(np.abs(c_2 - ab_2))
print("The maximum deviation for the first function is ",md1)
print("The maximum deviation for the first function is ",md2)

# print(Ac1)
# print(Ac2)

# print(A.shape, c_1.shape, c_2.shape, Ac1.shape, Ac2.shape)

plt.semilogy(x_ls,b_1,'ro',markersize=3)
plt.semilogy(x_ls,Ac1,'go',markersize=3)
plt.grid(True)
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$exp(x)$',size=20)
plt.title(r'The first function in semi log scale along y axis')
plt.show()

plt.plot(x_ls,b_1,'ro',markersize=3)
plt.plot(x_ls,Ac1,'go',markersize=3)
plt.grid(True)
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$exp(x)$',size=20)
plt.title(r'The first function')
plt.show()

plt.plot(x_ls,b_2,'ro',markersize=5)
plt.plot(x_ls,Ac2,'go',markersize=3)
plt.grid(True)
plt.xlabel(r'$x$',size=20)
plt.ylabel(r'$cos(cos(x))$',size=20)
plt.title(r'The second function')
plt.show()
