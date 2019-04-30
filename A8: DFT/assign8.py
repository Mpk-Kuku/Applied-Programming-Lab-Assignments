# Assignment 8
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.linalg import lstsq
import scipy.integrate as spint
import mpl_toolkits.mplot3d.axes3d as p3
from scipy import ndimage
import sympy as spy

x=np.random.rand(100)
X=np.fft.fft(x)
y=np.fft.ifft(X)
print(np.abs(x-y).max())

x=np.linspace(0,2*np.pi,128)
y=np.sin(5*x)
Y=np.fft.fft(y)
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.abs(Y),lw=2)
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of $\sin(5t)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(np.unwrap(np.angle(Y)),lw=2)
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
# savefig("fig9-1.png")
plt.show()

x=np.linspace(0,2*np.pi,129)
x=x[:-1]
y=np.sin(5*x)
Y=np.fft.fftshift(np.fft.fft(y))/128.0
w = np.linspace(-64,64,129)
w=w[:-1]
plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y),lw=2)
plt.xlim([-10,10])
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of $\sin(5t)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
ii=np.where(np.abs(Y)>1e-3)
plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
plt.xlim([-10,10])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
# savefig("fig9-2.png")
plt.show()

# Now we look at AM modulation. The function we want to analyse is 
# f(t)=(1+0.1 cos(t)) cos(10t)

t=np.linspace(0,2*np.pi,129)
t=t[:-1]
y=np.multiply((1+0.1*np.cos(t)),np.cos(10*t))
Y=np.fft.fftshift(np.fft.fft(y))/128.0
w = np.linspace(-64,64,129)
w=w[:-1]
plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y),lw=2)
plt.xlim([-15,15])
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
# ii=np.where(np.abs(Y)>1e-3)
# plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
plt.xlim([-15,15])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
# savefig("fig9-2.png")
plt.show()

t=np.linspace(-4*np.pi,4*np.pi,513)
t=t[:-1]
y=np.multiply((1+0.1*np.cos(t)),np.cos(10*t))
Y=np.fft.fftshift(np.fft.fft(y))/512.0
w = np.linspace(-64,64,513)
w=w[:-1]
plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y),lw=2)
plt.xlim([-15,15])
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
# ii=np.where(np.abs(Y)>1e-3)
# plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
plt.xlim([-15,15])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
# savefig("fig9-2.png")
plt.show()

#Question 2:

#sin(t)^3:
t=np.linspace(-4*np.pi,4*np.pi,513)
t=t[:-1]
y=np.multiply(np.multiply(np.sin(t),np.sin(t)),np.sin(t))
Y=np.fft.fftshift(np.fft.fft(y))/512.0
w = np.linspace(-64,64,513)
w=w[:-1]
plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y),lw=2)
plt.xlim([-15,15])
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of sin(t)^3")#$\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
# ii=np.where(np.abs(Y)>1e-3)
# plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
plt.xlim([-15,15])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
# savefig("fig9-2.png")
plt.show()
#This is correct because we know that frequency components 1 and 3 exist in sin (t)^3.

#cos(t)^3:
t=np.linspace(-4*np.pi,4*np.pi,513)
t=t[:-1]
y=np.multiply(np.multiply(np.cos(t),np.cos(t)),np.cos(t))
Y=np.fft.fftshift(np.fft.fft(y))/512.0
w = np.linspace(-64,64,513)
w=w[:-1]
plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y),lw=2)
plt.xlim([-15,15])
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of cos(t)^3")#$\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
# ii=np.where(np.abs(Y)>1e-3)
# plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
plt.xlim([-15,15])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
# savefig("fig9-2.png")
plt.show()
#This is correct because we know that frequency components 1 and 3 exist in cos(t)^3.

#Q3:

#cos(20t+5cos(t)):
t=np.linspace(-4*np.pi,4*np.pi,513)
t=t[:-1]
y=np.cos(20*t+5*np.cos(t))
Y=np.fft.fftshift(np.fft.fft(y))/512.0
w = np.linspace(-64,64,513)
w=w[:-1]
plt.figure()
plt.subplot(3,1,1)
plt.plot(w,abs(Y),lw=2)
plt.xlim([-40,40])
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of 20t+5cos(t)")#$\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
# ii=np.where(np.abs(Y)>1e-3)
# plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
plt.xlim([-40,40])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
plt.subplot(3,1,3)
# plt.plot(w,np.angle(Y),'ro',lw=2)
ii=np.where(np.abs(Y)>1e-3)
plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
plt.xlim([-40,40])
plt.ylabel(r"Phase of $Y$")#,size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
# savefig("fig9-2.png")
plt.show()

# What is happenning?

#Q4:
Time_start=-200
Time_end=200
N_pts=1024
t=np.linspace(Time_start,Time_end,N_pts+1)
t=t[:-1]
y=np.exp(-0.5*np.multiply(t,t))
Y=np.fft.fftshift(np.fft.fft(y))/float(N_pts)
w = np.linspace(-64,64,N_pts+1)
w=w[:-1]
plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y),lw=2)
plt.xlim([-30,30])
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of the gaussian function")#$\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=1)
# ii=np.where(np.abs(Y)>1e-3)
# plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
plt.xlim([-30,30])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
# savefig("fig9-2.png")
plt.show()

# Time_start=-10, Time_end=10 doesn't give a smooth gaussian.
# Time_start=-20, Time_end=20 gives a smooth gaussian.
# As we increase time range, higher frequencies come in to the spectrum 
# and the peak at highest frequency decreases.
