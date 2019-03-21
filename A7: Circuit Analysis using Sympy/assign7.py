#Assignment 7
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.linalg import lstsq
import scipy.integrate as spint
import mpl_toolkits.mplot3d.axes3d as p3
from scipy import ndimage
import sympy as spy

s = spy.symbols('s')

#Gives the s-domain output voltage of the given low pass filter
def lowpass(R1,R2,C1,C2,G,Vi):
	A = spy.Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0],[0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
	b = spy.Matrix([0,0,0,Vi/R1])
	V = A.inv()*b #Output voltage is V[3]
	return (A,b,V)

#Gives the s-domain output voltage of the given high pass filter
def highpass(R1,R3,C1,C2,G,Vi):
	A = spy.Matrix([[0,0,G,-1],[s*C2*R3,-1-s*C2*R3,0,0],[0,G,-G,-1],[1+s*C2*R1+s*C1*R1,-s*C2*R1,0,-1]])
	b = spy.Matrix([0,0,0,s*C1*R1*Vi])
	V = A.inv()*b #Output voltage is V[3]
	return (A,b,V)

#To get the coefficients of the s-domain expression
#Mainly used to convert a function to time domain from s domain equation.
def get_coeffs(expr):
	num,den = expr.as_numer_denom()
	return [spy.Poly(num,s).all_coeffs(), spy.Poly(den,s).all_coeffs()]

#The example in the problem statement with its magnitude plot:
A,b,V = lowpass(10000,10000,1e-9,1e-9,1.586,1)
# print ('G=1000')
Vo=V[3]
print(Vo)
w=np.logspace(0,8,801)
ss=1j*w
hf = spy.lambdify(s,Vo,'numpy')
v=hf(ss)
plt.loglog(w,abs(v),lw=2)
plt.xlabel('omega, w')
plt.ylabel('Magnitude, x')
plt.grid(True)
plt.show()

#Q1: input is unit step.
A,b,V = lowpass(10000,10000,1e-9,1e-9,1.586,1/s)
# print ('G=1000')
Vo=V[3]
# print(Vo)
w=np.logspace(0,8,801)
ss=1j*w
hf = spy.lambdify(s,Vo,'numpy')
v=hf(ss)
plt.loglog(w,abs(v),lw=2)
plt.xlabel('omega, w')
plt.ylabel('Magnitude, x')
plt.grid(True)
plt.show()

#Q2: Input is given in question:
#First term:
# Num = np.poly1d([2000*np.pi])
# Den = np.poly1d([1,0,(2000*np.pi)**2])
# Vi1 = sp.lti(Num,Den)
# #Second term:
# Num = np.poly1d([1,0])
# Den = np.poly1d([1,0,(2e6*np.pi)**2])
# Vi2 = sp.lti(Num,Den)
# Vi = Vi1 + Vi2
# print(Vi)
Vi = (s**3 + (2e3*np.pi*s**2) + (4e6*(np.pi**2)*s + 8e15*np.pi**3))/(s**4 + 4e12*np.pi*np.pi*s**2 + 16e18*np.pi**4)
A,b,V = lowpass(10000,10000,1e-9,1e-9,1.586,Vi)
# print ('G=1000')
Vo=V[3]
# print(Vo)
w=np.logspace(0,8,801)
ss=1j*w
hf = spy.lambdify(s,Vo,'numpy')
v=hf(ss)
#Magnitude plot of the output:
plt.loglog(w,abs(v),lw=2)
plt.xlabel('omega, w')
plt.ylabel('Magnitude, x')
plt.grid(True)
plt.show()

# #To get the time domain output and the plot:
# Vo = spy.simplify(Vo)
# print(Vo)
# n,d=get_coeffs(Vo)
# print(n)
# print(d)
# n=[float(x) for x in n]
# d=[float(x) for x in d]
# Vo_s = sp.lti(n,d)
# # print(n,d)
# # print("hello")
# t,v_o = sp.impulse(Vo_s,None,np.linspace(0,10,10)) 
# print(v_o)
# # Output time domain signal:
# plt.plot(t,v_o,lw=2)
# plt.xlabel('time, t')
# plt.ylabel('Output signal')
# plt.grid(True)
# plt.show()

#Q3:
# Transfer function of the given high pass filter is plotted:
A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,1)
# print ('G=1000')
Vo=V[3]
# print(Vo)
w=np.logspace(0,8,801)
ss=1j*w
hf = spy.lambdify(s,Vo,'numpy')
v=hf(ss)
plt.loglog(w,abs(v),lw=2)
plt.xlabel('omega, w')
plt.ylabel('Magnitude, x')
plt.grid(True)
plt.show()

#Q4:
#To plot and observe the magnitude response of the output when the input is a damped sinusoid of a chosen frequency.
#Damped sinusoid: sin 2000t e**(-t)
Vi=2e3/((s+1)**2 + 4e6)
n,d=get_coeffs(Vi)
# print(n)
# print(d)
n=[float(x) for x in n]
d=[float(x) for x in d]
Vi_s = sp.lti(n,d)
# print(n,d)
# print("hello")
t,v_i = sp.impulse(Vi_s,None,np.linspace(0,10,10001)) 
# Plot of input time domain signal:
plt.plot(t,v_i,lw=2)
plt.xlabel('time, t')
plt.ylabel('Input signal')
plt.grid(True)
plt.show()

A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,Vi)
# print ('G=1000')
Vo=V[3]
# print(Vo)
w=np.logspace(0,8,801)
ss=1j*w
hf = spy.lambdify(s,Vo,'numpy')
v=hf(ss)
plt.loglog(w,abs(v),lw=2)
plt.xlabel('omega, w')
plt.ylabel('Magnitude, x')
plt.grid(True)
plt.show()
# print(hf)

n,d=get_coeffs(Vo)
# print(n)
# print(d)
n=[float(x) for x in n]
d=[float(x) for x in d]
Vo_s = sp.lti(n,d)
# print(n,d)
# print("hello")
t,v_o = sp.impulse(Vo_s,None,np.linspace(0,10,10001)) 
# Output time domain signal:
plt.plot(t,v_o,lw=2)
plt.xlabel('time, t')
plt.ylabel('Output signal')
plt.grid(True)
plt.show()

# #Damped sinusoid: sin t e**(-t)
# Vi=1/((s+1)**2 + 1)
# n,d=get_coeffs(Vi)
# # print(n)
# # print(d)
# n=[float(x) for x in n]
# d=[float(x) for x in d]
# Vi_s = sp.lti(n,d)
# # print(n,d)
# # print("hello")
# t,v_i = sp.impulse(Vi_s,None,np.linspace(0,10,10001)) 
# # Plot of input time domain signal:
# plt.plot(t,v_i,lw=2)
# plt.xlabel('time, t')
# plt.ylabel('Input signal')
# plt.grid(True)
# plt.show()

# A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,Vi)
# # print ('G=1000')
# Vo=V[3]
# # print(Vo)
# w=np.logspace(0,8,801)
# ss=1j*w
# hf = spy.lambdify(s,Vo,'numpy')
# v=hf(ss)
# plt.loglog(w,abs(v),lw=2)
# plt.xlabel('omega, w')
# plt.ylabel('Magnitude, x')
# plt.grid(True)
# plt.show()
# # print(hf)

# n,d=get_coeffs(Vo)
# # print(n)
# # print(d)
# n=[float(x) for x in n]
# d=[float(x) for x in d]
# Vo_s = sp.lti(n,d)
# # print(n,d)
# # print("hello")
# t,v_o = sp.impulse(Vo_s,None,np.linspace(0,10,10001)) 
# # Output time domain signal:
# plt.plot(t,v_o,lw=2)
# plt.xlabel('time, t')
# plt.ylabel('Output signal')
# plt.grid(True)
# plt.show()

# #Damped sinusoid: sin 2e10t e**(-t)
# Vi=2e10/((s+1)**2 + 4e20)
# n,d=get_coeffs(Vi)
# # print(n)
# # print(d)
# n=[float(x) for x in n]
# d=[float(x) for x in d]
# Vi_s = sp.lti(n,d)
# # print(n,d)
# # print("hello")
# t,v_i = sp.impulse(Vi_s,None,np.linspace(0,10,10001)) 
# # Plot of input time domain signal:
# plt.plot(t,v_i,lw=2)
# plt.xlabel('time, t')
# plt.ylabel('Input signal')
# plt.grid(True)
# plt.show()

# A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,Vi)
# # print ('G=1000')
# Vo=V[3]
# # print(Vo)
# w=np.logspace(0,8,801)
# ss=1j*w
# hf = spy.lambdify(s,Vo,'numpy')
# v=hf(ss)
# plt.loglog(w,abs(v),lw=2)
# plt.xlabel('omega, w')
# plt.ylabel('Magnitude, x')
# plt.grid(True)
# plt.show()
# # print(hf)

# n,d=get_coeffs(Vo)
# # print(n)
# # print(d)
# n=[float(x) for x in n]
# d=[float(x) for x in d]
# Vo_s = sp.lti(n,d)
# # print(n,d)
# # print("hello")
# t,v_o = sp.impulse(Vo_s,None,np.linspace(0,10,10001)) 
# # Output time domain signal:
# plt.plot(t,v_o,lw=2)
# plt.xlabel('time, t')
# plt.ylabel('Output signal')
# plt.grid(True)
# plt.show()


#Q5:
#Unit step response of the high pass filter:
A,b,V = highpass(10000,10000,1e-9,1e-9,1.586,1/s)
# print ('G=1000')
Vo=V[3]
# print(Vo)
w=np.logspace(0,8,801)
ss=1j*w
hf = spy.lambdify(s,Vo,'numpy')
v=hf(ss)
plt.loglog(w,abs(v),lw=2)
plt.xlabel('omega, w')
plt.ylabel('Magnitude, x')
plt.grid(True)
plt.show()

# THE END
