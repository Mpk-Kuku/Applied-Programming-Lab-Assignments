#Assignment 6
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.linalg import lstsq
import scipy.integrate as spint
import mpl_toolkits.mplot3d.axes3d as p3
from scipy import ndimage

#The definition of functions given in the problem statement
def f_t(t,decay,freq):
	ans=np.cos(freq*t)*np.exp((-1*decay)*t)
	ii=np.where(t<0)
	ans[ii]=0
	return ans

def v_t(t):
	ii=np.where(t<0)
	ans=np.cos(1000*t)-np.cos(1e6*t)
	ans[ii]=0
	return ans

#Q1:
num=np.poly1d([1])
den=np.poly1d([1,0,2.25])
H=sp.lti(num,den) #Defining the transfer function of the LTI system
t,h=sp.impulse(H,None,np.linspace(0,50,10001)) #To get the impulse response from transfer function
# plt.plot(t,h)
# plt.show()
f1=f_t(t,0.5,1.5)
t,y1,svec=sp.lsim(H,f1,t)
plt.plot(t,y1)
plt.xlabel('time, t')
plt.ylabel('position, x')
plt.show()

#Q2:
f2=f_t(t,0.05,1.5)
t,y2,svec=sp.lsim(H,f2,t) #Convolves impulse respons
#The plot of the output:
plt.plot(t,y2)
plt.xlabel('time, t')
plt.ylabel('position, x')
plt.show()

#Q3:
rng=np.arange(1.4,1.6,0.05)
for i in rng:
	f=f_t(t,0.05,i)
	t,y,svec=sp.lsim(H,f,t)
	plt.plot(t,y,label=i)
	# plt.show()
plt.legend(loc='upper left')
plt.title('Time response of the spring for various frequencies of cosines')
plt.xlabel('time, t')
plt.ylabel('position, x')
plt.show()
#Note that the frequency matches the natural frequency at 1.5!

#Q4:
num=np.poly1d([1,0,2,0])
den=np.poly1d([1,0,3,0,0])
X=sp.lti(num,den)
t,x=sp.impulse(X,None,np.linspace(0,20,10001)) #X is converted to time domain
plt.plot(t,x)
plt.title('Plot of x vs t in the coupled spring problem')
plt.xlabel('time, t')
plt.ylabel('position, x')
plt.show()

num=np.poly1d([2,0])
den=np.poly1d([1,0,3,0,0])
Y=sp.lti(num,den) 
t,y=sp.impulse(Y,None,np.linspace(0,20,10001)) #Converted to time domain
plt.plot(t,y)
plt.title('Plot of y vs t in the coupled spring problem')
plt.xlabel('time, t')
plt.ylabel('position, y')
plt.show()

#Plotting x and y together:
plt.plot(t,x,label='x')
plt.plot(t,y,label='y')
plt.title('Coupled spring problem')
plt.legend(loc='upper left')
plt.xlabel('time, t')
plt.ylabel('position')
plt.show()

# Q5:
#Bode plot of the given function
num=np.poly1d([1e6])
den=np.poly1d([1e-6,100,1e6])
H=sp.lti(num,den)
w,S,phi=H.bode()
plt.subplot(2,1,1)
# plt.title('Magnitude plot')
plt.semilogx(w,S)
plt.xlabel('w')
plt.ylabel('Magnitude')
plt.subplot(2,1,2)
# plt.title('Phase plot')
plt.semilogx(w,phi)
plt.xlabel('w')
plt.ylabel('Phase')
plt.show()

#Q6:
#Converting it to time domain and plotting:
t,h=sp.impulse(H,None,np.linspace(0,10*1e-3,100001))
v=v_t(t)
t,out,svec=sp.lsim(H,v,t)
plt.plot(t,out)
plt.title('Output voltage vs time')
plt.xlabel('time, t')
plt.ylabel('voltage, v0')
plt.show()
