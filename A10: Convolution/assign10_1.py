# Assignment 10: Part 1
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.linalg import lstsq
import scipy.integrate as spint
import mpl_toolkits.mplot3d.axes3d as p3
from scipy import ndimage
import sympy as spy
# import pandas as pd
import csv

# Q1:
b = []#np.full(0,0.0)
with open("h.csv") as csv_file:
	csv = csv.reader(csv_file, delimiter = ',')
	for row in csv:
		# b=np.concatenate(b,float(row[0]))
		b.append(float(row[0]))
		# print(type(row[0]))
	print(b)

# Q2:
w,h = sp.freqz(b)
# plt.figure()
# plt.plot(w, abs(h), 'r')
# # plt.title("Magnitude response of FIR filter")
# plt.ylabel("Magnitude")
# plt.xlabel(r"$\omega$")
# plt.show()
plt.subplot(2,1,1)
plt.plot(w,np.abs(h),'r')
# plt.xlim([-10,10])
plt.ylabel("Magnitude")
plt.title("Magnitude and phase response of FIR filter")
# plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w, np.unwrap(np.angle(h)),'g')#np.unwrap
# plt.xlim([-10,10])
plt.ylabel("Phase",size=16)
plt.xlabel(r"$\omega$")
# plt.grid(True)
# savefig("fig9-1.png")
plt.show()

# Q3:
n= np.arange(1025)
n = n[1:]
# print(n)
x = np.cos(0.2*np.pi*n) + np.cos(0.85*np.pi*n)

plt.figure()
plt.plot(n,x,'g-')
plt.title(r"Graph of $cos(0.2\pi n) + cos(0.85\pi n)$", size=16)
plt.ylabel("x[n]")
plt.xlabel("n")
plt.grid(True)
plt.show()

# Q4: Linear convolution
# x = np.concatenate((np.array([0]*(11), dtype = float),x))
y = np.array([0]*1035, dtype = float)
b = np.array(b)
# print(x)
# print(b)
# print(y)
y2 = np.convolve(x,b)
# for i in range(11,1035):
# 	y[i]=0
# 	for k in range(12):
# 		y[i] += b[k]*x[i-k]
plt.figure()
# print(n.shape, y.shape, x.shape)
plt.plot(np.concatenate((n,np.arange(1025,1036))),y2,'r-')
plt.title("Output of filter: Linear convolution", size=16)
plt.ylabel("y[n]")
plt.xlabel("n")
plt.grid(True)
plt.show()
# Note that only the low frequency component will remain.

# Q5: Circular convolution
# x1=np.fft.fft(x)
# print(len(x))
# x2 = np.fft.fft(np.concatenate((b,np.array([0]*(len(x)-len(b)), dtype = float))))
# print(x1.shape, x2.shape)
y1 = np.fft.ifft(np.multiply(np.fft.fft(x) , np.fft.fft(np.concatenate((b,np.array([0]*(len(x)-len(b)), dtype = float))))))
print(y1.shape)
plt.figure()
# plt.plot(np.concatenate((n,np.arange(1025,1036))),y1,'g-')
plt.plot(n,y1,'g-')
plt.title("Output of filter: Circular convolution", size=16)
plt.ylabel("y[n]")
plt.xlabel("n")
plt.grid(True)
plt.show()

# Q6: Linear convolution using circular convolution
P = len(b)
L = 2**6

# y_2 = np.full(len(x)+P-1,0.0)
y_2 = np.full(len(x)-1+L,0.0+0.0j)
# for k in np.arange(0,2**10,L):
# 	w = np.convolve(b, x[k:k+L])
# 	y_2[k:k+P-1] += w[0:P-1]
# 	y_2[k+P:k+P+L-1] += w[P:L+P-1]
# 	# y_2[k:k+P+L-1] += w[0:L+P-1]
pos=0
x=np.concatenate((np.array([0]*(P-1)),x))
x=np.concatenate((x,np.array([0]*(P-1+L))))
while(pos+L <=len(x)):
# for k in np.arange(0,2**10,L):
	# w = np.convolve(b, x[k:k+L])
	# abc1=np.fft.fft(x[k:k+L+P])
	# abc2=np.fft.fft(np.concatenate((b,np.array([0]*L, dtype = float))))
	# print(abc1.shape,abc2.shape)
	# if(abc1.shape==abc2.shape):
		# x1=x[k:k+L+1]; #x1[:P]=np.array([0]*P)
		# x1=np.concatenate((x[k:k+L+1],np.array([0]*(P-1))))
		# x1 = np.concatenate((np.array([0]*(P-1)),x1))
	x1=x[pos:pos+L]
	w = np.fft.ifft(np.multiply(np.fft.fft(x1) , np.fft.fft(np.concatenate((b,np.array([0]*(L-P), dtype = float))))))
	# print(w.shape)
	# y_2[k:k+P-1] = y_2[k:k+P-1] + w[0:P-1]
	# y_2[k+P:k+P+L-1] += w[P:L+P-1]
	y_2[pos:pos+L-P]=w[P:]
		# y_2[k:k+L] += w[P:P+L]#L+P-1]
	pos+=(L-P-1)
	# y_2[k:k+L] += w[0:L]
	# print(pos)


# P = 12
# N = 1024
# L = 64
# n = np.arange(1,1025,1)
# x = np.cos(0.2*np.pi*n) + np.cos(0.85*np.pi*n)
# y_full = np.array([])
# x = np.concatenate((x,np.array([0]*48)))
# for i in range(20):
# 	xi = x[(53*i):(53*i)+L]
# 	yi1 = np.fft.ifft( np.fft.fft(xi)*np.fft.fft(np.concatenate((b,np.array([0]*(len(xi) - len(b)))))))
# 	yi1 = yi1[(P-1):]
# 	y_full = np.concatenate((y_full,yi1))
plt.figure()	
# plt.plot(n,y_full[:1024])
plt.plot(np.arange(1,len(y_2)+1),y_2)
plt.xlabel("n",size=14)
plt.ylabel("y[n]",size=14)
plt.title("Linear convolution using circular convolution",size=16)
plt.show()
# plt.plot(n[:50],y_full[:50])
# plt.title("Linear convolution using circular convolution",size=16)
# plt.xlabel("n",size=14)
# plt.ylabel("y[n]",size=14)
# plt.show()
