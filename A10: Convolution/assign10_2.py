# Assignment 10: Part 2
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

# Q7:

b = []
with open('x1.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		a = row[0]
		a = a.replace(" ","")
		a = a.replace("i", "j")
		b.append(complex(a))
bshift = [0,0,0,0,0] + b
m = np.arange(1,840,1)
plt.figure()
plt.plot(m, b)
plt.title("The Zadoff Chu Sequence", size =16)
plt.xlabel("n")
plt.ylabel("y[n]")
plt.grid(True)
plt.show()

b = np.array(b)
c = np.array(bshift)
# y1=np.convolve(b,c)
y1 = np.fft.ifft( np.concatenate(( np.fft.fft(b),np.array([0]*5) ))*np.fft.fft(np.conj(c)))
plt.figure()
# plt.plot(np.arange(1682), y1)
plt.plot(m[:300], y1[:300])
plt.title("Correlation with a cyclic shifted version")
plt.ylim([-10,250])
plt.xlabel("n")
plt.grid(1)
plt.ylabel("y[n]")
plt.show()

# END OF CODE
