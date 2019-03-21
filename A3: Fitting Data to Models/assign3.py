#Assignment 3
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.linalg import lstsq
file_data = np.loadtxt("fitting.dat")
# print(file_data)

# Our funtion approximation:
def fitting_func_g(t,C1,C2):
	res=C1*sp.jn(2,t)+C2*t
	return res

correct_C1 = 1.05
correct_C2 = (-0.105)
x_real = np.linspace(0,10,101)
y_real = fitting_func_g(x_real,correct_C1,correct_C2)


# Plotting data that has to be fitted:
plt.plot(x_real,file_data[:,1],label='standard_deviation = 0.100')
plt.plot(x_real,file_data[:,2],label='standard_deviation = 0.056')
plt.plot(x_real,file_data[:,3],label='standard_deviation = 0.032')
plt.plot(x_real,file_data[:,4],label='standard_deviation = 0.018')
plt.plot(x_real,file_data[:,5],label='standard_deviation = 0.010')
plt.plot(x_real,file_data[:,6],label='standard_deviation = 0.006')
plt.plot(x_real,file_data[:,7],label='standard_deviation = 0.003')
plt.plot(x_real,file_data[:,8],label='standard_deviation = 0.002')
plt.plot(x_real,file_data[:,9],label='standard_deviation = 0.001')
plt.plot(x_real,y_real,label='true value')
plt.legend(loc='upper right')
plt.xlabel(r'$t$',size=20)
plt.ylabel(r'$g(t)+noise$',size=20)
plt.title(r'The true value and the data to be fitted')
plt.show()

y_first_column = file_data[:,1]
# # std_dev=np.std(y_first_column) #--> gives the standard deviation of y_first_column
# # We need the standard deviation of the noise, which we know is 0.1
# # print(x_real.shape,y_first_column.shape)

plt.errorbar(x_real[::5],y_first_column[::5],0.1,fmt='ro',markersize=5, label="error bars") 
#Used the std_dev of noise.
# plt.errorbar(x_real[::5],y_first_column[::5],std_dev,fmt='ro--',markersize=3, label="error bars") 
plt.plot(x_real,y_real,label="Real function")
plt.legend(loc='upper right')
plt.xlabel(r'$t$',size=20)
plt.ylabel(r'$g(t)$',size=20)
plt.title(r'Real function and the first noisy data (standard deviation = 0.1)')
plt.show()

J2_x_real = sp.jn(2,x_real)
# print(J2_x_real.shape)
M = np.c_[J2_x_real,x_real]
#Take A_0=1.05, B_0=(-0.105)
C_0 = np.array([1.05,-0.105])
M_p = np.dot(M,C_0)
# print(M.shape,M_p.shape)
# print(M_p.shape,y_real.shape)
#This should be compared with y_real.
if np.array_equal(M_p,y_real):
	print("Verified that they are both same.")
else:
	print("Debug the code. :P ")

A_values = np.arange(0,2.01,0.1)
B_values = np.arange(-0.2,0.01,0.01)
A_val , B_val = np.meshgrid(A_values,B_values)
# print(A_val.shape)
# print(B_val.shape)
A_size = A_values.size
B_size = B_values.size
A_val_3d = A_val[:,:,np.newaxis]
B_val_3d = B_val[:,:,np.newaxis]
# print(A_val_3d.shape,B_val_3d.shape)
# print(A_size,B_size)
# print((A_values*B_values).shape) 
# g_first_column = fitting_func_g(x_real, A_values, B_values)
# print(g_first_column.shape)

# Vectorised implementation:
errors = (np.square(y_first_column - fitting_func_g(x_real, A_val_3d, B_val_3d))).mean(axis=2)
errors = np.transpose(errors)
# print(errors.shape)

# Not vectorised:
# error = np.full((A_size,B_size),0.0)
# for a in range (A_size):
# 	for b in range (B_size):
# 		error[a][b]=(np.square(y_first_column - fitting_func_g(x_real, A_values[a], B_values[b]))).mean(axis=None);
# 		# print(error[a][b])
# print(error.shape)

#Contour plot:
plt.figure()
cp = plt.contour(A_val,B_val,errors)
# plt.legend(loc='center right')
plt.clabel(cp,cp.levels[0:5],inline=True)
plt.title(r'Contour plot of mean squared error of first column of data')
# plt.plot(correct_C1,correct_C2,'bo',label = 'minimum')
plt.scatter(correct_C1,correct_C2,color='r')
plt.xlabel('A')
plt.ylabel('B')
plt.annotate("True value",xy=(correct_C1,correct_C2))
plt.show()

sigma=np.logspace(-1,-3,9) # noise stdev
# print(sigma)

#Solution for all columns:
sol = np.full((0,2),0.0)
for i in range(1,10):
	y_column = file_data[:,i]
	sol_i = lstsq(M,y_column)[0]
	# print(sol_i.shape)
	sol=np.concatenate((sol,sol_i[None,:]),axis=0)
# print(sol)
real_ans = np.array([correct_C1,correct_C2])
# print(real_ans)
observed_error = abs(sol - real_ans)
# print(observed_error)
error_A = observed_error[:,0]
error_B = observed_error[:,1]
# print(error_A)
# print(error_B)

#Plotting errors with stdev of noise:

plt.plot(sigma,error_A,'ro--',label='Error in A',linewidth=1.0)
plt.plot(sigma,error_B,'go--',label='Error in B',linewidth=1.0)
plt.legend(loc='upper left')
plt.xlabel(r'Standard Deviation of the noise',size=20)
plt.ylabel(r'Error',size=20)
plt.title(r'Variation of errors with noise')
plt.show()

# plt.plot(sigma,error_A)
# plt.xlabel(r'Standard Deviation of the noise',size=20)
# plt.ylabel(r'Error in estimation of A',size=20)
# plt.title(r'Error in the least squares estimate of A')
# plt.show()

# plt.plot(sigma,error_B)
# plt.xlabel(r'Standard Deviation of the noise',size=20)
# plt.ylabel(r'Error in estimation of B',size=20)
# plt.title(r'Error in the least squares estimate of B')
# plt.show()

plt.loglog(sigma,error_A,'ro--',label='Error in A')
plt.loglog(sigma,error_B,'go--',label='Error in B')
plt.legend(loc='upper left')
plt.xlabel(r'Standard Deviation of the noise',size=15)
plt.ylabel(r'Error',size=15)
plt.title(r'Error in the least squares estimates (log log plot)')
plt.show()

# plt.loglog(sigma,error_A)
# plt.xlabel(r'Standard Deviation of the noise',size=20)
# plt.ylabel(r'Error in estimation of A',size=20)
# plt.title(r'Error in the least squares estimate of A (log log plot)')
# plt.show()

# plt.loglog(sigma,error_B)
# plt.xlabel(r'Standard Deviation of the noise',size=20)
# plt.ylabel(r'Error in estimation of B',size=20)
# plt.title(r'Error in the least squares estimate of B (log log plot)')
# plt.show()

# plt.semilogx(sigma,error_A)
# plt.xlabel(r'Standard Deviation of the noise',size=20)
# plt.ylabel(r'Error in estimation of A',size=20)
# plt.title(r'Error in the least squares estimate of A (semi log plot)')
# plt.show()

# plt.semilogx(sigma,error_B)
# plt.xlabel(r'Standard Deviation of the noise',size=20)
# plt.ylabel(r'Error in estimation of B',size=20)
# plt.title(r'Error in the least squares estimate of B (semi log plot)')
# plt.show()

# plt.semilogy(sigma,error_A)
# plt.xlabel(r'Standard Deviation of the noise',size=20)
# plt.ylabel(r'Error in estimation of A',size=20)
# plt.title(r'Error in the least squares estimate of A (semi log plot)')
# plt.show()

# plt.semilogy(sigma,error_B)
# plt.xlabel(r'Standard Deviation of the noise',size=20)
# plt.ylabel(r'Error in estimation of B',size=20)
# plt.title(r'Error in the least squares estimate of B (semi log plot)')
# plt.show()