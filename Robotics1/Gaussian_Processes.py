# Required python packages
# pip install numpy
# pip install matplotlib
# pip install scikit-learn



from functools import partial
import numpy as np
from numpy import exp, sqrt, sinc
from numpy import array, diag, eye
from numpy.linalg import inv, solve
rng = np.random.default_rng(4)

from matplotlib import pyplot as plt
#squared-exponential radial basis function (SE-RBF)
def kernel(x1, x2, sigma=1, lengthscale=1):
    return sigma**2 * exp( -1/(2*lengthscale**2) * (x1 - x2)**2 )  # TODO complete the kernel definition to visualise the kernel.

# RBF Kernel (Squared Exponential Kernel):
def rbf_kernel(x1, x2, lengthscale=1):
    return np.exp(-0.5 * ((x1 - x2) / lengthscale)**2)

#-----------------------------


data = np.genfromtxt('Safe.csv', delimiter=',', names=True, dtype=None, encoding=None)

# Access the 'xd' and 'yd' columns as NumPy arrays.
xd = data['time']
yd = data['yd']* 0.001
nd = len(yd) 

std_n = 0.1

max_time = xd.max()
xq = np.linspace(0, +max_time, len(xd))
# xq = x  # query locations

#--------------------------------

## evaluating the kernel
Kdd = kernel(xd[:,None], xd)  
Kqd = kernel(xq[:,None], xd)  
Kdq = Kqd.T  
Kqq = kernel(xq[:,None], xq)  

## the GP equations
Snn = std_n**2 * eye(nd)  # TODO
mean_q =       Kqd @ inv(Kdd + Snn) @ yd  
cov_qq = Kqq - Kqd @ inv(Kdd + Snn) @ Kdq  
var_q  = diag(cov_qq)  # TODO
std_q  = sqrt(var_q)  # TODO
yq = rng.multivariate_normal(mean_q, cov_qq, 3).T  

## plot
# plt.plot(xq, true_function(xq), label="true function")
# print(xq.shape)
# print(xq)
# print(mean_q.shape)
# print(mean_q)
# print(std_q.shape)
# print(std_q)
yd = yd * 1000
mean_q = mean_q * 1000
std_q = std_q * 1000
yq = yq * 1000

plt.plot(xd, yd, 'ro', label="observed data")
plt.plot(xq, mean_q, 'k', label="mean")
plt.fill_between(xq, mean_q + std_q, mean_q - std_q, color='k', alpha=0.3, label="std dev")
plt.plot(xq, yq, ':', label="sample")
plt.legend()
plt.xlabel("Time")  # Label for the x-axis
plt.ylabel("PPM")  # Label for the y-axis
plt.show()



max_index = np.argmax(mean_q)
max_time = xq[max_index]
max_value = mean_q[max_index]
plt.plot(xd, yd, 'ro', label="observed data")
plt.plot(xq, mean_q, 'k', label="mean")
plt.fill_between(xq, mean_q + std_q, mean_q - std_q, color='k', alpha=0.3, label="std dev")
plt.plot(xq, yq, ':', label="sample")
plt.scatter(max_time, max_value, c='g', marker='o', s=100, label="Max Value")  # Highlight the max value
plt.xlim(max_time - 2, max_time + 2)  # Adjust the x-axis limits for zooming
plt.ylim(max_value - 2 * std_q[max_index], max_value + 2 * std_q[max_index])  # Adjust the y-axis limits for zooming
plt.legend()
plt.xlabel("Time")  # Label for the x-axis
plt.ylabel("PPM*1000 (Parts per thousand)")  # Label for the y-axis
plt.show()


# # Maybe can use this to detect difference in standards vs dangerous levels
# from sklearn.metrics import mean_squared_error
# y_true = true_function(xq)
# MSE = mean_squared_error(y_true , mean_q)
# print(f'MSE is {MSE}')





