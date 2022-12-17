import numpy as np
import matplotlib.pyplot as plt
import math
from templates import *

def dyn1(x, u, dist):
	nu = dist.sample()
	# print(nu)
	x_dot = np.zeros((3,1))
	# print(x.shape,u.shape, nu.shape)
	x_dot[0] = (u[0] + nu[0])*np.cos(x[2])
	x_dot[1] = (u[0] + nu[0])*np.sin(x[2])
	x_dot[2] = u[1] + nu[1]
	return x_dot

def dyn2(x,u,dist):
	x_dot = np.zeros((2,1))
	x_dot[0] = x[1]
	x_dot[1] = u - 0.1*x[1] 
	return x_dot


if __name__=='__main__':
	# ----------------------------
	# iter = 100
	# n = 2
	# d = distribution(type="GAUSSIAN", n = n, mu=[0,0], cov=[[1,0],[0, 2]])
	# x = np.zeros((n,iter))
	# for i in range(iter):
	# 	x[:,i] = d.sample()
	# # print(x)
	# plt.scatter(x[0,:], x[1,:])
	# plt.show()


	# ----
	# n = 3; m =2 ; x0=[0, 0, 0] ; d = distribution(type ="GAUSSIAN", n =2, mu=[0,0], cov=[[0,0],[0,0]]); dt =  1e-1;
	# mod = model(n=n, m=2, x0=x0, dist=d, dt=dt, dynamics = dyn1);
	# for i in range(iter):
		# u = np.random.random((m,1))*10
		# print(u)
		# print(u.shape)
		# mod.aplycntl(u)
		# plt.scatter(mod.x[0,:],mod.x[1,:])
		# plt.pause(0.5)
	# print (mod.true_states.shape)
	# plt.subplot(3,1,1)
	# plt.plot(np.arange((mod.true_states.shape[1]))*dt, mod.true_states[0,:])
	# plt.subplot(3,1,2)
	# plt.plot(np.arange((mod.true_states.shape[1]))*dt, mod.true_states[1,:])
	# plt.subplot(3,1,3)
	# plt.plot(np.arange((mod.true_states.shape[1]))*dt, mod.true_states[2,:])
	# plt.scatter(mod.true_states[0,:],mod.true_states[1,:])
	# plt.show()
	# ---

	# ----
	# n = 2; m =1; x0=[0, 0] ; d = distribution(type ="GAUSSIAN", n =2, mu=[0,0], cov=[[0,0],[0,0]]); dt =  1e-2
	# mod = model(n=n, m=2, x0=x0, dist=d, dt=dt, dynamics = dyn2);
	# for i in range(iter):
	# 	u = np.random.random((m,1))
	# 	# print(u)
	# 	# print(u.shape)
	# 	mod.aplycntl(u)
	# 	# plt.scatter(mod.x[0,:],mod.x[1,:])
	# 	# plt.pause(0.5)
	# # print (mod.true_states.shape)
	# plt.subplot(2,1,1)
	# plt.plot()

	# plt.scatter(mod.true_states[0,:],mod.true_states[1,:])
	# plt.show()	

	#======================================================================
	n_dim = 3; m_dim = 2; x0 = [0,0,0];dt = 1e-2; mu=[0,0]; cov = [[1,0],[0,1]]
	m = model
