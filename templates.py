import math
import numpy as np

import matplotlib.pyplot as plt

class distribution():
	## add	define constants this class can handle
	def __init__(self,**kwargs):
		self.TYPE = kwargs['type']
		self.SIZE = kwargs['n']
		# self.PARAMETERS 


		## add some checks to ensure parameters are consistent with each other
		## like n = #mu
		## check whether distribution can be handled ( POISSION, ...)

		if 'mu' in kwargs.keys():
			self.mu = kwargs['mu']
		else:
			self.mu = np.zeros(self.SIZE)

		if 'cov' in kwargs.keys():
			self.cov = kwargs['cov']
		else:
			self.cov = np.random.random((self.SIZE, self.SIZE))

		if self.TYPE == "GAUSSIAN":
			self.sample = self.gStep
			## add cases for other distributions

	## define sample function for other distribution types
	def gStep(self):
		nu = np.random.multivariate_normal(self.mu, self.cov)
		return nu


class model():
	def __init__(self, **kwargs):
		## some check on kwargs

		self.x0 = np.array(kwargs['x0']).reshape((self.n, 1))
		self.x = self.x0	
		self.dist = kwargs['dist']
		self.n = kwargs['n']		
		self.m = kwargs['m']
		self.dt = kwargs['dt'] #sample time

		if 'dynamics' in kwargs.keys():
			self.derivate = kwargs['derivate']
		else :
			self.derivate = self.dynamics

		self.integration = self.euler

		self.true_states = np.array(self.x0)
				
	def step(self, u):
		x_dot = self.derivate(u, self.dist)
		x = self.euler(x, x_dot)
		self.x  = x
		self.true_states = np.append(self.true_states, self.x)		

	def dynamics(self, u, dist):

	def euler(self, x, x_dot) 
		return x + x_dot*self.dt









if __name__=='__main__':
	iter = 100
	n = 2
	d = distribution(type="GAUSSIAN", n = n, mu=[0,0], cov=[[1,0],[0, 2]])
	x = np.zeros((n,iter))
	for i in range(iter):
		x[:,i] = d.sample()
	# print(x)
	plt.scatter(x[0,:], x[1,:])
	plt.show()


	