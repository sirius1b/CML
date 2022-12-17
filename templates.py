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
		return np.expand_dims(nu,axis=1)


class model():
	def __init__(self, **kwargs):
		## some check on kwargs
		self.n = kwargs['n']		
		self.x = np.array(kwargs['x0']).reshape((self.n, 1))
		self.dist = kwargs['dist']
		self.m = kwargs['m']
		self.dt = kwargs['dt'] #sample time

		if 'dynamics' in kwargs.keys():
			self.derivate = kwargs['dynamics']
		else :
			raise Exception("dynamics not defined")

		self.integration = self.__euler
		self.true_states = np.array(self.x)
				
	def aplycntl(self, u):
		x_dot = self.derivate(self.x, u, self.dist)
		self.x = self.integration(self.x, x_dot)
		self.true_states = np.append(self.true_states, self.x, axis = 1)		

	def __euler(self, x, x_dot): 
		return x + x_dot*self.dt


class estimator:
	def __init__(self,**kwargs):
		self.x_hat = np.expand_dims(np.array(kwargs['xh']),axis=1)
		self.p_hat = np.array(kwargs['ph'])
		self.dyn1 = kwargs['dyn1']
		self.dyn1lin = kwargs['dyn1lin']
		self.dt = kwargs['dt']
		self.Q =  np.array(kwargs['Q'])
		self.dist = distribution(type="GAUSSIAN", n = self.x_hat.shape[0], mu=[0,0], cov=[[0,0],[0, 0]])

	def estimate(self, u):
		self.__predict(u)
		

	def __predict(self, u):
		x_dot = self.dyn1(self.x_hat, u, self.dist)
		F = self.dyn1lin(self.x_hat, u)
		self.p_hat = F*self.p_hat*F.T + self.Q
		self.x_hat = self.__euler(self.x_hat, x_dot)

	def __euler(self, x, x_dot):
		return x + x_dot*self.dt;

class record():
	def __init__(self, **kwargs):
		self.Xs = None
		self.Xh	= None 
		self.Ph = None 
		self.dt = kwargs['dt']

	def update(self, m, e):
		# if self.Xs is not None:
		# 	print(self.Xs.shape, self.Xh.shape, self.Ph.shape,'-1')
		self.__addXs(m)	
		self.__addXh(e)
		self.__addPh(e)
		# print(self.Xs.shape, self.Xh.shape, self.Ph.shape,'-2')

	def plotErrors(self):
		n = self.Xs.shape[0]
		time = np.arange(self.Xs.shape[1])*self.dt
		for i in range(n):
			plt.subplot(n+1,1,i+1)
			plt.plot(time, (self.Xs[i,:]-self.Xh[i,:]))

		plt.subplot(n+1, 1, n+1)
		plt.plot(time, self.Ph)
		plt.show()


	def __addXs(self, m):
		if self.Xs is not None:
			self.Xs = np.append(self.Xs, m.x, axis =1)
		else:
			self.Xs = np.array(m.x)

	def __addXh(self, e):
		if self.Xh is not None:
			self.Xh = np.append(self.Xh, e.x_hat, axis=1)
		else:
			self.Xh = np.array(e.x_hat)

	def __addPh(self, e):
		if self.Ph is not None:
			# self.Ph = np.append(self.Ph, e.p_hat, axis=1)
			self.Ph = np.append(self.Ph, np.trace(e.p_hat).reshape(1), axis= 0)
		else:
			# self.Ph = np.array(e.p_hat)
			self.Ph = np.trace(e.p_hat).reshape(1)

	










	