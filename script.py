import numpy as np
import matplotlib.pyplot as plt
import math
from templates import *

def dyn1(x, u, dist):
	nu = dist.sample()
	# print(nu)
	x_dot = np.zeros((3,1))
	# print(x.shape,u.shape, nu.shape)
	# print(np.cos(x[2]).shape,(u[0] + nu[0]).shape,((u[0] + nu[0])*np.cos(x[2])).shape)
	x_dot[0,:] = (u[0] + nu[0])*np.cos(x[2])
	x_dot[1,:] = (u[0] + nu[0])*np.sin(x[2])
	x_dot[2,:] = u[1] + nu[1]
	return x_dot

def dyn1lin(x, u):
	F = np.zeros((3,3))
	F[0,0] = 1
	F[0,2] = -u[0]*np.sin(x[2])
	F[1,1] = 1
	F[1,2] = u[0]*np.cos(x[2])
	F[2,2] = 1
	return F

if __name__=='__main__':
	n_dim = 3; m_dim = 2; nu_dim = 2; x0 = [0,0,0]; xh=x0; dt = 1e-2; mu=[0,0]; cov = [[1,0],[0,1]]; ph= [[1,0,0],[0,1,0],[0,0,1]]; Q = ph
	m = model(n = n_dim, m = m_dim, x0=x0, dt= dt,
				dist = distribution(type='GAUSSIAN', n=n_dim, mu=mu, cov=cov ),
				dynamics = dyn1)
	e = estimator(xh=x0, ph=ph, dyn1=dyn1, dt= dt, Q=Q,
					dyn1lin=dyn1lin)	

	r = record(dt = dt)
	r.update(m,e)

	# print(r.Xs.shape, r.Xh.shape, r.Ph.shape)	

	T0 = 0; Tf  = 10;
	for n in range(int((Tf - T0)/dt)):
		t = n*dt
		# print(n)
		# ---
		u = np.random.random((m_dim,1))
		
		# ---	
		m.aplycntl(u)
		e.estimate(u)

		r.update(m,e)

	# print(r.Xs.shape, r.Xh.shape, r.Ph.shape)	
	r.plotErrors()


