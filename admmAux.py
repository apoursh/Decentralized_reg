import numpy as np
from scipy.optimize import fmin_l_bfgs_b as bfgs
from numpy.linalg import cholesky


def l2_log(x, C, z, u, rho):
    v = x - z + u
    f = np.sum(np.log(1.0 + np.exp(np.dot(x,np.transpose(C))))) + 1/2.0 * rho * np.dot(np.transpose(v),v)
    eCx = np.exp(np.dot(x, np.transpose(C)))
    g = np.dot(np.transpose(C), (eCx/(1+eCx)))+ rho*v
    return f,g

def bfgs_update(C, u, z, rho, x0):
    args = (C, z, u, rho)
    return bfgs(l2_log, x0, args=args, bounds=None, callback=callback)

def shrinkage(a, kappa):
    z = np.maximum(0, a-kappa) - np.maximum(0, -a-kappa)
    return z

def l1_OLS(A, b, lam, x, z):
    return 1.0/2 * np.sum((A.dot(x) - b)**2) + lam * np.norm(z,1)

def lasso_admm_cholesky(A,rho):
    if A.shape[0] > A.shape[1]: # tall matrix
        L = cholesky(A.T.dot(A) + rho * np.eye(A.shape[1],dtype=float))
    else: # fat matrix (unlikely but possible)
        L = cholesky(np.eye(A.shape[0],dtype=float) + 1.0/rho * A.dot(A.T))
    return L



def callback(x):
    #print x
    pass


if __name__ == '__main__':
    print "running"
