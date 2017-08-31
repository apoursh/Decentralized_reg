import numpy as np
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, zero_one_loss
from numpy.linalg import norm, solve
from admmAux import *
import sys
#import pdb
_EPS=1e-8
_EPS_BIG=1e-6


class Error(Exception):
    pass

class InputError(Error):
    """Error in input."""
    def __init__(self):
        pass


def all_regression(datahub, logistic=True, verbose=False):
    datahub.count_data()
    N = datahub.k
    p = datahub.beta.shape[0]
    data = np.empty((N,p - 1))
    y    = np.empty((N,))
    i = 0
    for repo in datahub.repos:
        nInRepo = repo.X.shape[0] + i
        data[i:nInRepo,:] = repo.X[:,1:]
        y[i:nInRepo,] = repo.Y if logistic else repo.Ylinear
        i = nInRepo
    if verbose:
        print("total size of data is: ", data.shape)
    try:
        mdl = LogisticRegression(C=1e9, tol=0.00001,  solver='newton-cg') if logistic else LinearRegression()
        mdl.fit(data, y)
        beta = np.append(mdl.intercept_, mdl.coef_)
    except ValueError:
        beta = np.array([np.nan for i in range(p)])
    if logistic:
        yPred = sigmoid(data.dot(beta[1:]) + beta[0])
        try:
            rocauc = roc_auc_score(y, yPred)
            cost = log_loss(y, yPred)
            zeroOne = zero_one_loss(y, (yPred + 0.5).astype(int))
        except ValueError:
            rocauc = np.nan
            cost = np.nan
            zeroOne = np.nan
        return beta, cost, rocauc, zeroOne

    else: #for linear regression
        yPred = data.dot(beta[1:]) + beta[0]
        return beta, mean_squared_error(y,yPred)
    #yPred = mdl.predict(data)

def cross_entropy(y_true, y_pred):
    if 1 - y_pred < _EPS:
        y_pred = 1-_EPS
    elif y_pred < _EPS:
        y_pred = _EPS
    return(np.log(y_true * y_pred + (1 - y_true) * (1.0 - y_pred)))


class DataRepo(object):
    # This represents a hospital or whatnot

    def __init__(self, size, draws, dim, beta, hub, f=0.5):
        # contained is deciding on it's membership in container
        #self.beta = beta[1:]
        #self.intercept = beta[0]
        self.beta = beta
        self.hub = hub
        self.hub.add_repo(self)
        self.chol = None # cholesky factorization
        if size > 0:
            if isinstance(draws, basestring) and (draws.lower() == 'u' or draws.lower() == 'uniform'):
                #print "drawing from uniform distribution"
                self.X = (np.random.random((size, dim)) - 0.5) * 5
            elif isinstance(draws, basestring)and (draws.lower() == 'n' or draws.lower() == 'normal'):
                alpha = np.random.random(dim)*3
                self.X = np.random.normal(loc = 0.0, scale = 10**alpha, size=(size, dim))
            elif isinstance(draws, basestring) and (draws.lower() == 'm' or draws.lower()=='mixed'):
                ndisc = int(f * dim)
                p = np.random.random(ndisc)*(0.48) + .01 # draw frequencies for roughly half the covariates
                q = 1-p
                Xdisc = np.random.choice([0.0,1.0,2.0], (size, ndisc), [p**2,2*p*q,q**2])
                alpha = np.random.random(dim-ndisc)
                Xcont = np.random.normal(loc=0,scale=10**alpha,size=(size, dim-ndisc))
                self.X = np.hstack((Xdisc, Xcont))

            #customized data
            elif isinstance(draws, list) and all(isinstance(draw, float) for draw in draws) and np.array(draws).shape[1] == len(beta):
                if np.array(draws).shape[0] == size:
                    self.X = draws
                else:
                    print ('Provided array of shape {} does not have the right shape.'.format(str(np.array(draws).shape)))
                    raise InputError
            else:
                print ('{} is not a known draw model'.format(draws))
                raise InputError
            #response variable
            self.X = np.hstack((np.ones((self.X.shape[0],1)), self.X))
            #self.Y = np.random.binomial(1,sigmoid(np.dot(self.X, self.beta) + self.intercept + np.random.normal(loc=0.0, scale=0.1*np.sqrt(dim+1), size=
            #    self.X.shape[0])))
            self.Y = np.random.binomial(1, sigmoid(np.dot(self.X,self.beta) + np.random.normal(loc=0.0, scale=0.1*np.sqrt(dim+1), size=
                self.X.shape[0])))
            self.symY = np.sign(self.Y - 0.5)
            self.symY.shape = (self.X.shape[0],1)
            self.Ylinear = self.X.dot(self.beta) + np.random.normal(loc=0.0, scale=0.1*(dim+1), size=self.X.shape[0])
        else:
            self.X = np.empty((0,dim))
            self.Y = np.empty((0,1))

#    def old_SGD(self, curBeta, stepSize, k, average=False):
#        pred = sigmoid(np.dot(self.X[k,:], curBeta[1:]) + curBeta[0])
#        g = self.X[k,:]*(pred - self.Y[k])
#        g0 = pred - float(self.Y[k])
#        b0, bk = curBeta[0] - stepSize * g0, curBeta[1:] - stepSize * g
#        curBeta[0] = b0
#        curBeta[1:] = bk[:]
#        #Add code for averaging
#        return curBeta

    def num_inds(self):
        return self.X.shape[0]

    def localLogistic(self):
        model = LogisticRegression(C = 1e9, tol=0.00001, solver='newton-cg')
        try:
            model = model.fit(self.X[:,1:], self.Y)
            beta = np.append(model.intercept_, model.coef_)
        except ValueError:
            beta = np.array([np.nan for i in self.beta])
        return beta
    def localLinearReg(self):
        model = LinearRegression()
        model = model.fit(self.X[:,1:], self.Ylinear)
        beta  = np.append(model.intercept_, model.coef_)
        return beta

    def admmUpdate(self, u, z, rho, x0, logistic=True):
        if logistic:
            b = self.symY
            K = - self.X #np.hstack((-np.ones((self.X.shape[0],1)), -self.X))
            K = K * b
            x, v, d = bfgs_update(K, u, z, rho, x0)
        else:
            tall = self.X.shape[0] > (self.X.shape[1] - 1)
            K = np.hstack((np.ones((self.X.shape[0],1)), self.X)) # add 1 to the covariates for intercept
            if self.chol == None:
                self.chol = lasso_admm_cholesky(K,rho)
            temp = K.T.dot(self.Ylinear) + rho * (z-u)
            if tall:
                x = solve(self.chol.T, solve(self.chol, temp))
            else:
                x = temp/rho - (K.T * solve(self.chol.T, solve(self.chol, K.dot(temp)))) / rho ** 2
        return x


    def grad(self, curBeta, k, logistic=True):
        xbeta = np.dot(self.X[k,1:], curBeta[1:]) + curBeta[0]
        pred = sigmoid(xbeta) if logistic else xbeta
        g = np.zeros(len(curBeta))
        res = pred
        res -= self.Y[k] if logistic else self.Ylinear[k]
        g[0] = np.mean(res)
        if isinstance(k, int):
            g[1:] = self.X[k,1:] * (res)
        else :
            g[1:] = np.dot(res, self.X[k,1:])/float(len(k))#np.mean(self.X[k,:] * (pred-self.Y[k]), axis=0)
        return g
    def symgrad(self, curBeta, k, logistic=True):
        #y = np.sign(self.Y[k]-0.5)
        #X = np.hstack((np.ones((len(k),1)), self.X[k,:]))
        X = self.X[k,:]
        curBeta.shape = (curBeta.shape[0],1)
        if logistic:
            sig = np.exp(-self.symY[k] * np.dot(X, curBeta))
            g = -1.0/len(k) * ((sig/(1 + sig) * self.symY[k] ).T.dot(X)).T
        else:
            g = 1.0/len(k) * X.T.dot(X.dot(curBeta) - self.symY[k])
        return g


    def logisticHessian(self, curBeta, v, k,lam=0):
        #X = np.hstack((np.ones((len(k),1)),self.X[k,:]))
        #y = np.sign(self.Y[k]-0.5)
        X = self.X[k,:]
        curBeta.shape = (curBeta.shape[0],1)
        sig = np.exp(-self.symY[k] * np.dot(X, curBeta))
        sig /= (1 + sig)
        sig.shape = (sig.shape[0],)
        ts = np.dot(np.dot(np.diag(sig - sig**2), X),v)
        H = 1.0 / len(k) * np.dot(np.transpose(X), ts) + lam * v
        return H


    def LinearRegHessian(self, curBeta, v, k):
        #X = np.hstack((np.ones((len(k),1)),self.X[k,:]))
        X = self.X[k,:]
        y = self.Ylinear[k]
        H = 1.0 / len(k) * X.T.dot(X).dot(v)
        return H

    def scales(self):
        return np.mean(self.X,axis=0), np.std(self.X, axis=0)

    def scaleData(self,mu,sig):
        self.X[:,1:] -= mu[1:]
        self.X[:,1:] /= sig[1:]

    def add_data(self, X, Y):
        """This will be horribly inefficient but will do for now
        This function updates the central hub's count (k)
        """
        self.X = np.vstack((self.X, X))
        self.Y = np.append(self.Y, Y)
        self.symY = np.append(self.symY, np.sign(Y - 0.5))
        if self.hub.update_beta(self):
            self.hub.SGD(self, len(self.X) - 1, 1.0 / (1.0 + self.hub.k) ** 0.52)
            self.hub.k += 1
        return self.hub.beta


class DataCenter(object):

    def __init__(self, dim):
        self.repos = []
        self.dim = dim + 1 # intercept
        self.beta = np.array([0.0 for i in range(int(dim) + 1)])
        self.count_data()
        self.mu = None
        self.sig = None
    def count_data(self):
        self.k = 0
        for repo in self.repos:
            self.k += repo.num_inds()
    def update_beta(self, name):
        return True
    def scale(self):
        self.mu = 0.0
        self.sig = 0.0
        for repo in self.repos:  # find the right scaling parameters
            repo_mean, repo_sigma = repo.scales()
            frac = repo.num_inds() / (float(self.k - 1)) # unbiased estimator
            self.mu += frac * repo_mean
            self.sig += frac * repo_sigma ** 2
        #scale the data
        for repo in self.repos:
            repo.scaleData(self.mu, np.sqrt(self.sig))

    def add_repo(self, name):
        self.repos.append(name) ##more needs to be done here update beta....

    def SGD_regression(self, k=0, forward=True):
        """Deprecated. Use batch_SGD instead"""
        sys.error("SGD_regression needs to be improved before it's really usable. Use batch_SGD")
        if forward :
            for repo in self.repos:
                n = len(repo.X)
                for ind in xrange(n):
                    self.SGD(repo, ind, 1.0/(2.0 + k)**.51)
                    k += 1
        else:
            for repo in reversed(self.repos):
                n = len(repo.X)
                for ind in reversed(xrange(n)):
                    self.SGD(repo, ind, 1.0/(2.0 + k)**.51)
                    k += 1
    def SGD(self, repo, element, stepSize):
        g = repo.grad(self.beta, element)
        self.beta -= stepSize * g
        return (self.beta)

    def batch_SGD(self, frac_per_batch, eps, max_iters=100, step=None, stepAlg=0,
            scheduled=True, logistic=True, verbose=False):
        beta = np.zeros(self.beta.shape[0])
        p = 0.9
        betaOld = beta + 10 * _EPS_BIG
        if verbose:
            BETA = np.zeros((self.beta.shape[0], max_iters / 10 + 1))
        # determine stepsize
        G = lambda old, gr : 0
        ssTheta = lambda sst, gr, gg : 0
        if step != None:
            if not isinstance(step, float):
                raise ValueError("specified stepsize is not a float")
        if stepAlg == 1:
            step = 0.01 if step == None else step
            sz = lambda x, gg, sst : step/np.sqrt(gg + _EPS_BIG)
            G = lambda old, gr: old + gr ** 2
        elif stepAlg == 2: # rms prop
            step = 0.001 if step == None else step
            sz = lambda x,gg,sst : step/np.sqrt(gg + _EPS_BIG)
            G = lambda old, gr: p * old + (1 - p) * gr ** 2
        elif  stepAlg == 3: # adadelta
            step = 0.01 if step == None else step
            ssTheta = lambda sst, gr, gg : p*sst + (1 - p) * step ** 2 * gr ** 2 / (gg + _EPS_BIG)
            sz = lambda x,gg,sst : np.sqrt((sst + _EPS_BIG) / (gg + _EPS_BIG))
            G = lambda old, gr: p * old + (1 - p) * gr ** 2

        else:
            if not scheduled:
                sz = lambda x, y, sst: step
            else:
                sz = lambda x, y, sst: step / (1.0 + x) ** 0.51 #if x < 500 else step/(1.0 + x-500)**0.51

        i, j, grad = 1, 0, 0
        gg, sstheta  = 0, 0 #np.sqrt(_eps)
        g_of_batch, n_of_batch = np.zeros((len(self.repos), self.dim)), np.zeros((len(self.repos),1))
        # I'll check condition before every call
        while (norm(beta - betaOld) > _EPS and j < max_iters):
            #ranger = lambda i, bs : range((i - 1) * bs, i * bs)
            #k = range((i - 1) * bs, i * bs)
            for batch_i, repo in enumerate(self.repos):
       #         if repo.X.shape[0] > i * bs:
       #             g.append(repo.grad(beta,k))
       #     if len(g) > len(self.repos)/2.0: #most of them don't have more data
       #         betaOld = beta.copy()
       #         gr = np.mean(g,axis=0)
       #         gg = G(gg,gr)
       #         beta -= sz(j,gg,sstheta) * gr
       #         sstheta = ssTheta(sstheta, gr, gg)
       #         i += 1
       #         j += 1
       #     else:
                n_i_f = repo.num_inds() * frac_per_batch
                n_i   = int(n_i_f)
                if np.random.uniform < (n_i_f - n_i):
                    n_i += 1
                subset = np.random.choice(range(repo.X.shape[0]), n_i)
                g_of_batch[batch_i,:] = repo.grad(beta,subset,logistic=logistic).ravel()
                n_of_batch[batch_i,0] = n_i
            n_tot = np.sum(n_of_batch)
            grad = np.mean(g_of_batch * n_of_batch/n_tot, axis=0)
            betaOld = beta.copy()
            #gr = np.mean(g, axis=0)
            gg = G(gg,grad)
            beta -= sz(j,gg,sstheta) * grad
            sstheta = ssTheta(sstheta, grad, gg)
            j += 1
            if verbose and j % 10 == 0:
                BETA[:, j / 10] = beta
        if verbose:
            BETA[:,max_iters / 10] = beta
            return BETA

        return beta, j

    def evaluate(self,beta, logistic=True):
        tot = self.k
        cost = 0
        rocauc = 0
        zeroOne = 0
        pred = np.zeros((tot,1))
        actual = np.zeros((tot,1))
        i,j = 0,0
        for repo in self.repos:
            xbeta = repo.X[:,1:].dot(beta[1:]) + beta[0]
            j = xbeta.shape[0] + i
            pred[i:j,0] = sigmoid(xbeta) if logistic else xbeta
            actual[i:j,0] = repo.Y if logistic else repo.Ylinear
            i = j
        try:
                #cost += log_loss(repo.Y,pred,normalize=False) if logistic else mean_squared_error(repo.Ylinear, pred) * repo.X.shape[0]
            if logistic:
                try:
                    rocauc = roc_auc_score(actual, pred)
                    cost = log_loss(actual, pred)
                    zeroOne = zero_one_loss(actual, (pred + 0.5).astype(int))
                except ValueError:
                    rocauc = np.nan
                    cost = np.nan
                    zeroOne = np.nan
            else:
                cost = mean_squared_error(actual, pred)
        except ValueError:
            print pred
            return np.nan
        #return cost/float(self.k), rocauc/float(self.k), zeroOne/float(self.k) if logistic else cost/float(self.k)
        return cost, rocauc, zeroOne if logistic else cost

    def SQN(self, m, l, eta, frac_for_grad, frac_for_hess,
            max_iters=100, logistic=True, verbose=False):
        beta, betaBar, betaOld = np.zeros(len(self.beta)),0,0
        if verbose: # Somewhat useless
            BETA = np.zeros((beta.shape[0], max_iters/10+1))
        Hk0 = np.zeros((self.dim,1))
        t = -1
        #s,y= np.zeros((m,1)), np.zeros((m,1))
        st,yt = [], []
        g_of_batch, n_of_batch = np.zeros((len(self.repos), self.dim)), np.zeros((len(self.repos),1))
        i, k= 0, 0
        while i < max_iters:
            for batch_i, repo in enumerate(self.repos):
                n_i_f = repo.num_inds() * frac_for_grad
                n_i   = int(n_i_f)
                if np.random.uniform < (n_i_f - n_i):
                    n_i += 1
                subset = np.random.choice(range(repo.X.shape[0]), n_i)
                g_of_batch[batch_i,:] = repo.symgrad(beta,subset,logistic=logistic).ravel()
                n_of_batch[batch_i,0] = n_i
            n_tot = np.sum(n_of_batch)
            grad = np.mean(g_of_batch * n_of_batch/n_tot, axis=0)
            grad.shape = (grad.shape[0],1)
            # run m steps of sgd
            betaBar += beta
            betaOld = beta.copy()
            if k < 2 * l:
                beta -= eta * grad
                Hk0 = 0.9 * Hk0 + 0.1 * grad**2
            else:
                update, Hk0 = self.sqn2loop(grad, m, st, yt, Hk0)
                beta -= eta * update
            if k % l == 0:
                if verbose and i % 10 == 0:
                    BETA[:,i / 10] = beta
                i += 1
                t += 1
                betan = betaBar / float(l)
                if t > 0:
                    H = 0
                    s = betan - betaOld
                    for repo in self.repos:
                        nH_i_f = repo.num_inds() * frac_for_hess
                        nH_i   = int(nH_i_f)
                        if np.random.uniform < (nH_i_f - nH_i):
                            nH_i += 1
                        subset = np.random.choice(range(repo.X.shape[0]), nH_i, False)
                        if logistic:
                            H = H + repo.logisticHessian(betan, s, subset)
                        else:
                            H = H + repo.LinearRegHessian(betan,s, subset)
                    H /= float(len(self.repos))
                    #if np.dot(s,H)/np.dot(H,H) > _eps:
                    st.append(s)
                    yt.append(H)
                    #else:
                    #    print "useless"
                betaOld = betan
                betaBar = 0
            if verbose and i % 10 == 0:
                BETA[:,i / 10] = beta

            k += 1
            i += 1
        if verbose:
            BETA[:,max_iters / 10]=beta
            return BETA

        return beta



    def sqn2loop(self,gr,m,st,yt,Hk0): #https://github.com/keskarnitish/minSQN/blob/master/%2Bhelpers/QuasiNewton.m
        #update adagrad estimate
        grads = 0.9 * Hk0 + 0.1 * gr ** 2
        Hk0 = 1.0 / np.sqrt(grads + np.sqrt(_EPS))
        #Hk0 = np.ones(len(gr))
        m = min(len(st),m)
        q = gr
        rho = np.zeros(m)
        alpha = np.zeros(m)
        for i in range(m - 1, -1, -1): #count down
            rho[i] = 1.0/np.dot(yt[i].T,st[i])
            #rho[i] = 1.0/np.dot(yt[i],np.transpose(st[i]))
            alpha[i] = rho[i] * np.transpose(st[i]).dot(q)
            q -= alpha[i] * yt[i]
        R = Hk0 * q

        for j in xrange(m):
            beta = rho[j] * np.transpose(yt[j]).dot(R)
            R += st[j] * (alpha[j] - beta)
        return R, grads



    def avgLogistic(self, unaveraged = False):
        if unaveraged:
            repoBetas = np.zeros((len(self.repos), self.dim))
            for i, repo in enumerate(self.repos):
                repoBetas[i,:] = repo.localLogistic()
            else:
                return np.nanmean(repoBetas, axis=0)
        else:
            average = np.zeros(self.dim)
            for repo in self.repos:
                beta = repo.localLogistic()
                if not np.any(np.isnan(beta[0])):  # nans appear b/c of pure classes at repos
                    average += repo.X.shape[0]/float(self.k) * beta
            return average

    def avgLinearReg(self):
        avg = np.zeros(self.dim + 1)
        for repo in self.repos:
            beta = repo.localLinearReg()
            avg += repo.num_inds() / float(self.k) * beta
        return avg

    def ADMM(self, rho, max_iters= 10, alpha=1.0, mu=1e-9, logistic=True, verbose=False):# cite the bfgs papers
        """insert comments"""
        N = len(self.repos)
        x = np.zeros((self.dim, N))
        z = np.zeros((self.dim, N))
        u = np.zeros((self.dim, N))
        if verbose:
            Z = np.zeros((self.dim, max_iters/10 + 1))

        for k in range(max_iters):
            # update x
            for i in range(N):
                x[:,i] = self.repos[i].admmUpdate(u[:,i], z[:,i], rho, x[:,i], logistic)
            # Update z
            zold = z
            x_hat = alpha * x + (1.0 - alpha) * zold
            ztilde = np.mean(x_hat + u, axis = 1)
            ztilde[2:] = shrinkage(ztilde[2:], self.k * mu/float(rho * N))
            ztilde.shape = (ztilde.shape[0],1)
            z = np.dot(ztilde , np.ones((1,N)))

            # Update u
            u = u + (x_hat - z)
            if verbose and k % 10 == 0:
                Z[:,k / 10] = z[:,0]

        if verbose:
            Z[:,k / 10 + 1] = z[:,0]
            return Z
        return z[:,0]


    def adaGrad(self, forward=True, SSG=0):
        """Depricated! Use batch_SGD"""
        sys.error("Don't use this!")
        if forward:
            for repo in self.repos:
                grad = repo.grad
                n = len(repo.X)
                for ind in xrange(n):
                    g = grad(self.beta, ind)
                    SSG += g**2
                    nu = 0.01/np.sqrt(SSG + _EPS)
                    self.beta = self.beta - nu * g
        return(SSG)

    def adaDelta(self, SSG=0, SSDbeta=0, forward=True):
        """Depricated! Use batch_SGD"""
        sys.error("Don't use this!")
        if forward:
            for repo in self.repos:
                grad = repo.grad
                n = len(repo.X)
                for ind in xrange(n):
                    g = grad(self.beta, ind)
                    SSG = 0.999 * SSG + 0.001 * g**2
                    deltaBeta = - np.sqrt(SSDbeta + _EPS)/np.sqrt(SSG + _EPS) * g
                    SSDbeta = 0.999 * SSDbeta + 0.001 * deltaBeta**2
                    self.beta = self.beta + deltaBeta

        return SSG, SSDbeta

    def predict(self, predX):
        if predX.shape[1] == beta.shape[0] - 1:
            return sigmoid(np.dot(predX, self.beta[1:]) + self.beta[0])
        elif predX.shape[1] == beta.shape[0]:
            return sigmoid(np.dot(predX, self.beta))
        else:
            sys.error("preX doesn't have the right shape")


if __name__ == '__main__':
    center = DataCenter(3)
    Hospital1 = DataRepo(10, 'u', 3, [1,1,1,1], center)

