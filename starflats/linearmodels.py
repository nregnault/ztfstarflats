"""
This module define a class and tools to efficiently solve large
linear problems in python.

The difficulty to efficiently go through large dataset in pure python is
solved through the use of scipy.sparse to make the actual
computation. For the same price we get the possibility of solving very
large sparse system :)

The code now handle outliers rejection with rank-one update of the
cholesky decomposition. It was also significantly refactored with a new
and incompatible API.

TODO
----
- Use studentized residuals in the outliers detection
- Wrap spqr
- Implement linear equality constrainsts
- Test resolution with orthogonal factorization of A
- Damping ? 
"""

import copy
import logging

from scipy import sparse
import numpy as np

try:
    from sksparse.cholmod import cholesky_AAt, cholesky
except ImportError:
    from scikits.sparse.cholmod import cholesky_AAt, cholesky

from robuststat import binned_curve, polyfit
from fitparameters import FitParameters, structarray
from indextools import make_index
import selinv


class SparseSystem(object):
    def __init__(self, A, b, **keys):
        self.b = b
        self.keys = keys
        self.factor = cholesky_AAt(A.T, **keys)

    def update(self, A, b, subtract=True):
        self.factor.update_inplace(A.T, subtract=subtract)
        if subtract:
            self.b -= b
        else:
            self.b += b

    def refact(self, A, b):
        self.b = b
        self.factor.cholesky_AAt_inplace(A.T)
        
    def solve(self):
        x = self.factor(self.b)
        try:
            return x.squeeze()
        except AttributeError:
            return x

    def fullinv(self):
        n = self.factor.D().shape[0]
        D = sparse.eye(n,n).tocsc()
        return self.factor(D)

class SparseLocalGlobalSystem(object):
    '''When a very big block of the Hessian is known to be diagonal, we
    can take explicitly advantage of that structure to speed up things
    rather than resorting to the generic sparse factorisation.

    '''
    def __init__(self, A, b, n):
        self.b = b
        self.n = n
        A1 = A[:, :self.n]
        A2 = A[:, self.n:]
        self.D = (A1.T * A1).diagonal()
        self.B = A1.T * A2
        self.M = A2.T * A2
        #self.M = A.T * A
        self._fill()

    def _fill(self):
        self.b1 = self.b[:self.n]
        self.b2 = self.b[self.n:]
        #self.Dinv = 1./(A1.T*A1).diagonal()
        self.Dinv = 1./self.D
        #self.B = A1.T * A2
        self.Bp = self.B.copy()
        self.Bp.data *= self.Dinv[self.Bp.indices] #self.Bp = self.Dinv * B
        self.S = self.M - self.B.T * self.Bp
        #self.Mfact = cholesky(A2.T * A2 - self.B.T * self.Bp)
        self.Mfact = cholesky(self.S)
        
    def solve(self):
        x1 = self.Dinv * self.b1
        x2 = self.Mfact(self.b2)
        #x3 = self.Bp * x2
        x4 = self.Mfact(self.B.T * x1)
        return np.hstack([x1 + self.Bp * (x4 - x2), -x4 + x2])

    def update(self, W, b, subtract=True):
        ''' For now we redo the factorization. I think a true update might be envisionned
        '''
        W1, W2 = W[:, :self.n], W[:, self.n:]
        if subtract:
            D = -(W1.T * W1)
            DeltaB = -W1.T * W2
            self.S -= W2.T * W2
            self.b -= b
        else:
            D = (W1.T * W1)
            DeltaB = W1.T * W2
            self.S += W2.T * W2
            self.b += b
        self.b1 = self.b[:self.n]
        self.b2 = self.b[self.n:]
        nz = D.indices
        am1 = self.Dinv[nz]
        bovera= am1 * D.data
        Dw = -am1 / (1 + bovera) * bovera
        # A verifier D'' = D + Dw
        self.Dinv[nz] += Dw
        #assert (np.abs(self.Dinv * (self.D + D.diagonal())-1) < 1e-15).all()
        DeltaBp = DeltaB.copy()
        DeltaBp.data *= self.Dinv[DeltaBp.indices]
        D.data = Dw
        DeltaBp += D * self.B
        self.S -= self.B.T * DeltaBp
        self.Bp += DeltaBp
        self.B += DeltaB
        self.S -= DeltaB.T * self.Bp        
        self.Mfact.cholesky_inplace(self.S)

    def refact(self, A, b):
        ''' Same as above, restarting from scratch
        '''
        self.M = A.T * A
        self.b = b
        self._fill()

    def fullinv(self):
        n = self.b.shape[0]
        D = sparse.eye(n,n).tocsc()
        return self.factor(D)

    def diaginv(self):
        # Reusing the cholmod factorization is hard. In contrast
        # reusing the reordering is easy and is important for
        # performances because cholmod metis ordering is typically
        # better than the default MMD ordering included in selinv.
        Adiag = self.D
        return selinv.selinv(self.S, self.Mfact.P())

        
class RobustLinearSolver(object):
    """ Direct solver for sparse linear models with iterative outlier rejection

    Basics:
    -------

    1) This class provides the function "solve" to compute the
    Ordinary Least Square (OLS) estimator \hat{X} for the linear
    model:

    $$y = Ax + n$$,

    by the closed-form relation:

    $$\hat{x} = (A^T A)^{-1} A^T y.$$

    Both the computation of A^T A and A^T y and the resolution of the
    linear system are made using the sparse matrix package from scipy.

    2) Optionnaly one can provide a vector of weights w in which case
    the Weighted Least Square problem is first transformed to a OLS
    problem by the following modifications:

    $$y \rightarrow \diag(w) y $$

    and

    $$A \rightarrow \diag(w) A$$

    2bis) an alternative is to provide the class with a non-diagonal 
    *sparse* weight matrix. In this case, the Cholesky decomposition 
    of W is determined, W = LL' and the WLS is transformed as follows:
    
    $$y \rightarrow L' & y$$

    and 

    $$A \rightarrow L' A$$
     
    3) The function "robust_solution" performs outlier rejections,
    solving for the screening problem iteratively. The flagging of
    outliers is controlled by the flag_outliers function.

    The default implementation use a careful algorithm well adapted to
    problems with a lot of local parameters, but which may converge too
    slowly for other problems. The class can be derived easily to
    reimplement the flag_outliers function.

    4) get_cov compute the Fisher matrix of a subset of best-fit
    parameters.

    5) get_diag_cov compute the diagonal of the Fisher matrix of
    best-fit parameters in an efficient (sparse) way.

    6) diagnostic_plot is an utilitary function that displays the
    distribution of residuals and their dispersion as a function of
    given variables.

    7) weight_model is an utilitary function that fit a polynomial P
    of an explanatory variable v to the empirical rms of residuals in
    bins of v.

    8) reweight fit and apply a weight model to the problem
    """
    def __init__(self, model, y, weights=None, verbose=1, nlocal=0, **keys):
        self.verbose = verbose
        
        self.A = model.get_free_mat()
        self.y = model.get_free_y(y)
        self.model = model
        
        # Apply weights if needed
        if weights is not None:
            if weights.ndim == 1:
                assert len(weights) == len(y), (
                    'The lenght of the provided weight vector differs'
                    ' from the number of measurements')
                assert np.isfinite(weights).all(), (
                    "Some weights have non finite values,"
                    )
                self.A = self.A.tocoo()
                self.A.data *= weights[self.A.row]
                # self.y is already initialized to a copy of y
                #self.y = y * weights
                self.y *= weights
                # else not needed: self.y is already initialized to a copy of y
                #        else:
                #            self.y = y
            elif issubclass(type(weights), sparse.spmatrix):
                shape = weights.shape
                assert shape[0] == shape[1] == len(y), (
                    'The length of thr provided weight vector differs'
                    ' from the number of measurements')
                fact = cholesky(weights)
                Lt = fact.L().T
                self.A = Lt * self.A
                self.y = Lt * self.y
                
        # Cholesky decomposition
        self.A = self.A.tocsr()
        if nlocal:
            self.system = SparseLocalGlobalSystem(self.A, self.A.T * self.y, nlocal)
        else:
            self.system = SparseSystem(self.A, self.A.T * self.y, **keys)
        self.bads = None

    def suppress_one_at_a_time(self, r, local_param):
        """ For each parameter, suppress only the worst outlier

        This is hardly maintainable. Did not find an efficient yet readable way to write this.
        """       
        suppress_one = np.zeros_like(self.suppress[self.suppress])
        # Get the indices of entries corresponding to outliers
        rows, cols = self.A[self.suppress.nonzero()[0], :].nonzero()
        # Extract the residuals corresponding to outliers
        rsuppress = r[self.suppress]
        # For each parameter (column) store the list of corresponding entries in rsuppress
        index = make_index(cols)
        # For each parameter in the set of local_param, go through the
        # list of corresponding residuals, find the worst and put the
        # corresponding row in a set
        indexes = self.model.params[local_param].indexof()
        indexes = indexes[indexes != -1]
        start = indexes.min()
        stop = indexes.max()
        worst = list(set([rows[ind][np.argmax(rsuppress[rows[ind]])] for ind in index
                          if (cols[ind[0]] >= start) and
                              (cols[ind[0]] <= stop)]))
        #raise ValueError()
        # Not sure the sorting is necessary
        worst.sort()
        # Now we indeed suppress the rows corresponding to the worst
        # outlier in each column
        suppress_one[worst] = True
        self.suppress[self.suppress] = suppress_one
        #print np.bincount(self.A[self.suppress.nonzero()[0], :].nonzero()[1]).max()
        #print np.where(np.bincount(self.A[~self.suppress, :].nonzero()[1]) == 0)[0]
        
    def robust_solution(self, **keys):
        """Compute a robust solution to the linear system, iteratively discarding
        outliers.

        Parameters:
        -----------
        keys: keyword arguments are passed directly to self.flag_outliers.
        """
        x = self.system.solve()
        self.flag_outliers(x, **keys)
        niter = 1
        while self.suppress is not None or self.restore is not None:
            if self.suppress is not None:
                C = self.A[self.suppress.nonzero()[0],:]
                self.system.update(C, C.T * self.y[self.suppress])
            if self.restore is not None:
                C = self.A[self.restore.nonzero()[0],:]
                self.system.update(C, C.T * self.y[self.restore], subtract=False)
            x = self.system.solve()
            self.flag_outliers(x, **keys)
            niter += 1
        if self.verbose:
            Nout = self.bads.sum()
            N = float(len(self.bads))
            print(("Converged in %d iter, cutting %d/%d (%.1g%%) of the measurements" % (niter, Nout, N, Nout / N * 100)))
        return x

    def flag_outliers(self, x, nsig=5, local_param=None):
        """Given a solution x, flag measurements whose residuals to the fit

        deviates by more than nsig sigma from zero. The standard
        deviation of the residuals is evaluated as the square root of
        the chi2 divided by the number of degrees of freedom.

        If local_param is provided, make sure that for each local
        parameter at most one measurement affecting this parameter is
        suppressed.

        This methods uses the provided x and self.bads as starting
        points. It sets self.suppress, self.restore and self.bads.

        """
        r = (self.A * x).squeeze()
        r -= self.y
        r *= r
        if self.bads is not None:
            chi2 = np.sum(r[~self.bads])
            Nmeas = len(r) - np.sum(self.bads)
        else:
            chi2 = np.sum(r)
            Nmeas = len(r)
        assert np.isfinite(chi2)
        dof = Nmeas - self.A.shape[1]

        if self.verbose:
            med_chi2 = np.median(r)
            print(("Chi2: %f, D.o.F.: %d, Chi2/DoF: %f, median Chi2: %f" % (chi2, dof, chi2 / dof, med_chi2)))

        bads = r > (nsig ** 2 * chi2 / dof)

        if self.bads is not None:
            self.suppress = bads & ~self.bads
            self.restore = self.bads & ~bads
        else:
            self.suppress = bads
            self.restore = [False]
        if local_param is not None and self.suppress.any():
            self.suppress_one_at_a_time(r, local_param)

        Nsup = np.sum(self.suppress)
        Nres = np.sum(self.restore)
        if Nsup == 0:
            if self.bads is None:
                self.bads = self.suppress
            self.suppress = None
        else:
            if self.bads is not None:
                self.bads[self.suppress] = True
            else:
                self.bads = self.suppress
        if Nres == 0:
            self.restore = None
        else:
            self.bads[self.restore] = False
        if self.verbose:
            print(("Suppress %d new outliers" % Nsup))
            print(("Restore %d measurements" % Nres))
        self.r = r
        #self.bads = bads
        self.chi2 = chi2

    def get_res(self, y, x=None):
        """ Compute the residual
        """
        return y-self.model(x)

    def get_wres(self, x=None):
        """ Compute weighted residual
        """
        if x is None:
            x = self.x
        r = (self.A * x).squeeze()
        r -= self.y
        return r
    
    def get_cov(self):
        """ Compute the full covariance matrix

        Computation of the full inverse of the FIM which can be pretty
        expansive for large problems. See get_diag_cov and get_block_cov
        for a workable alternative in large sparse problems.
        """
        return self.system.fullinv()

    def get_block_cov(self, block):
        """ Compute a block of the covariance matrix

        Computation of a block (typically small) of the inverse of the FIM.
        """
        if self.bads is not None:
            M = self.A[(~self.bads).nonzero()[0], :]
        else:
            M = self.A
        nonblock = np.ones(self.A.shape[1], dtype=bool)
        nonblock[block] = False
        nonblock = np.where(nonblock)[0]
        A1 = M[:, nonblock]
        A2 = M[:, block]
        A22 = A2.T * A2
        A12 = A1.T * A2
        bigsys = SparseSystem(A1, A12)
        sol = A12.T * bigsys.solve()
        B = A22 - sol.todense()
        return np.linalg.inv(B)
    
    def get_diag_cov(self):
        """ Compute the diagonal of the covariance matrix.

        This uses the selinv algorithm from Lin Lin et al. 2011 and can
        be much faster than computing the full covariance matrix if the
        FIM is sparse.
        """
        if self.bads is not None:
            M = self.A[(~self.bads).nonzero()[0], :]
        else:
            M = self.A
        # Reusing the cholmod factorization is hard. In contrast
        # reusing the reordering is easy and is important for
        # performances because cholmod metis ordering is typically
        # better than the default MMD ordering included in selinv.
        return selinv.selinv(M.T * M, self.system.factor.P())
    
    def diagnostic_plot(self, disperse=[],deg=2, pbins=None, nbins=10, data=False, keep_outliers=False, win=1):
        """ Plot the distribution of residuals.

        Optionnally, plot the dispersion of residuals against the
        provided variables.

        Parameters:
        -----------
        disperse: list
                  the list of arrays to be used as abscisse for the
                  dispersion plots.
        """
        import matplotlib.pyplot as plt
        from saunerie.plottools import binplot
        from scipy.stats import chi2


        # self.r always exists and has the length of the data
        # while self.bads can be None
        if keep_outliers or self.bads is None:
            Nmeas = len(self.r)
        else:
            Nmeas = float(np.sum(~self.bads))
        npars = self.A.shape[1]
        if not disperse:
            plt.figure(win)
            plt.clf()
            if keep_outliers:
                N, bins, h = plt.hist(self.r, bins=100, normed=True, log=True, color='k', histtype="step", label="residuals")
            else:
                N, bins, h = plt.hist(self.r[~self.bads], bins=100, normed=True, log=True, color='k', histtype="step", label="residuals")
            x = 0.5 * (bins[1:] + bins[:-1])
            plt.plot(x, chi2.pdf(x, 1 - npars / Nmeas), 'r-', label="chi2 pdf")
            plt.legend()
            
        for x in disperse:
#            if len(disperse) > 1:
#                plt.figure()
            # even if there is only one "disperse" I find it better to have it on its own plot
            plt.figure(win)
            plt.clf()
            if keep_outliers:
                binplot(x, self.r, data=data, marker='s', nbins=nbins, bins=pbins, capsize=0, color="k", label="binned residuals")
            else:
                binplot(x[~self.bads], self.r[~self.bads], data=data, marker='s', nbins=nbins, bins=pbins, capsize=0, color="k", label="binned residuals")
            plt.axhline(1 - npars / Nmeas, color='k', ls='-.', label="Expected residual value")
            if pbins is not None:
                w_model = self.weight_model(x,deg=deg, bins=pbins)
                z = np.linspace(*(plt.axis()[:2]))
                plt.plot(z, np.polyval(w_model, z), color='r', label="deg %d model" % (deg))
            plt.legend()
            
    def weight_model(self, variable, deg=2, bins=10):
        """ Todo: Turn that into something proper
        """
        
        # self.bads can be None:
        if self.bads is None:
            x, y, ey = binned_curve(variable, self.r, error=True, bins=bins)
        else:
            x, y, ey = binned_curve(variable[~self.bads], self.r[~self.bads], error=True, bins=bins)
        goods = ey > 0
        p, err = polyfit(x[goods], y[goods], deg=deg, w=1. / ey[goods] ** 2)
        return p.reshape(-1)

    def ndof(self):
        if self.bads is None:
            Nmeas = self.A.shape[0]
        else:
            Nmeas = float(np.sum(~self.bads))
        npars = self.A.shape[1]
        return Nmeas - npars
        
    def reweight(self, variable, nan_exception=False, **keys):
        Nmeas = float(np.sum(~self.bads))
        npars = self.A.shape[1]

        w_model = self.weight_model(variable, **keys)
        if self.verbose:
            print(("Warping weights according to a polynomial fit of residuals: " + str(w_model)))
        weights = np.sqrt(1 - npars/Nmeas) / np.sqrt(np.polyval(w_model, variable))
        if nan_exception:
            weights[np.isnan(weights)] = np.nanmax(weights)
        
        self.A = self.A.tocoo()
        self.A.data *= weights[self.A.row]
        self.y *= weights
        self.A = self.A.tocsr()
        
        # Redo the numerical factorisation
        self.system.refact(self.A, self.A.T * self.y)
        self.bads = None
        return weights

class LinearModel(object):
    """Facilitate composite descriptions of linear problems of the kind:
    $$y = A_0 x_0 + A_1 x_1 + ... + A_J x_J + n$$,

    that can be subject to linear constraints of the kind:
    $$B x = d$$

    1) Each of the $A_j$ are provided in sparse format through the triplet
    rows, cols and vals. LinearModels then define the concatenation
    operator + that will handle the concatenation of matrices.

    2) Provided that the columns of the concatenated matrix $A = \left[ A_0
    A_1 ... A_J]$ remain linearly independant, the result can be passed through 
    RobustLinearSolver.

    3) LinearModel handles an instance of fitparameters and makes it
    easy to fix or free a subset of parameters.

    4) Not Implemented: we may provide least-square non-trivial linear
    equality constraints at some point. This is in wait for the wrapping
    of spqr.
    """
    def __init__(self, rows, cols, vals, struct=None, name=None, valid=None):
        self.rows = np.asarray(rows)
        self.cols = np.asarray(cols)
        self.vals = np.asarray(vals)
        self._valid = valid
        self.struct = struct
        if self.struct is None:
            if name is None:
                self.struct = [self.cols.max() + 1]
            else:
                self.struct = [(name, self.cols.max() + 1)]
        self.params = FitParameters(self.struct)

    def __add__(self, m):
        # Concatenate all the pieces of the matrix
        def none_to_ones(val, length):
            if val is None:
                return np.full(length, True, dtype=bool)
            else:
                return val
        if m._valid is not None or self._valid is not None:
            valid = np.hstack([none_to_ones(self._valid, len(self.rows)),
                               none_to_ones(m._valid, len(m.rows))])
        else:
            valid = None
        return LinearModel(np.hstack([self.rows, m.rows]),
                           np.hstack([self.cols, m.cols + self.cols.max() + 1]),
                           np.hstack([self.vals, m.vals]),
                           struct=self.struct + m.struct, valid=valid)
    
    def getmat(self):
        """ Lazy matrix construction"""
        if not hasattr(self, 'A'):
            if self._valid is None:
                self.A = sparse.coo_matrix((self.vals, (self.rows, self.cols))).tocsc()
            else:
                self.A = sparse.coo_matrix((self.vals[self._valid], (self.rows[self._valid], self.cols[self._valid]))).tocsc()
        return self.A
        
    def __call__(self, p=None):
        self.getmat()
        if p is None:
            return self.A * self.params.full
        else:
            return self.A * p

    def get_free_mat(self):
        return self.getmat()[:, np.nonzero(self.params._free)[0]]

    def get_free_y(self, y):
        if not self.params._free.all():
            # I don't understand why we subtract A[:,where the param
            # is free] * the parameter value where the param is not
            # free
            return y - self.getmat()[:, np.nonzero(~self.params._free)[0]] * self.params.full[~self.params._free]
        else:
            # We need a copy, otherwise self.y is a reference to the
            # initial data, that will be changed by a reweighting
            # return y
            return copy.copy(y)
        
def linear_func(*args, **keys):
    N = len(args[0])
    for arg in args:
        assert len(arg) == N
    data = np.hstack(args)
    cols = np.repeat(np.arange(len(args)), N)
    rows = np.tile(np.arange(N), len(args))
    return LinearModel(rows, cols, data, **keys)    
    
def indic(col_index, val=1.0, but=None, **keys):
    """ Construct a sparse matrix A such that the model:
    
    y = x(i) * val

    with x a parameter constant for all measures sharing the same
    i can be written in matrix form as:

    Y = A X

    Parameters
    ----------
    col_index: array of int
               the indice of the bin to which each measurement belongs
    val: array of float
         the multiplicative factor for each measurement
    but: int
         if provided x(but) is withdrawn from the fit (the
         corresponding A matrix columns is suppressed)

    Exemple:
    --------
    
    Let us assume that we have noisy measurements of the flux of a bunch
    of stars uniquely identified by an integer 0 <= star_id < Nstar. We
    form the vector star_id such that for each measurement Y[i] for 0 <=
    i < Nmeas, star_id[i] gives the id of the corresponding star.

    The matrix implementing the data model:
    Y = A * star_flux + n

    can be obtained as
    rows, cols, vals = indic(star_id)
    A = scipy.sparse.coo_matrix((vals, (rows, cols)))
    """
    cols = col_index.copy()
    rows = np.arange(len(cols))
    data = np.ones(len(cols)) * val
    if but is not None:
        index = (cols != but)
        cols[cols > but] -= 1
        cols = cols[index]
        rows = rows[index]
        data = data[index]
    return LinearModel(rows, cols, data, **keys)

    

class TestCase():
    """ Construct a fake y=Ax+n sparse linear system

    Which has a quite specific structure:
    - G global parameters
    - L local parameters
    - M measurements
    ie:
    - x has shape (G + L, 1)
    - A has shape (M, G + L)
    - y and n have shape (M, 1)
    """
    def __init__(self, G, L, M):
        cols1 = np.random.randint(0, high=G, size=M)
        index = cols1 != 0
        cols1 = cols1[index]
        model1 = indic(cols1, name="G")
        cols2 = np.repeat(np.arange(L), M / L)
        model2 = indic(cols2, name='L')
        self.model = model1 + model2
        self.x = np.random.randn(L + G)
        self.y = self.model(self.x)
        self.flux = np.random.random(M)
        self.y += np.random.randn(M) * (np.polyval((1,1,1), self.flux))
        self.model.params['G'].fix(0, self.x[0])
        self.x = structarray(self.x, self.model.params._struct)
        
if __name__ == "__main__":
    testcase = TestCase(10 ** 3, 10 ** 4, 10 ** 5)
    lm = RobustLinearSolver(testcase.model, testcase.y)
    solution1 = lm.robust_solution(nsig=5)
    lm.diagnostic_plot([testcase.flux])
    lm.reweight(testcase.flux)
    solution2 = lm.robust_solution(nsig=5)
    lm.diagnostic_plot([testcase.flux])

    from matplotlib.pyplot import show, hist, figure
    figure()
    testcase.model.params.free=solution1
    print(((testcase.model.params['G'].full - testcase.x['G']).std()))
    hist(testcase.model.params['G'].full - testcase.x['G'], bins=20)
    
    testcase.model.params.free=solution2
    print(((testcase.model.params['G'].full - testcase.x['G']).std()))
    hist(testcase.model.params['G'].full - testcase.x['G'], bins=20)
    
    show()
    










