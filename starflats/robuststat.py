import numpy as np
from indextools import make_index


#----------------- robust stat stuff -----------------------------
# def mad(x):
#     """Median of absolute deviation

#     Very robust estimator of the standard deviation. 67% noisier at the
#     normal.
#     """
#     m = np.ma.median(x)
#     return np.ma.median(abs(x - m)) * 1.4826

def mad(data, axis=0, scale=1.4826):
    """
    Median of absolute deviation along a given axis.  

    Normalized to match the definition of the sigma for a gaussian
    distribution.
    """
    if data.ndim == 1:
        med = np.ma.median(data)
        ret = np.ma.median(np.abs(data-med))
    else:
        med = np.ma.median(data, axis=axis)
        if axis>0:
            sw = np.ma.swapaxes(data, 0, axis)
        else:
            sw = data
        ret = np.ma.median(np.abs(sw-med), axis=0)
    return scale * ret

def fast_binned_curve(x, y, width=5, rms=False, error=False, robust=False):
    ''' Same as equal_binned_curve but faster because we treat the remainder differently'''
    _x = x[:len(x)//width * width].reshape((-1, width))
    _y = y[:len(x)//width * width].reshape((-1, width))
    if robust:
        mean_x, mean_y = np.median(_x, axis=1), np.median(_y, axis=1)
    else:
        mean_x, mean_y = np.mean(_x, axis=1), np.mean(_y, axis=1)
    ret = [mean_x, mean_y]
    return ret


def equal_binned_curve(x, y, width=5, rms=False, error=False):
    """ Average x and y in bins of x of size width
    """
    index = np.isfinite(x) & np.isfinite(y)
    _x = x[index]
    _y = y[index]
    s = _x.argsort()
    _x = _x[s]
    _y = _y[s]
    bins = (np.arange(len(_x)) / width).astype(int)
    n = np.bincount(bins)
    mean_x, mean_y = np.bincount(bins, _x) / n, np.bincount(bins, _y) / n
    ret = [mean_x, mean_y]
    if rms or error:
        rms_y = np.sqrt(np.bincount(bins, (_y - mean_y[bins]) ** 2) / n)
        if rms:
            ret.append(rms_y)
        if error:
            ret.append(rms_y / np.sqrt(n))

    return ret

def binaverage(x, y, binkey=None, robust=False, scale=True):
    if x.shape != y.shape: raise ValueError('x and y have different length')
    if binkey is None:
        binkey = x
    ind = make_index(binkey)
    ybinned = [y[e] for e in ind]
    xbin = [x[e] for e in ind]
    if robust is True:
        yplot = [np.median(e) for e in ybinned]
        yerr = np.array([mad(e) for e in ybinned])
    elif robust:
        yres = [robust_average(e, sigma=None, clip=robust, mad=False, axis=0)
                for e in ybinned]
        yplot = [e[0] for e in yres]
        yerr = [np.sqrt(e[3]) for e in yres]
    else:
        yplot = [np.mean(e) for e in ybinned]
        yerr = np.array([np.std(e) for e in ybinned])

    if scale:
        yerr /= np.sqrt(np.array([len(e) for e in ybinned]))
    xplot = [np.mean(e) for e in xbin]
    return np.array(xplot), np.array(yplot), np.array(yerr)

def binned_curve(x, y, bins=10, rms=False, error=False):
    """ Average x and y in bins of x of size width

    Exemple
    -------
    >>> x = np.linspace(0, 1, 100)
    >>> y = np.random.randn(100)
    >>> 
    """
    index = np.isfinite(x) & np.isfinite(y)
    _x = x[index]
    _y = y[index]
    maxv = _x.max()
    
    if np.isscalar(bins):
        bins = np.linspace(x.min(), maxv * (1 + np.sign(maxv) * 1e-7), bins)
        
    binned = np.digitize(_x, bins) - 1

    n = np.bincount(binned)
    mean_x = np.bincount(binned, _x) / n
    mean_y = np.bincount(binned, _y) / n

    ret = [mean_x, mean_y]
    
    if rms or error:
        rms_y = np.sqrt(np.bincount(binned, (_y - mean_y[binned]) ** 2) / n)
        if rms:
            ret.append(rms_y)
        if error:
            ret.append(rms_y / np.sqrt(n))
    return ret


def slide_win(a, func, width=5):
    """ Apply func to vector a on a sliding window of size width.

    Return a vector of same length than a.
    """
    if len(a) > width:
        l = len(a) - width + 1
    else:
        l = 1
        width = len(a)
    res = np.zeros(l)
    for i in range(l):
        res[i] = func(a[i:i + width])
    res2 = np.zeros(len(a))
    res2[width // 2:width // 2 + l] = res
    res2[0:width // 2] = res[0]
    res2[width // 2 + l:] = res[-1]
    return res2


def robust_average(a, sigma=None, axis=None, clip=5, mad=False,
                   mask=None, mini_output=False, scale_var=True):
    """ Perform an iteratively clipped weighted averaging.

    After each step, chi2 for values in a are computed and outlying
    values are weighted to zeros before a second step. A mask of
    detected outliers is kept and return.

    Parameters:
    -----------
    sigma: the average will be weighted by 1/sigma**2
    axis: int, compulsory
          Axis along which the means are computed.
    clip: float
          Outliers lying at more than clip * std from the mean will be
          iteratively discarded from the averaging.
    mad: bool
         If True, use the median of absolute deviation to evaluate the
         width of the distribution. Can be usefull for large numbers
         of outliers but slow down the computation.
    mask: bool array
          Specify values to discard from the mean (outlier rejection
          is still performed besides this).
    mini_output: bool
                 Shorten the return value to the average.
    scale_var: bool
               If true, the return variance is 1 / w ** 2 * std**2
               where std is the standard deviation of residuals else,
               return simply 1 / (w ** 2). The scaled estimate is not
               reliable for small samples.

    Returns
    -------
    m: the weighted average along the given axis
    if mini_output:
    mask: the mask finally applied to the data
    var: the inverse sum of weights scaled by the variance of the residuals
    rchi2: the reduced chi2
    """
    data = np.ma.masked_invalid(a)
    if mask is not None:
        data.mask |= mask
    #reps = [1,]*len(a.shape)
    #reps[axis] = a.shape[axis]
    mshape = [s for s in data.shape]
    mshape[axis] = 1

    if sigma is None:
        sigma = np.ones(a.shape)

    wrong = np.ones(1, dtype='bool')  # do it once
    while wrong.any():
        m, w = np.ma.average(data, weights=1 / sigma ** 2,
                             axis=axis, returned=True)
        m = m.reshape(mshape)
        r = (data - m) / sigma
        if mad:  # robust but slow. Can save a whole iteration though
            dev = np.ma.median(abs(r), axis=axis).reshape(mshape) * 1.4826
        else:
            dev = r.std(axis=axis).reshape(mshape)
        if data.shape[axis] < 2:
            print('Warning: cannot compute a meaningful deviation of residuals')
            dev = np.ones_like(dev)
        wrong = abs(r) > clip * dev
        data.mask = wrong.filled(fill_value=True)
    dev = dev.squeeze()
    var = 1 / w
    if scale_var:
        var *= dev ** 2
    if mini_output:
        return m.squeeze()
    else:
        return m.squeeze(), data.mask, var, dev ** 2


def _polymod(x, deg):
    cols = np.repeat(np.arange(deg + 1), len(x))
    rows = np.tile(np.arange(len(x)), deg + 1)
    data = np.concatenate(
        [x ** i for i in range(deg, 0, -1)] + [np.ones(len(x))])
    return rows, cols, data


def polyfit(x, y, w=None, deg=1, full_output=False, polymod=_polymod):
    """ numpy.polyfit clone allowing for easy data points weighting

    Deprecated: no interest wrt numpy.polyfit

    Beware nothing is done to make this stable for high degree
    polynomials. Primary intent are linear fit.

    Input:
    ------
    x: array_like, shape (M,)
       x-coordinates of the M sample points ``(x[i], y[i])``.
    y: array_like, shape (M,)
       y-coordinates of the sample points.

    Optional arguments:
    -------------------
    w: array_like, shape (M,)
       weights associated to each data points, use weigths equal to 1
       if not provided

    full_output: If set to True, return also the chi2 of the fit, and
                 the complete linearModel object.

    Return:
    -------
    p: ndarray, shape (deg,)
       Best fit polynomial coefficients, higher order first

    err: ndarray, shape (deg,)
       Corresponding 1 sigma uncertainty on the polynomial
       coefficients according to the inverse Fisher matrix. You may
       want to scale this by sqrt(chi2) if needed

    chi2: if full_output is set to True, the reduced chi2 resulting
          from the minimisation (approximatively the empirical
          variance of the residuals).
    """
    from . import linearmodels as lm
    model = lm.LinearModel(*polymod(x, deg))
    solver = lm.RobustLinearSolver(model, y, weights=w, verbose=0)
    p = solver.system.solve()
    err = solver.get_cov()
    if full_output:
        r = (solver.A * p).squeeze()
        r -= solver.y
        r *= r
        return p, err, r.sum() / solver.ndof(), solver
        
    else:
        return p, err


def robust_polyfit(x, y, w=None, deg=1, full_output=False, nsig=4, verbose=0):
    """ numpy.polyfit clone allowing for data points weighting and
    outliers detection.

    Beware nothing is done to make this stable for high degree
    polynomials. Primary intent are linear fit.

    Input:
    ------
    x: array_like, shape (M,)
       x-coordinates of the M sample points ``(x[i], y[i])``.
    y: array_like, shape (M,)
       y-coordinates of the sample points.


    Optional arguments:
    -------------------
    w: array_like, shape (M,)
       weights associated to each data points, use weigths equal to 1
       if not provided w is the diagonal of the square root of the
       inverse of the covariance matrix. You should provide 1/sigma
       for optimal weighting.

    full_output: bool
       If set to True, return also the chi2 of the fit, and the
       complete linearModel object.

    nsig: float
       Iteratively reject outlier at nsig * sigma where sigma is the
       rms of the residuals.

    Return:
    -------
    p: ndarray, shape (deg,)
       Best fit polynomial coefficients, higher order first
    err: ndarray, shape (deg,)
       Corresponding 1 sigma uncertainty on the polynomial
       coefficients according to the inverse Fisher matrix. You may
       want to scale this by sqrt(chi2) if needed

    Optional return:
    ----------------
    rchi2: float
       if full_output is set to True, the reduced chi2 resulting from
       the minimisation (approximatively the empirical variance of the
       residuals).
    M: calib.LinearModel
       The complete model object.
    """
    from . import linearmodels as lm
    model = lm.LinearModel(*_polymod(x, deg))
    solver = lm.RobustLinearSolver(model, y, weights=w, verbose=verbose)
    p = solver.robust_solution(nsig=nsig)
    err = solver.get_cov()
    if full_output:
        return p, err, solver.r[~solver.bads].sum() / solver.ndof(), solver
    else:
        return p, err


def bootstrap(data, estimator, n=100, show=False):
    """ Computes (unefficiently) the bootstrap distribution of an estimator

    Parameters:
    -----------

    - data: a column vector providing measure and measurement error
    respectively. Slicing for bootstrap is made on the first
    dimension.
    - estimator: a function returning a couple (estimate, uncertainty).
    - n: the number of random dataset drawn
    - show: if True display the bootstrap distribution

    Return
    ------
    A triplet holding:
    - K: the estimated value on the whole dataset
    - B_err: the standard deviation of the bootstrap_distribution
    - bootstrap_distrib: the bootstrap distribution
    """
    K, K_err = estimator(data)
    try:
        m = len(K)
    except:
        K = [K]
        K_err = [K_err]
        m = 1
    bootstrap_distrib = np.zeros((n, m))
    for i in range(n):
        index = np.random.randint(0, data.shape[0], size=data.shape[0])
        bootstrap_distrib[i, :] = estimator(data[index, ...])[0]
    B_err = np.zeros(m)
    for i in range(m):
        B_err[i] = bootstrap_distrib[:, i].std()
        if show:
            import matplotlib.pyplot as plt
            from pylab import normpdf
            plt.figure()
            plt.hist(bootstrap_distrib[:, i], normed=True)
            x = np.linspace(K[i] - 5 * B_err[i], K[i] + 5 * B_err[i])
            plt.plot(x, normpdf(x, K[i], B_err[i]), 'k')
            ax = plt.gca()
            plt.text(0.1, 0.95,
                     r'${\rm normal:}%.3e \pm %.1e$' % (K[i], K_err[i]),
                     transform=ax.transAxes)
            plt.text(0.1, 0.85,
                     r'${\rm btstrap:}%.3e \pm %.1e$' %
                     (np.mean(bootstrap_distrib[:, i]), B_err[i]),
                     transform=ax.transAxes)
    return K, B_err, bootstrap_distrib


def rtls(x, y, ex, ey, cut=3):
    ind = np.ones(len(x), dtype=bool)
    indold = np.zeros(len(x), dtype=bool)
    if ex is None:
        ex = np.ones(len(x))
    if ey is None:
        ey = np.ones(len(y))
    i = 0
    while((ind != indold).any()):
        #print "%d new outliers"%sum((ind != indold).astype('int'))
        indold = ind
        a, b = total_least_squares(x[ind], y[ind], ex[ind], ey[ind])
        rx = (a + b) * (y - a * x - b) / np.sqrt(1 + (a + b) ** 2)
        ry = a * rx
        r = np.sqrt((rx ** 2 + ry ** 2) / (ey ** 2 + a ** 2 * ex ** 2))
        if i < 10:
            ind = (r < cut ** 2 * np.median(r))
        else:
            ind = (r < cut ** 2 * np.median(r)) & indold
        i = i + 1
    return a, b, ind


def total_least_squares(data1, data2, data1err=None, data2err=None,
                        print_results=False, ignore_nans=True, intercept=True,
                        return_error=False, inf=1e10):
    """
    Use Singular Value Decomposition to determine the Total Least
    Squares linear fit to the data.

    (e.g. http://en.wikipedia.org/wiki/Total_least_squares)
    data1 - x array
    data2 - y array

    if intercept:
        returns m,b in the equation y = m x + b
    else:
        returns m

    print tells you some information about what fraction of the
    variance is accounted for

    ignore_nans will remove NAN values from BOTH arrays before computing

    Parameters
    ----------
    data1, data2 : np.ndarray
        Vectors of the same length indicating the 'x' and 'y' vectors to fit

    data1err, data2err : np.ndarray or None
        Vectors of the same length as data1,data2 holding the 1-sigma
        error values

    """

    if ignore_nans:
        badvals = np.isnan(data1) + np.isnan(data2)
        if data1err is not None:
            badvals += np.isnan(data1err)
        if data2err is not None:
            badvals += np.isnan(data2err)
        goodvals = True - badvals
        if goodvals.sum() < 2:
            if intercept:
                return 0, 0
            else:
                return 0
        if badvals.sum():
            data1 = data1[goodvals]
            data2 = data2[goodvals]

    if intercept:
        dm1 = data1.mean()
        dm2 = data2.mean()
    else:
        dm1, dm2 = 0, 0

    arr = np.array([data1 - dm1, data2 - dm2]).T

    U, S, V = np.linalg.svd(arr, full_matrices=False)

    # v should be sorted.
    # this solution should be equivalent to v[1,0] / -v[1,1]
    # but I'm using this: 
    # http://stackoverflow.com/questions/5879986/pseudo-inverse-of-sparse-matrix-in-python
    M = V[-1, 0] / -V[-1, -1]

    varfrac = S[0] / S.sum() * 100
    if varfrac < 50:
        raise ValueError("ERROR: SVD/TLS Linear Fit accounts for less than hal"
                         "f the variance; this is impossible by definition.")

    # this is performed after so that TLS gives a "guess"
    if data1err is not None or data2err is not None:
        try:
            from scipy.odr import RealData, Model, ODR
        except ImportError:
            raise ImportError("Could not import scipy;"
                              " cannot run Total Least Squares")

        def linmodel(B, x):
            if intercept:
                return B[0] * x + B[1]
            else:
                return B[0] * x 

        if data1err is not None:
            data1err = data1err[goodvals]
            data1err[data1err <= 0] = inf
        if data2err is not None:
            data2err = data2err[goodvals]
            data2err[data2err <= 0] = inf

        if any([data1.shape != other.shape
                for other in (data2, data1err, data2err)]):
            raise ValueError("Data shapes do not match")

        linear = Model(linmodel)
        data = RealData(data1, data2, sx=data1err, sy=data2err)
        B = data2.mean() - M * data1.mean()
        beta0 = [M, B] if intercept else [M]
        myodr = ODR(data, linear, beta0=beta0)
        output = myodr.run()

        if print_results:
            output.pprint()

        if return_error:
            return np.concatenate([output.beta, output.sd_beta])
        else:
            return output.beta

    if intercept:
        B = data2.mean() - M * data1.mean()
        if print_results:
            print(("TLS Best fit y = %g x + %g" % (M, B)))
            print(("The fit accounts for %0.3g%% of the variance." % (varfrac)))
            print(("Chi^2 = %g, N = %i" % (
                ((data2 - (data1 * M + B)) ** 2).sum(),
                data1.shape[0] - 2)))
        return M, B
    else:
        if print_results:
            print(("TLS Best fit y = %g x" % (M)))
            print(("The fit accounts for %0.3g%% of the variance." % (varfrac)))
            print(("Chi^2 = %g, N = %i" % (((data2 - (data1 * M)) ** 2).sum(),
                                          data1.shape[0] - 1)))
        return M


# ------------------ Pierce method for outlier rejection ----------------------

#!/usr/bin/python
#
# peirce_dev.py
# created 16 July 2013
#
#
import numpy
import scipy.special


#### FUNCTION ####
def peirce_dev(N, n, m):
    """ Compute Pierce's criterion

    Code taken from the wikipedia page
    N :: number of observations
    n :: number of outliers to be removed
    m :: number of model unknowns (e.g., regression parameters)
    """
    # Assign floats to input variables:
    N = float(N)
    n = float(n)
    m = float(m)

    # Check number of observations:
    if N > 1:
        # Calculate Q (Nth root of Gould's equation B):
        Q = (n ** (n / N) * (N - n) ** ((N - n) / N)) / N

        # Initialize R values (as floats)
        Rnew = 1.0
        Rold = 0.0  # <- Necessary to prompt while loop

        # Start iteration to converge on R:
        while (abs(Rnew - Rold) > (N * 2.0e-16)):
            # Calculate Lamda
            # (1/(N-n)th root of Gould's equation A'):
            ldiv = Rnew ** n
            if ldiv == 0:
                ldiv = 1.0e-6
            Lamda = ((Q ** N) / (ldiv)) ** (1.0 / (N - n))

            # Calculate x-squared (Gould's equation C):
            x2 = 1.0 + (N - m - n) / n * (1.0 - Lamda ** 2.0)

            # If x2 goes negative, return 0:
            if x2 < 0:
                x2 = 0.0
                Rold = Rnew
            else:
                # Use x-squared to update R (Gould's equation D):
                Rold = Rnew
                Rnew = numpy.exp((x2 - 1) / 2.0) * scipy.special.erfc(
                    numpy.sqrt(x2) / numpy.sqrt(2.0))
    else:
        x2 = 0.0
    return x2


def peirce_meandev(data):
    data = np.asarray(data)
    m = data.mean()
    d = data.std()
    N = len(data)
    n = 1
    nold = 0
    while nold != n:
        R = np.sqrt(peirce_dev(N, n, 1))
        delta = R * d
        out = abs(data - m) > delta
        if sum(out) > nold:
            n = sum(out) + 1
            nold = sum(out)
        else:
            nold = n
    return R, out


def peirce_out(data):
    data = np.asarray(data)
    m = 0
    d = 1
    N = len(data)
    n = 1
    nold = 0
    while nold != n:
        R = np.sqrt(peirce_dev(N, n, 0))
        delta = R * d
        out = abs(data - m) > delta
        if sum(out) > nold:
            n = sum(out) + 1
            nold = sum(out)
        else:
            nold = n
    return R, out
