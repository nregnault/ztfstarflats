#!/usr/bin/env python3

import numpy as np
from astropy.time import Time
from scipy.stats import median_abs_deviation
from ztfimg.utils.tools import ccdid_qid_to_rcid
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import erfa
from indextools import make_index


ztffiltercodes = ['zg', 'zr', 'zi']
gaiarefmjd = Time(2015.5, format='byear').mjd
quadrant_width_px, quadrant_height_px = 3072, 3080
quadrant_size_px = {'x': quadrant_width_px, 'y': quadrant_height_px}

idx2markerstyle = ['*', 'x', '.', 'v', '^', '<', '>', 'o', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', '+', 'D', 'd']

def radectoaltaz(ra, dec, lat, lon, jd):
    """
    Convert Right Ascension (RA) and Declination (Dec) to Altitude (a) and Azimuth (az).


    This function takes the Right Ascension and Declination of a celestial object, the
    observer's latitude (lat), observer's longitude (lon), and Julian Date (jd) as input
    and computes the Altitude (a) and Azimuth (az) of the object as observed from the
    specified location and time.

    Parameters:
        ra (float): Right Ascension of the celestial object in radians.
        dec (float): Declination of the celestial object in radians.
        lat (float): Observer's latitude in radians.
        lon (float): Observer's longitude in radians.
        jd (float): Julian Date representing the time of observation.

    Returns:
        Tuple (az, a, st, H):
        - az (float): Azimuth of the object in radians.
        - a (float): Altitude of the object in radians.
        - st (float): Sidereal Time in radians.
        - H (float): Hour Angle of the object in radians.
    """
    pi = 3.141592653589793
    # Calculate the Greenwich Mean Sidereal Time (GMST) for the given Julian Date.
    gmst = erfa.gmst06(jd, 0, jd, 0)
    st = np.mod(gmst + lon, 2 * pi)  # Calculate the Sidereal Time
    H = (st - ra)  # Calculate the Hour Angle
    H = np.mod(H, 2 * np.pi)

    # Calculate the Azimuth and Altitude.
    az = np.arctan2(np.sin(H), np.cos(H) * np.sin(lat) - np.tan(dec) * np.cos(lat))
    a = np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(H))
    az = az - pi
    az = np.mod(az, 2 * np.pi)

    return az, a, st, H


def get_airmass_derivatives(ra, dec, obs_date, lat=33.356*np.pi/180., lon=-116.863*np.pi/180.):
    gmst = erfa.gmst06(obs_date, 0, obs_date, 0)
    st = np.mod(gmst + lon, 2 * np.pi)  # Calculate the Sidereal Time
    H = (st - ra)  # Calculate the Hour Angle
    H = np.mod(H, 2 * np.pi)

    A = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(H)
    X = 1/A
    dXdra = -np.cos(lat)*np.cos(dec)*np.sin(H)/A**2
    dXddec = -(np.sin(lon)*np.cos(dec)-np.cos(lat)*np.sin(dec)*np.cos(H))/A**2

    return X, (dXdra, dXddec)


def get_airmass(ra, dec, obs_date, lat=33.356*np.pi/180, lon=-116.863*np.pi/180):
    """
    Calculate the airmass for a celestial object.

    This function computes the airmass for a celestial object with given Right Ascension
    (RA) and Declination (Dec) observed from a specific location and date.

    Parameters:
        ra (float): Right Ascension of the celestial object in radians.
        dec (float): Declination of the celestial object in radians.
        obs_date (float): Julian Date representing the date of observation.
        lat (float, optional): Observer's latitude in radians. Default is 33.356 degrees.
        lon (float, optional): Observer's longitude in radians. Default is -116.863 degrees.

    Returns:
        float: Airmass of the object, calculated as 1 divided by the sine of the object's altitude.
    """

    pi = 3.141592653589793

    # Calculate the Altitude (alt) and other parameters using the radectoaltaz function.
    az, alt, st, H = radectoaltaz(ra, dec, lat, lon, obs_date)

    # Calculate the zenith distance in radians.
    zenith_rad = pi / 2 - alt

    # Calculate and return the airmass.
    airmass = 1 / np.sin(alt)

    return airmass


def binplot(x, y, nbins=10, robust=False, data=True,
            scale=True, bins=None, weights=None, ls='none',
            dotkeys={'color': 'k'}, xerr=True, rms=False, **keys):
    """ Bin the y data into n bins of x and plot the average and
    dispersion of each bins.

    Arguments:
    ----------
    nbins: int
      Number of bins

    robust: bool
      If True, use median and nmad as estimators of the bin average
      and bin dispersion.

    data: bool
      If True, add data points on the plot

    scale: bool
      Whether the error bars should present the error on the mean or
      the dispersion in the bin

    bins: list
      The bin definition

    weights: array(len(x))
      If not None, use weights in the computation of the mean.
      Provide 1/sigma**2 for optimal weighting with Gaussian noise

    dotkeys: dict
      To keys to pass to plot when drawing data points

    **keys:
      The keys to pass to plot when drawing bins

    Exemples:
    ---------
    >>> x = np.arange(1000); y = np.random.rand(1000);
    >>> binplot(x,y)
    """
    ind = ~np.isnan(x) & ~np.isnan(y)
    x = x[ind]
    y = y[ind]
    if weights is not None:
        weights = weights[ind]
    if bins is None:
        bins = np.linspace(x.min(), x.max() + abs(x.max() * 1e-7), nbins + 1)
    ind = (x < bins.max()) & (x >= bins.min())
    x = x[ind]
    y = y[ind]
    if weights is not None:
        weights = weights[ind]
    yd = np.digitize(x, bins)
    index = make_index(yd)
    ybinned = [y[e] for e in index]
    xbinned = 0.5 * (bins[:-1] + bins[1:])
    usedbins = np.array(np.sort(list(set(yd)))) - 1
    xbinned = xbinned[usedbins]
    bins = bins[usedbins + 1]
    if data and not 'noplot' in keys:
        plt.plot(x, y, ',', **dotkeys)

    if robust is True:
        yplot = [np.median(e) for e in ybinned]
        yerr = np.array([mad(e) for e in ybinned])
    elif robust:
        yres = [robust_average(e, sigma=None, clip=robust, mad=False, axis=0)
                for e in ybinned]
        yplot = [e[0] for e in yres]
        yerr = [np.sqrt(e[3]) for e in yres]
    elif weights is not None:
        wbinned = [weights[e] for e in index]
        yplot = [np.average(e, weights=w) for e, w in zip(ybinned, wbinned)]
        if not scale:
            #yerr = np.array([np.std((e - a) * np.sqrt(w))
            #                 for e, w, a in zip(ybinned, wbinned, yplot)])
            yerr = np.array([np.sqrt(np.std((e - a) * np.sqrt(w)) ** 2 / sum(w))
                             for e, w, a in zip(ybinned, wbinned, yplot)])
        else:
            yerr = np.array([np.sqrt(1 / sum(w))
                             for e, w, a in zip(ybinned, wbinned, yplot)])
        scale = False
    else:
        yplot = [np.mean(e) for e in ybinned]
        yerr = np.array([np.std(e) for e in ybinned])
        if rms:
            yerr = [np.sqrt(np.mean(e**2)) for e in ybinned]

    if scale:
        yerr /= np.sqrt(np.bincount(yd)[usedbins + 1])

    if xerr:
        xerr = np.array([bins, bins]) - np.array([xbinned, xbinned])
    else:
        xerr = None
    if not 'noplot' in keys:
        plt.errorbar(xbinned, yplot, yerr=yerr,
                     xerr=xerr,
                     ls=ls, **keys)
    return xbinned, yplot, yerr


class SuperpixelizedZTFFocalPlane:
    def __init__(self, resolution):
        self.__resolution = resolution
        self.xbins = np.linspace(0., quadrant_width_px, self.__resolution+1)
        self.ybins = np.linspace(0., quadrant_height_px, self.__resolution+1)

    @property
    def resolution(self):
        return self.__resolution

    def superpixelize(self, x, y, ccdid, qid):
        ix = np.digitize(x, self.xbins, right=True) - 1
        iy = np.digitize(y, self.ybins, right=True) - 1

        return self.__resolution**2*(4*(ccdid-1)+qid)+iy*self.__resolution+ix

    @property
    def vecsize(self):
        return 64*self.__resolution**2

    def vecrange(self, ccdid, qid, vec_map=None):
        if vec_map:
            if qid in [0, 1, 2]:
                return slice(vec_map[self.__resolution**2*(4*(ccdid-1)+qid)], vec_map[self.__resolution**2*(4*(ccdid-1)+qid+1)])
            else:
                return slice(vec_map[self.__resolution**2*(4*(ccdid-1)+qid)], vec_map[self.__resolution**2*4*ccdid])
        else:
            if qid in [0, 1, 2]:
                return slice(self.__resolution**2*(4*(ccdid-1)+qid), self.__resolution**2*(4*(ccdid-1)+qid+1))
            else:
                return slice(self.__resolution**2*(4*(ccdid-1)+qid), self.__resolution**2*4*ccdid)

    def plot(self, fig, vec, vec_map=None, cmap=None, vlim=None, f=None, cbar_label=None, mask=None):
        if isinstance(vec_map, dict):
            _vec = np.full([64*self.__resolution**2], np.nan)
            np.put_along_axis(_vec, np.array(list(vec_map.keys())), vec, 0)
            vec = _vec
        elif hasattr(vec_map, '__iter__'):
            vec = np.where(vec_map==-1, np.nan, vec[vec_map])

        focal_plane_dict = dict([(ccdid, dict([(qid, vec[self.vecrange(ccdid, qid)]) for qid in range(4)])) for ccdid in range(1, 17)])

        if f is not None:
            for ccdid in focal_plane_dict.keys():
                for qid in focal_plane_dict[ccdid].keys():
                    focal_plane_dict[ccdid][qid] = focal_plane_dict[ccdid][qid]-f(focal_plane_dict[ccdid][qid])

        if vlim is None or isinstance(vlim, str):
            values = np.array([[focal_plane_dict[ccdid][qid] for qid in range(4)] for ccdid in range(1, 17)]).flatten()
            values = values[~np.isnan(values)]

            if vlim is None:
                vmin, vmax = np.min(values), np.max(values)
            elif vlim == 'mad':
                vmax = 5.*median_abs_deviation(values)
                vmin = -5.*median_abs_deviation(values)
            elif vlim == 'mad_positive':
                vmax = 10*median_abs_deviation(values)
                vmin = 0.
            elif vlim == 'sigma_clipping':
                m, s = np.mean(values), np.std(values)
                vmax = m+s
                vmin = m-s

        else:
            vmin, vmax = vlim

        def _plot(ax, val, ccdid, qid, rcid):
            if val is not None:
                ax.imshow(np.flip(np.flip(val.reshape(self.__resolution, self.__resolution), axis=1), axis=0), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set(xticks=[], yticks=[])
                ax.set_aspect('auto')

                ax.text(0.5, 0.5, rcid+1, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        plt.axis('off')
        plt.axis('equal')
        plot_ztf_focal_plane(fig, focal_plane_dict, _plot, True)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(ScalarMappable(Normalize(vmin=vmin, vmax=vmax), cmap=cmap), cax=cbar_ax, label=cbar_label)

    def vec_to_block(self, vec, vec_map=None, f=None):
        if isinstance(vec_map, dict):
            _vec = np.full([64*self.__resolution**2], np.nan)
            np.put_along_axis(_vec, np.array(list(vec_map.keys())), vec, 0)
            vec = _vec
        elif hasattr(vec_map, '__iter__'):
            vec = np.where(vec_map==-1, np.nan, vec[vec_map])

        if f is None:
            f = lambda x: x
        d = {}
        for ccdid in range(1, 17):
            ccd = np.hstack([np.vstack([np.flip(np.flip(f(vec[self.vecrange(ccdid, 2)]).reshape(self.__resolution, self.__resolution), axis=1), axis=0),
                                        np.flip(np.flip(f(vec[self.vecrange(ccdid, 1)]).reshape(self.__resolution, self.__resolution), axis=1), axis=0)]),
                             np.vstack([np.flip(np.flip(f(vec[self.vecrange(ccdid, 3)]).reshape(self.__resolution, self.__resolution), axis=1), axis=0),
                                        np.flip(np.flip(f(vec[self.vecrange(ccdid, 0)]).reshape(self.__resolution, self.__resolution), axis=1), axis=0)])])
            d[ccdid] = ccd

        plane = np.vstack([np.hstack([d[4], d[3], d[2], d[1]]),
                           np.hstack([d[8], d[7], d[6], d[5]]),
                           np.hstack([d[12], d[11], d[10], d[9]]),
                           np.hstack([d[16], d[15], d[14], d[13]])])

        return plane

def plot_ztf_focal_plane(fig, focal_plane_dict, plot_fun, plot_ccdid=False):
    ccds = fig.add_gridspec(4, 4, wspace=0.02, hspace=0.02)
    for i in range(4):
        for j in range(4):
            ccdid = 16 - (i*4+j)
            quadrants = ccds[i, j].subgridspec(2, 2, wspace=0., hspace=0.)
            axs = quadrants.subplots()

            for k in range(2):
                for l in range(2):
                    rcid = (ccdid-1)*4 + k*2
                    qid = k*2
                    if k > 0:
                        rcid += l
                        qid += l
                    else:
                        rcid -= (l - 1)
                        qid -= (l - 1)

                    plot_fun(axs[k, l], focal_plane_dict[ccdid][qid], ccdid, qid, rcid)

            if plot_ccdid:
                ax = fig.add_subplot(ccds[i, j])
                ax.text(0.5, 0.5, ccdid, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontweight='black', fontsize='xx-large')
                ax.axis('off')

    for ax in fig.get_axes():
        ss = ax.get_subplotspec()
        ax.spines.top.set_visible(ss.is_first_row())
        ax.spines.top.set(linewidth=1.)
        ax.spines.bottom.set_visible(ss.is_last_row())
        ax.spines.bottom.set(linewidth=1.)
        ax.spines.left.set_visible(ss.is_first_col())
        ax.spines.left.set(linewidth=1.)
        ax.spines.right.set_visible(ss.is_last_col())
        ax.spines.right.set(linewidth=1.)


def make_index_from_array(array):
    s = np.array(list(set(array)))
    m = dict(list(zip(s, list(range(len(s))))))
    indices = np.fromiter((m[e] for e in array), 'int')
    return m, indices


def sanitize_data(df, photo_key):
    measure_count = len(df)
    print("Removing negative flux")
    df = df.loc[df[photo_key]>0.]
    print("Removed {} negative measures".format(measure_count-len(df)))
    print("Measure count={}".format(len(df)))
    print("")

    measure_count = len(df)
    print("Removing out of bound measures")
    df = df.loc[df['x']>0.]
    df = df.loc[df['x']<=quadrant_width_px]
    df = df.loc[df['y']>0.]
    df = df.loc[df['y']<=quadrant_height_px]
    print("Removed {} out of bound measures".format(measure_count-len(df)))
    print("Measure count={}".format(len(df)))
    print("")

    return df
