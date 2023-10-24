#!/usr/bin/env python3

import numpy as np
from astropy.time import Time
from scipy.stats import median_abs_deviation

ztffiltercodes = ['zg', 'zr', 'zi']
gaiarefmjd = Time(2015.5, format='byear').mjd
quadrant_width_px, quadrant_height_px = 3072, 3080
quadrant_size_px = {'x': quadrant_width_px, 'y': quadrant_height_px}


class SuperpixelizedZTFFocalPlan:
    def __init__(self, resolution):
        self.__resolution = resolution

    def resolution(self):
        return self.__resolution

    def superpixelize(self, x, y, ccdid, qid):
        xbins = np.linspace(0., quadrant_width_px, self.__resolution+1)
        ybins = np.linspace(0., quadrant_height_px, self.__resolution+1)

        ix = np.digitize(x, xbins) - 1
        iy = np.digitize(y, ybins) - 1

        return self.__resolution**2*(4*(ccdid-1)+qid)+iy*self.__resolution+ix

    def vecsize(self):
        return 64*self.__resolution**2

    def vecrange(self, ccdid, qid):
        if qid in [0, 1, 2]:
            return slice(self.__resolution**2*(4*(ccdid-1)+qid), self.__resolution**2*(4*(ccdid-1)+qid+1))
        else:
            return slice(self.__resolution**2*(4*(ccdid-1)+qid), self.__resolution**2*4*ccdid)

    def plot(self, fig, vec):
        focal_plane_dict = dict([(ccdid+1, dict([(qid, self.vecrange(ccdid+1, qid)) for qid in range(4)])) for ccdid in range(16)])

        def _plot(ax, val, ccdid, qid, rcid):
            if val is not None:
                zps = vec[self.vecrange(ccdid, qid)].reshape(self.__resolution, self.__resolution)
                zps = zps - np.median(zps)
                vmax = 1.*median_abs_deviation(vec)
                vmin = -1.*median_abs_deviation(vec)
                ax.imshow(zps, vmin=vmin, vmax=vmax)
                ax.set(xticks=[], yticks=[])
                ax.set_aspect('auto')

                ax.text(0.5, 0.5, rcid+1, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        plot_ztf_focal_plane(fig, focal_plane_dict, _plot, True)


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
