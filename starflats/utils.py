#!/usr/bin/env python3

import numpy as np

from astropy.time import Time

ztffiltercodes = ['zg', 'zr', 'zi']
gaiarefmjd = Time(2015.5, format='byear').mjd
quadrant_width_px, quadrant_height_px = 3072, 3080
quadrant_size_px = {'x': quadrant_width_px, 'y': quadrant_height_px}


def superpixelize(x, y, ccdid, qid, res):
    xbins = np.linspace(0., quadrant_width_px, res+1)
    ybins = np.linspace(0., quadrant_height_px, res+1)

    ix = np.digitize(x, xbins) - 1
    iy = np.digitize(y, ybins) - 1
    idx = res*res*4*(ccdid-1)+4*qid+iy*res+ix
    return idx, res*res*64
