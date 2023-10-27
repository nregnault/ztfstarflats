#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import models
from utils import SuperpixelizedZTFFocalPlan, plot_ztf_focal_plane, quadrant_width_px, quadrant_height_px, get_airmass
from linearmodels import indic

class FullStarflatModel(models.StarflatModel):
    def __init__(self, config_path, dataset_path):
        super().__init__(config_path, dataset_path)

        self.superpixels = SuperpixelizedZTFFocalPlan(self.config['zp_resolution'])
        self.dp.add_field('dzp', self.superpixels.superpixelize(self.dp.x, self.dp.y, self.dp.ccdid, self.dp.qid))
        self.dp.add_field('dk', self.superpixels.superpixelize(self.dp.x, self.dp.y, self.dp.ccdid, self.dp.qid))
        self.dp.add_field('X', get_airmass(np.deg2rad(self.dp.ra), np.deg2rad(self.dp.dec), Time(self.dp.mjd, format='mjd').jd))

        self.dp.make_index('gaiaid')
        self.dp.make_index('dk')
        self.dp.make_index('dzp')
        self.dp.make_index('mjd')

    def build_model(self):
        model = indic(self.dp.gaiaid_index, name='m') + indic(self.dp.dzp_index, name='dzp') + indic(self.dp.mjd_index, name='zp') + indic(self.dp.dk_index, val=self.dp.BP-self.dp.RP, name='dk') + indic([0]*len(self.dp.nt), val=(self.dp.X-1), name='k')
        model.params['dzp'].fix(0, 0.)
        model.params['zp'].fix(0, 0.)
        model.params['dk'].fix(0, 0.)
        return model

    @staticmethod
    def model_desc():
        return "Full starflat model. Fit star magnitude and superpixelized ZP, color term over the focal plane, ZP wrt mjd and differential airmass term."

    @staticmethod
    def model_math():
        return "$m_\mathrm{ADU}=m_s+\delta ZP(u, v) + ZP(mjd) + \delta k C_{Bp-Rp} + k(X(u, v)-1)$"

    def plot(self, output_path):
        super().plot(output_path)

        plt.subplots(figsize=(10., 5.))
        plt.suptitle("Residuals wrt airmass")
        m = self.dp.G[~self.bads] < 13
        plt.plot(self.dp.X[~self.bads][m]-1., self.res[~self.bads][m], ',')
        plt.xlabel("$X(\\alpha,\\delta)-1$")
        plt.ylabel("$y_\mathrm{ADU}-y_\mathrm{model}$ [mag]")
        plt.grid()
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("$\delta ZP(u, v)$")
        self.superpixels.plot(fig, self.fitted_params['dzp'].full, cmap='viridis', f=np.median, vlim='mad', cbar_label="$\delta ZP$ [mag]")
        plt.show()

        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("$\delta k(u, v)$")
        self.superpixels.plot(fig, self.fitted_params['dk'].full, cmap='viridis', f=np.median, vlim='mad', cbar_label="$\delta k$")
        plt.show()

        fig, axs = plt.subplots(figsize=(12., 12.))
        plt.suptitle("Measure count per superpixel")
        self.superpixels.plot(fig, np.bincount(self.dp.dzp_index), cbar_label="Measure count")
        plt.show()


models.register_model('full', FullStarflatModel)
