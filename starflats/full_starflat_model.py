#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import models
from color_starflat_model import ColorStarflatModel
from utils import SuperpixelizedZTFFocalPlan, plot_ztf_focal_plane, quadrant_width_px, quadrant_height_px, get_airmass
from linearmodels import indic

class FullStarflatModel(ColorStarflatModel):
    def __init__(self, config_path, dataset_path):
        super().__init__(config_path, dataset_path)

        self.dp.add_field('X', get_airmass(np.deg2rad(self.dp.ra), np.deg2rad(self.dp.dec), Time(self.dp.mjd, format='mjd').jd))

    def build_model(self):
        model = indic(self.dp.gaiaid_index, name='m') + indic(self.dp.dzp_index, name='dzp') + indic(self.dp.mjd_index, name='zp') + indic(self.dp.dk_index, val=self.dp.col, name='dk') + indic([0]*len(self.dp.nt), val=(self.dp.X-1), name='k')
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

        chi2_ndof = np.sum(self.wres[~self.bads]**2)/self.ndof

        # plt.subplots(figsize=(10., 5.))
        # plt.suptitle("Residuals wrt airmass")
        # m = self.dp.G[~self.bads] < 13
        # plt.plot(self.dp.X[~self.bads][m]-1., self.res[~self.bads][m], ',')
        # plt.xlabel("$X(\\alpha,\\delta)-1$")
        # plt.ylabel("$y_\mathrm{ADU}-y_\mathrm{model}$ [mag]")
        # plt.grid()
        # plt.tight_layout()
        # plt.savefig(output_path.joinpath("res_airmass.png"), dpi=300.)
        # plt.close()

        # fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        # plt.suptitle("$\delta ZP(u, v) - {}$\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
        # self.superpixels.plot(fig, self.fitted_params['dzp'].full, cmap='viridis', f=np.median, vlim='mad', cbar_label="$\delta ZP$ [mag]")
        # plt.savefig(output_path.joinpath("dzp.png"), dpi=300.)
        # plt.close()

        # fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        # plt.suptitle("$\delta k(u, v) - {}$\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
        # #self.superpixels.plot(fig, self.fitted_params['dk'].full, cmap='viridis', f=np.median, vlim='mad', cbar_label="$\delta k$")
        # self.superpixels.plot(fig, self.fitted_params['dk'].full, cmap='viridis', vlim='mad', cbar_label="$\delta k$")
        # plt.savefig(output_path.joinpath("dk.png"), dpi=300.)
        # plt.close()

        # fig, axs = plt.subplots(figsize=(12., 12.))
        # plt.suptitle("Measure count per superpixel")
        # self.superpixels.plot(fig, np.bincount(self.dp.dzp_index), cbar_label="Measure count")
        # plt.savefig(output_path.joinpath("superpixel_count.png"), dpi=300.)
        # plt.close()


models.register_model('full', FullStarflatModel)
