#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import models
import zp_starflat_model
from utils import SuperpixelizedZTFFocalPlan, plot_ztf_focal_plane, quadrant_width_px, quadrant_height_px, get_airmass
from linearmodels import indic


class ColorStarflatModel(zp_starflat_model.ZPStarflatModel):
    def __init__(self, config_path, dataset_path):
        super().__init__(config_path, dataset_path)

        self.color_superpixels = SuperpixelizedZTFFocalPlan(self.config['color_resolution'])
        self.dp.add_field('dk', self.color_superpixels.superpixelize(self.dp.x, self.dp.y, self.dp.ccdid, self.dp.qid))

        self.dp.make_index('dk')

    def build_model(self):
        model = indic(self.dp.gaiaid_index, name='m') + indic(self.dp.dzp_index, name='dzp') + indic(self.dp.mjd_index, name='zp') + indic(self.dp.dk_index, val=self.dp.col, name='dk')
        # model.params['dzp'].fix(0, 0.)
        # model.params['zp'].fix(0, 0.)
        # model.params['dk'].fix(0, 0.)
        return model

    @staticmethod
    def model_desc():
        return "Color starflat model. Fit star magnitude and superpixelized ZP, centered color term over the focal plane, ZP wrt mjd."

    @staticmethod
    def model_math():
        return "$m_\mathrm{ADU}^{s,q}=m^s+\delta ZP(u, v) + ZP^q + \delta k(u, v) C_{Bp-Rp}^s$"

    def fix_params(self, model):
        super().fix_params(model)
        model.params['dk'].fix(0, 0.)

    def eq_constraints(self, model, mu=0.1):
        constraints = super().eq_constraints(model, mu)
        constraints.append([model.params['dk'].indexof(), mu])

        return constraints

    def plot(self, output_path):
        super().plot(output_path)

        chi2_ndof = np.sum(self.wres[~self.bads]**2)/self.ndof

        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("$\delta k(u, v) - {}$\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
        self.color_superpixels.plot(fig, self.fitted_params['dk'].full, cmap='viridis', vlim='mad', cbar_label="$\delta k$")
        plt.savefig(output_path.joinpath("dk.png"), dpi=300.)
        plt.close()

    def _dump_recap(self):
        d = super()._dump_recap()

        d['color_resolution'] = self.config['color_resolution']
        return d

models.register_model('color', ColorStarflatModel)
