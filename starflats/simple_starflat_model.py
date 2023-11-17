#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import models
from utils import SuperpixelizedZTFFocalPlan, plot_ztf_focal_plane, quadrant_width_px, quadrant_height_px
from linearmodels import indic

class SimpleStarflatModel(models.StarflatModel):
    def __init__(self, config_path, dataset_path):
        super().__init__(config_path, dataset_path)

        self.config['zp_resolution'] = 25
        self.superpixels = SuperpixelizedZTFFocalPlan(self.config['zp_resolution'])
        self.dp.add_field('dzp', self.superpixels.superpixelize(self.dp.x, self.dp.y, self.dp.ccdid, self.dp.qid))
        self.dp.make_index('gaiaid')
        self.dp.make_index('dzp')
        self.dp.make_index('qid')
        self.dp.make_index('ccdid')

        bc = np.bincount(self.dp.dzp_index)

    def build_model(self):
        model = indic(self.dp.gaiaid_index, name='m') + indic(self.dp.dzp_index, name='dzp')
        model.params['dzp'].fix(0, 0.)
        return model

    @staticmethod
    def model_desc():
        return "Simplest starflat model. Fit star magnitude and superpixelized ZP over the focal plane."

    @staticmethod
    def model_math():
        return "$m_\mathrm{ADU}=m_s+\delta ZP(u, v)$"

    def plot(self, output_path):
        super().plot(output_path)

        chi2_ndof = np.sum(self.wres[~self.bads]**2)/self.ndof

        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("$\delta ZP(u, v) - {}$\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
        self.superpixels.plot(fig, self.fitted_params['dzp'].full, cmap='viridis', f=np.median, vlim='mad', cbar_label="$\delta ZP$ [mag]")
        plt.savefig(output_path.joinpath("dzp.png"))
        plt.close()

        fig, axs = plt.subplots(figsize=(12., 12.))
        plt.suptitle("Measure count per superpixel")
        self.superpixels.plot(fig, np.bincount(self.dp.dzp_index), cbar_label="Measure count")
        plt.savefig(output_path.joinpath("superpixel_count.png"), dpi=300.)
        plt.close()


models.register_model('simple', SimpleStarflatModel)
