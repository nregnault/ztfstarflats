#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import models
from utils import SuperpixelizedZTFFocalPlan, plot_ztf_focal_plane, quadrant_width_px, quadrant_height_px
from linearmodels import indic

class SimpleStarflatModel(models.StarflatModel):
    def __init__(self, config_path, dataset_path):
        super().__init__(config_path, dataset_path)

        self.superpixels = SuperpixelizedZTFFocalPlan(self.config['zp_resolution'])
        self.dp.add_field('zp', self.superpixels.superpixelize(self.dp.x, self.dp.y, self.dp.ccdid, self.dp.qid))
        self.dp.make_index('gaiaid')
        self.dp.make_index('zp')

    def build_model(self):
        model = indic(self.dp.gaiaid_index, name='m') + indic(self.dp.zp_index, name='zp')
        model.params['zp'].fix(0, 0.)
        return model

    @staticmethod
    def model_desc():
        return "Simplest starflat model. Fit star magnitude and superpixelized ZP over the focal plane."

    @staticmethod
    def model_math():
        return "$m_\mathrm{ADU}=m_s+\delta ZP(u, v)$"

    def plot(self, output_path):
        super().plot(output_path)

        # ccdid, qid = 2, 0
        # m = np.all([self.dp.ccdid==ccdid, self.dp.qid==qid], axis=0)
        # img = self.fitted_params['zp'].full[self.superpixelized_zps.vecrange(ccdid, qid)].reshape(self.superpixelized_zps.resolution, self.superpixelized_zps.resolution)
        # zps = self.fitted_params['zp'].full[self.dp.zp_index[m]]
        # plt.imshow(img, extent=[0., quadrant_width_px, 0., quadrant_height_px], origin='lower')
        # plt.scatter(self.dp.x[m], self.dp.y[m], c=zps)
        # plt.show()
        # return

        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("$\delta ZP(u, v)$")
        self.superpixels.plot(fig, self.fitted_params['zp'].full, cmap='viridis', f=np.median, vlim='mad', cbar_label="$\delta ZP$ [mag]")
        plt.show()

        fig, axs = plt.subplots(figsize=(12., 12.))
        plt.suptitle("Measure count per superpixel")
        self.superpixels.plot(fig, np.bincount(self.dp.zp_index), cbar_label="Measure count")
        plt.show()


models.register_model('simple', SimpleStarflatModel)
