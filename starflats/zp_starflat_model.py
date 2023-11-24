#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import models
import simple_starflat_model
from utils import SuperpixelizedZTFFocalPlan, plot_ztf_focal_plane, quadrant_width_px, quadrant_height_px, get_airmass
from linearmodels import indic


class ZPStarflatModel(simple_starflat_model.SimpleStarflatModel):
    def __init__(self, config_path, dataset_path):
        super().__init__(config_path, dataset_path)

    def build_model(self):
        model = indic(self.dp.gaiaid_index, name='m') + indic(self.dp.dzp_index, name='dzp') + indic(self.dp.mjd_index, name='zp')
        model.params['dzp'].fix(0, 0.)
        model.params['zp'].fix(0, 0.)
        return model

    @staticmethod
    def model_desc():
        return "ZP starflat model. Fit star magnitude, superpixelized ZP and ZP wrt mjd."

    @staticmethod
    def model_math():
        return "$m_\mathrm{ADU}=m_s+\delta ZP(u, v) + ZP(mjd)$"

    def plot(self, output_path):
        super().plot(output_path)

        chi2_ndof = np.sum(self.wres[~self.bads]**2)/self.ndof

        plt.subplots(figsize=(8., 5.))
        plt.suptitle("ZP term variation with exposure")
        plt.plot(self.dp.mjd_map.keys(), self.fitted_params['zp'].full, '.')
        plt.grid()
        plt.xlabel("MJD")
        plt.ylabel("$\mathrm{ZP}^q$ [mag]")
        plt.tight_layout()
        plt.savefig(output_path.joinpath("zp.png"), dpi=300.)
        plt.close()

models.register_model('zp', ZPStarflatModel)
