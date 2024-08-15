#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pathlib

import models
from utils import SuperpixelizedZTFFocalPlan, plot_ztf_focal_plane, quadrant_width_px, quadrant_height_px
from linearmodels import indic

class SimpleStarflatModel(models.StarflatModel):
    def __init__(self, config=None, mask=None):
        super().__init__(config=config, mask=mask)
        self.superpixels = SuperpixelizedZTFFocalPlan(self.config['zp_resolution'])

    def load_data(self, dataset_path):
        super().load_data(dataset_path)

        self.dp.add_field('dzp', self.superpixels.superpixelize(self.dp.x, self.dp.y, self.dp.ccdid, self.dp.qid))
        self.dp.make_index('dzp')

        self.dzp_to_index = -np.ones(64*self.superpixels.resolution**2, dtype=int)
        np.put_along_axis(self.dzp_to_index, np.array(list(self.dp.dzp_map.keys())), np.array(list(self.dp.dzp_map.values())), axis=0)

    def build_model(self):
        model = indic(self.dp.gaiaid_index, name='m') + indic(self.dp.dzp_index, name='dzp')
        return model

    @staticmethod
    def model_desc():
        return "Simplest starflat model. Fit star magnitude and superpixelized ZP over the focal plane."

    @staticmethod
    def model_math():
        return "$m_\mathrm{ADU}=m_s+\delta ZP(u, v)$"

    @staticmethod
    def model_name():
        return 'simple'

    def fix_params(self, model):
        # model.params['dzp'].fix(0, 0.)
        model.params['dzp'].fix(self.superpixels.vecrange(7, 0).start, 0.)

    def eq_constraints(self, model, mu=0.1):
        return [[model.params['dzp'].indexof(), mu]]

    def _dump_recap(self):
        d = super()._dump_recap()
        d['zp_resolution'] = self.config['zp_resolution']

        return d

    def plot(self, output_path):
        super().plot(output_path)

        chi2_ndof = np.sum(self.wres[~self.bads]**2)/self.ndof

        fig, axs = plt.subplots(figsize=(12., 12.))
        plt.suptitle("Measure count per superpixel")

        self.superpixels.plot(fig, np.bincount(self.dp.dzp), cbar_label="Measure count")
        plt.savefig(output_path.joinpath("superpixel_count.png"), dpi=300.)
        plt.close()

        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("$\delta ZP(u, v)$ - {}\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
        self.superpixels.plot(fig, self.fitted_params['dzp'].full, vec_map=self.dp.dzp_map, cmap='viridis', f=np.median, vlim='mad', cbar_label="$\delta ZP$ [mag]")
        plt.savefig(output_path.joinpath("dzp.png"))
        plt.close()

        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("$\delta ZP(u, v)$ without gain substraction - {}\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
        self.superpixels.plot(fig, self.fitted_params['dzp'].full, vec_map=self.dp.dzp_map, cmap='viridis', cbar_label="$\delta ZP$ [mag]", vlim='sigma_clipping')
        plt.savefig(output_path.joinpath("dzp_gain.png"))
        plt.close()

        wres_dzp = np.bincount(self.dp.dzp_index, weights=self.wres**2)/np.bincount(self.dp.dzp_index)

        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("Stacked $\chi^2$ - {}\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
        self.superpixels.plot(fig, wres_dzp, vec_map=self.dp.dzp_map, vlim='sigma_clipping')
        plt.savefig(output_path.joinpath("chi2_superpixels.png"))
        plt.close()

        # for mjd in self.dp.mjd_map.keys():
        #     mjd_index = self.dp.mjd_map[mjd]
        #     mask = (self.dp.mjd_index == mjd_index)
        #     wres_dzp = np.bincount(self.dp.dzp_index[mask], weights=self.wres[mask]**2, minlength=self.superpixels.vecsize)/np.bincount(self.dp.dzp_index[mask], minlength=self.superpixels.vecsize)

        #     fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        #     plt.suptitle("Stacked $\chi^2$ - {} - MJD={}\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, mjd, self.model_math(), chi2_ndof))
        #     self.superpixels.plot(fig, wres_dzp, vlim='mad_positive')
        #     plt.show()
        #     # plt.savefig(output_path.joinpath("chi2_superpixels.png"))
        #     plt.close()

    def apply_model(self, x, y, ccdid, qid, mag, **kwords):
        zp_index = self.dzp_to_index[self.superpixels.superpixelize(x, y, ccdid, qid)]
        return mag - self.fitted_params['dzp'].full[zp_index]



models.register_model(SimpleStarflatModel)
