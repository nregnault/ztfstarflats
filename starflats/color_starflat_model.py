#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import pathlib

import models
import zp_starflat_model
from utils import SuperpixelizedZTFFocalPlane, quadrant_width_px, quadrant_height_px, get_airmass
from linearmodels import indic


class ColorStarflatModel(zp_starflat_model.ZPStarflatModel):
    def __init__(self, config=None, mask=None):
        super().__init__(config, mask)
        self.color_superpixels = SuperpixelizedZTFFocalPlane(self.config['color_resolution'])

    def load_data(self, dataset_path):
        super().load_data(dataset_path)

        self.dp.add_field('dk', self.color_superpixels.superpixelize(self.dp.x, self.dp.y, self.dp.ccdid, self.dp.qid))
        self.dp.make_index('dk')

        self.dk_to_index = -np.ones(64*self.color_superpixels.resolution**2, dtype=int)
        np.put_along_axis(self.dk_to_index, np.array(list(self.dp.dk_map.keys())), np.array(list(self.dp.dk_map.values())), axis=0)

    def _build_model(self):
        return super()._build_model() + [indic(self.dp.dk_index, val=self.dp.col, name='dk')]

    @staticmethod
    def model_desc():
        return "Color starflat model. Fit star magnitude and superpixelized ZP, centered color term over the focal plane, ZP wrt mjd."

    @staticmethod
    def model_math():
        return "$m_\mathrm{ADU}^{s,q}=m^s+\delta ZP(u, v) + ZP^q + \delta k(u, v) C_{Bp-Rp}^s$"

    @staticmethod
    def model_name():
        return 'color'

    def fix_params(self, model):
        super().fix_params(model)
        model.params['dk'].fix(self.color_superpixels.vecrange(7, 0).start, 0.)

    def eq_constraints(self, model, mu=0.1):
        constraints = super().eq_constraints(model, mu)
        constraints.append([model.params['dk'].indexof(), mu])

        return constraints

    def plot(self, output_path):
        super().plot(output_path)

        chi2_ndof = np.sum(self.wres[~self.bads]**2)/self.ndof

        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("$\delta k(u, v) - {}$\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
        self.color_superpixels.plot(fig, self.fitted_params['dk'].full, self.dk_to_index, cmap='viridis', vlim='mad', cbar_label="$\delta k$")
        plt.savefig(output_path.joinpath("dk.png"), dpi=300.)
        plt.close()

        # Flagged measurement count per Delta k superpixel
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12))
        plt.suptitle("Outlier count per $\Delta k$ superpixel")
        self.color_superpixels.plot(fig, np.bincount(self.dp.dk_index, weights=self.bads), vec_map=self.dk_to_index, cbar_label="Outlier count")
        plt.savefig(output_path.joinpath("superpixel_dk_outlier.png"), dpi=300.)
        plt.close()

    def _dump_recap(self):
        d = super()._dump_recap()
        d.update({'dk': self.color_superpixels.vecsize})
        d.update({'dk_resolution': self.color_superpixels.resolution})
        d.update({'dk_parameter': len(self.dp.dk_set)})
        d.update({'dk_sum': np.sum(self.fitted_params['dk'].full).item()})
        return d

    def _dump_result(self):
        d = super()._dump_result()
        d['dk_to_index'] = self.dk_to_index
        return d


models.register_model(ColorStarflatModel)
