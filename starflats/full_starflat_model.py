#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import models
from color_starflat_model import ColorStarflatModel
from utils import get_airmass
from linearmodels import indic

class FullStarflatModel(ColorStarflatModel):
    def __init__(self, config, mask):
        super().__init__(config, mask)

    def load_data(self, dataset_path):
        super().load_data(dataset_path)

        self.dp.add_field('X', get_airmass(np.deg2rad(self.dp.ra), np.deg2rad(self.dp.dec), Time(self.dp.mjd, format='mjd').jd))

    def _build_model(self):
        return super()._build_model() + [indic([0]*len(self.dp.nt), val=(self.dp.X-1), name='k')]

    @staticmethod
    def model_desc():
        return "Full starflat model. Fit star magnitude and superpixelized ZP, color term over the focal plane, ZP wrt mjd and differential airmass term."

    @staticmethod
    def model_math():
        return "$m_\mathrm{ADU}=m_s+\delta ZP(u, v) + ZP(mjd) + \delta k C_{Bp-Rp} + k(X(u, v)-1)$"

    @staticmethod
    def model_name():
        return 'full'

    def fix_params(self, model):
        super().fix_params(model)

    def eq_constraints(self, model, mu=0.1):
        return super().eq_constraints(model, mu)

    def plot(self, output_path):
        super().plot(output_path)

    def _dump_recap(self):
        d = super()._dump_recap()
        d['k'] = self.fitted_params['k'].full[:].item()
        return d

    def _dump_result(self):
        return super()._dump_result()


models.register_model(FullStarflatModel)
