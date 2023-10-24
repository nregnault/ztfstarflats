#!/usr/bin/env python3

import models
from utils import SuperpixelizedZTFFocalPlan, plot_ztf_focal_plane
from linearmodels import indic

class SimpleStarflatModel(models.StarflatModel):
    def __init__(self, config_path, dataset_path):
        super().__init__(config_path, dataset_path)

        self.superpixelized_zps = SuperpixelizedZTFFocalPlan(self.config['zp_resolution'])
        self.dp.add_field('zp', self.superpixelized_zps.superpixelize(self.dp.x, self.dp.y, self.dp.ccdid, self.dp.qid))
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


models.register_model('simple', SimpleStarflatModel)
