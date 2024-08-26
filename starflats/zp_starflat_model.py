#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import models
import simple_starflat_model
from linearmodels import indic


class ZPStarflatModel(simple_starflat_model.SimpleStarflatModel):
    def __init__(self, config, mask):
        super().__init__(config, mask)

    def load_data(self, dataset_path):
        super().load_data(dataset_path)

    def _build_model(self):
        return super()._build_model() + [indic(self.dp.mjd_index, name='zp')]

    @staticmethod
    def model_desc():
        return "ZP starflat model. Fit star magnitude, superpixelized ZP and ZP wrt mjd."

    @staticmethod
    def model_math():
        return "$m_\mathrm{ADU}=m_s+\delta ZP(u, v) + ZP(mjd)$"

    @staticmethod
    def model_name():
        return 'zp'

    def fix_params(self, model):
        super().fix_params(model)
        if self.dp.nt.dtype.names:
            for sequenceid in self.dp.sequenceid_set:
                zp_idx = self.dp.mjd_index[self.dp.sequenceid==sequenceid][0]
                model.params['zp'].fix(zp_idx, 0.)
        else:
            model.params['zp'].fix(0, 0.)

    def eq_constraints(self, model, mu=0.1):
        constraints = super().eq_constraints(model)
        constraints.append([model.params['zp'].indexof(), mu])

        return constraints

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

    def _dump_recap(self):
        d = super()._dump_recap()
        d.update({'zp_count': len(self.dp.mjd_set)})
        return d

    def _dump_result(self):
        return super()._dump_result()

    # def apply_model(self, x, y, ccdid, qid, mag, **kwords):
    #     # Measurements are supposed to be already aligned
    #     return super().apply_model(x, y, ccdid, qid, mag, **kwords)


models.register_model(ZPStarflatModel)
