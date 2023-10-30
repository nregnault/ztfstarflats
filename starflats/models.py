#!/usr/bin/env python3

import pickle

from yaml import load, Loader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from linearmodels import RobustLinearSolver
from dataproxy import DataProxy
from utils import binplot

photometry_choices = ['psf']

photometry_choice_to_key = {'psf': 'psfflux'}
photometry_error_choice_to_key = {'psf': 'epsfflux'}


class StarflatModel:
    def __init__(self, config_path, dataset_path):
        with open(config_path, 'r') as f:
            self.__config = load(f, Loader=Loader)

        assert (self.__config['photometry'] in photometry_choices)

        photo_key = photometry_choice_to_key[self.__config['photometry']]
        photo_err_key = photometry_error_choice_to_key[self.__config['photometry']]

        df = pd.read_parquet(dataset_path)
        df['mag'] = -2.5*np.log10(df[photo_key])
        df['emag'] = 1.08*df[photo_err_key]/df[photo_key]

        kwargs = dict([(col_name, col_name) for col_name in df.columns])
        self.dp = DataProxy(df.to_records(), **kwargs)

    @property
    def config(self):
        return self.__config

    def build_model(self):
        raise NotImplementedError

    @staticmethod
    def model_desc():
        raise NotImplementedError

    @staticmethod
    def model_math():
        raise NotImplementedError

    def plot(self, output_path):
        wres = self.res/np.sqrt(self.dp.emag**2+self.config['piedestal']**2)

        plt.subplots(figsize=(12., 5.))
        plt.suptitle("Residual plot wrt $G$ magnitude\nModel: {}".format(self.model_math()))
        plt.plot(self.dp.G[~self.bads], self.res[~self.bads], ',')
        plt.xlabel("$m_G$ [AB mag]")
        plt.ylabel("$m_\mathrm{ADU}-m_\mathrm{model}$ [mag]")
        plt.ylim(-0.75, 0.75)
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_path.joinpath("residuals_mag.png"), dpi=300.)
        plt.close()

        plt.subplots(nrows=2, ncols=2, figsize=(10., 6.), gridspec_kw={'width_ratios': [5., 1.5], 'hspace': 0., 'wspace': 0.}, sharex=False, sharey=False)
        plt.suptitle("Standardized residuals\npiedestal={}".format(self.config['piedestal']))
        plt.subplot(2, 2, 1)
        xbinned_mag, yplot_stdres, stdres_dispersion = binplot(self.dp.G[~self.bads], wres[~self.bads], data=False, scale=False, nbins=5)
        plt.plot(self.dp.G[~self.bads], wres[~self.bads], ',', color='xkcd:light blue')
        plt.ylabel("$\\frac{m_{ADU}-m_\\mathrm{model}}{\\sigma_m}$ [mag]")
        plt.xlim([np.min(self.dp.G[~self.bads]), np.max(self.dp.G[~self.bads])])
        plt.grid()
        plt.subplot(2, 2, 2)
        plt.hist(wres, bins='auto', orientation='horizontal', density=True)
        m, s = norm.fit(wres[~self.bads])
        x = np.linspace(np.min(wres[~self.bads])-0.5, np.max(wres[~self.bads])+0.5)
        plt.plot(norm.pdf(x, loc=m, scale=s), x, label="$\sim\mathcal{{N}}(\mu={:.2f}, \sigma={:.2f})$".format(m, s))
        plt.plot(norm.pdf(x, loc=0., scale=1.), x, label="$\sim\mathcal{N}(\mu=0, \sigma=1)$")
        plt.legend()
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.plot(xbinned_mag, stdres_dispersion)
        plt.xlim([np.min(self.dp.G[~self.bads]), np.max(self.dp.G[~self.bads])])
        plt.xlabel("$m_\mathrm{G}$ [AB mag]")
        plt.ylabel("$\\sigma_{\\frac{m_{ADU}-m_\\mathrm{model}}{\\sigma_m}}$ [mag]")
        plt.axhline(1.)
        plt.grid()
        plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

        plt.subplots(figsize=(12., 5.))
        plt.suptitle("Residual plot wrt $Bp-Rp$ magnitude\nModel: {}".format(self.model_math()))
        plt.plot(self.dp.BP[~self.bads] - self.dp.RP[~self.bads], self.res[~self.bads], ',')
        plt.xlabel("$m_{Bp}-m_{Rp}$ [mag]")
        plt.ylabel("$m_\mathrm{ADU}-m_\mathrm{model}$ [mag]")
        plt.ylim(-0.75, 0.75)
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_path.joinpath("residuals_color.png"), dpi=300.)
        plt.close()

        plt.subplots(figsize=(12., 5.))
        plt.suptitle("Residual plot wrt MJD\nModel: {}".format(self.model_math()))
        plt.plot(self.dp.mjd[~self.bads], self.res[~self.bads], ',')
        plt.xlabel("MJD")
        plt.ylabel("$m_\mathrm{ADU}-m_\mathrm{model}$ [mag]")
        plt.ylim(-0.75, 0.75)
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_path.joinpath("residuals_mjd.png"), dpi=300.)
        plt.close()

    def solve(self):
        model = self.build_model()
        solver = RobustLinearSolver(model, self.dp.mag, weights=1./np.sqrt(self.dp.emag**2+self.config['piedestal']**2))
        solver.model.params.free = solver.robust_solution(local_param='m')

        self.fitted_params = solver.model.params
        self.bads = solver.bads
        self.res = solver.get_res(self.dp.mag)
        # self.cov = solver.get_cov()
        self.cov = None # Getting cov matrix leads to crash

    def dump_result(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump({'fitted_params': self.fitted_params, 'bads': self.bads, 'res': self.res, 'cov': self.cov}, f)

    def load_result(self, result_path):
        with open(result_path, 'rb') as f:
            d = pickle.load(f)

        self.fitted_params = d['fitted_params']
        self.bads = d['bads']
        self.res = d['res']
        self.cov = d['cov']


Models = {}

def register_model(name, model):
    Models[name] = model
