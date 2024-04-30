#!/usr/bin/env python3

import pickle
import time

from yaml import load, Loader, dump
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from ztfquery.fields import ccdid_qid_to_rcid

from linearmodels import RobustLinearSolver, LinearModel
from dataproxy import DataProxy
from utils import binplot, idx2markerstyle, make_index_from_array

photometry_choices = ['psf'] + ['apfl{}'.format(i) for i in range(10)]

photometry_choice_to_key = {'psf': 'psfflux'}
photometry_choice_to_key.update(dict([('apfl{}'.format(i), 'apfl{}'.format(i)) for i in range(10)]))
photometry_error_choice_to_key = {'psf': 'epsfflux'}
photometry_error_choice_to_key.update(dict([('apfl{}'.format(i), 'eapfl{}'.format(i)) for i in range(10)]))

class StarflatModel:
    def __init__(self, config_path, dataset_path):
        with open(config_path, 'r') as f:
            self.__config = load(f, Loader=Loader)

        photo_key = self.__config['photometry']
        photo_err_key = self.__config['photometry_error']
        photo_color_lhs = self.__config['color_lhs']
        photo_color_rhs = self.__config['color_rhs']
        photo_ext_cat = self.__config['photometry_ext_cat']

        df = pd.read_parquet(dataset_path)

        measure_count = len(df)
        df = df.loc[df[photo_key]>0.]
        print("Removed {} negative measures".format(measure_count-len(df)))
        print("Measure count={}".format(len(df)))

        df['mag'] = -2.5*np.log10(df[photo_key])
        df['emag'] = 1.08*df[photo_err_key]/df[photo_key]

        df['rcid'] = ccdid_qid_to_rcid(df['ccdid'], df['qid'])

        df['col'] = (df[photo_color_lhs] - df[photo_color_rhs]) - np.mean(df[photo_color_lhs] - df[photo_color_rhs])

        df['ext_cat_mag'] = df[photo_ext_cat]

        # Remove potential outliers
        if photo_color_lhs == 'BP' and photo_color_rhs == 'RP' and photo_ext_cat == 'G':
            measure_count = len(df)
            df = df.loc[df['G']>10.]
            df = df.loc[df['G']<20.5]
            df = df.loc[df['col']<2.5]
            df = df.loc[df['col']>-1.]
            print("Removed {} potential outliers".format(measure_count-len(df)))

        print("Removing stars that have less than {} measures...".format(5))
        gaiaid_index_map, gaiaid_index = make_index_from_array(df['gaiaid'].to_numpy())
        gaiaid_mask = (np.bincount(gaiaid_index) < 5)
        to_remove_mask = gaiaid_mask[gaiaid_index]
        df = df.loc[~to_remove_mask]
        print(" Removed {} measures.".format(sum(to_remove_mask)))

        kwargs = dict([(col_name, col_name) for col_name in df.columns])
        self.dp = DataProxy(df.to_records(), **kwargs)
        self.__dataset_name = dataset_path.stem

        self.dp.make_index('qid')
        self.dp.make_index('ccdid')
        self.dp.make_index('mjd')
        self.dp.make_index('gaiaid')

    @property
    def config(self):
        return self.__config

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def wres(self):
        return self.res/self.measure_errors

    @property
    def ndof_full(self):
        return len(self.dp.nt) - len(self.fitted_params.full)

    @property
    def ndof(self):
        return len(self.dp) - len(self.fitted_params.full) - sum(self.bads)

    @property
    def measure_errors(self):
        return np.sqrt(self.dp.emag**2+self.config['piedestal']**2)

    def build_model(self):
        raise NotImplementedError

    @staticmethod
    def model_desc():
        raise NotImplementedError

    @staticmethod
    def model_math():
        raise NotImplementedError

    @staticmethod
    def model_name():
        raise NotImplementedError

    def fix_params(self):
        raise NotImplementedError

    def eq_constraints(self):
        return NotImplementedError

    def plot(self, output_path):
        wres = self.wres

        # Plot dithering pattern
        plt.subplots(figsize=(8., 8.))
        plt.suptitle("Dithering pattern - {}\n {}".format(self.config['photometry'], self.dataset_name))
        mask = (self.dp.gaiaid_index == np.argmax(np.bincount(self.dp.gaiaid_index)))
        rcids = list(set(self.dp.rcid[mask]))

        for i, rcid in enumerate(rcids):
            rcid_mask = (self.dp.rcid[mask] == rcid)
            plt.plot(self.dp.x[mask][rcid_mask], self.dp.y[mask][rcid_mask], idx2markerstyle[i], label=rcid)

        plt.grid()
        plt.legend()
        plt.axis('equal')
        plt.xlabel("$x$ [pixel]")
        plt.ylabel("$y$ [pixel]")
        plt.savefig(output_path.joinpath("dithering.png"), dpi=300.)
        plt.close()

        plt.subplots(figsize=(12., 5.))
        plt.suptitle("Residual plot wrt ${}$ magnitude\nModel: {}".format(self.config['photometry_ext_cat'], self.model_math()))
        plt.plot(self.dp.ext_cat_mag[~self.bads], self.res[~self.bads], ',')
        plt.xlabel("$m_{}$ [mag]".format(self.config['photometry_ext_cat']))
        plt.ylabel("$m_\mathrm{ADU}-m_\mathrm{model}$ [mag]")
        plt.ylim(-0.75, 0.75)
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_path.joinpath("residuals_mag.png"), dpi=300.)
        plt.close()

        # Standardized residuals
        plt.subplots(nrows=2, ncols=2, figsize=(10., 6.), gridspec_kw={'width_ratios': [5., 1.5], 'hspace': 0., 'wspace': 0.}, sharex=False, sharey=False)
        plt.suptitle("Standardized residuals\npiedestal={}".format(self.config['piedestal']))

        plt.subplot(2, 2, 1)
        xbinned_mag, yplot_stdres, stdres_dispersion = binplot(self.dp.ext_cat_mag[~self.bads], wres[~self.bads], data=False, scale=False, nbins=5)
        plt.plot(self.dp.ext_cat_mag[~self.bads], wres[~self.bads], ',', color='xkcd:light blue')
        plt.ylabel("$\\frac{m_{ADU}-m_\\mathrm{model}}{\\sigma_m}$ [mag]")
        plt.xlim([np.min(self.dp.ext_cat_mag[~self.bads]), np.max(self.dp.ext_cat_mag[~self.bads])])
        plt.grid()

        plt.subplot(2, 2, 2)
        xmin, xmax = np.min(wres[~self.bads])-0.5, np.max(wres[~self.bads])+0.5
        plt.hist(wres, bins='auto', orientation='horizontal', density=True, range=[xmin, xmax])
        m, s = norm.fit(wres[~self.bads])
        x = np.linspace(xmin, xmax)
        plt.plot(norm.pdf(x, loc=m, scale=s), x, label="$\sim\mathcal{{N}}(\mu={:.2f}, \sigma={:.2f})$".format(m, s))
        plt.plot(norm.pdf(x, loc=0., scale=1.), x, label="$\sim\mathcal{N}(\mu=0, \sigma=1)$")
        plt.legend()
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.plot(xbinned_mag, stdres_dispersion)
        plt.xlim([np.min(self.dp.ext_cat_mag[~self.bads]), np.max(self.dp.ext_cat_mag[~self.bads])])
        plt.xlabel("$m_\mathrm{{{}}}$ [AB mag]".format(self.config['photometry_ext_cat']))
        plt.ylabel("$\\sigma_{\\frac{m_{ADU}-m_\\mathrm{model}}{\\sigma_m}}$ [mag]")
        plt.axhline(1.)
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path.joinpath("pull.png"))
        plt.close()

        plt.subplots(figsize=(12., 5.))
        plt.suptitle("Residual plot wrt ${}-{}$ magnitude\nModel: {}".format(self.config['color_lhs'], self.config['color_rhs'], self.model_math()))
        plt.plot(self.dp.col[~self.bads], self.res[~self.bads], ',')
        plt.xlabel("$m_{{{}}}-m_{{{}}}$ [mag]".format(self.config['color_lhs'], self.config['color_rhs']))
        plt.ylabel("$m_\mathrm{ADU}-m_\mathrm{model}$ [mag]")
        plt.ylim(-0.75, 0.75)
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_path.joinpath("residuals_color.png"), dpi=300.)
        plt.close()


        # Chi2 par exposure
        chi2_day = np.bincount(self.dp.mjd_index[~self.bads], weights=self.wres[~self.bads]**2)/np.bincount(self.dp.mjd_index[~self.bads])
        plt.subplots(figsize=(12., 5.))
        plt.suptitle("$\chi^2/day$ \nModel: {}".format(self.model_math()))
        plt.plot(list(self.dp.mjd_map.keys()), chi2_day, '.')
        plt.xlabel("MJD")
        plt.ylabel("$\chi^2/day$ [mag]")
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_path.joinpath("chi2_mjd.png"), dpi=300.)
        plt.close()

    def _dump_recap(self):
        d = {}
        d['photometry'] = self.config['photometry']
        d['dataset_name'] = self.dataset_name
        d['piedestal'] = self.config['piedestal']
        d['solver'] = self.config['solver']
        d['exposure_count'] = len(self.dp.mjd_map)
        d['star_count'] = len(self.dp.gaiaid_map)
        d['measure_count'] = len(self.dp.nt)
        d['bads_count'] = np.sum(self.bads).item()
        d['chi2'] = self.chi2
        d['ndof'] = self.ndof.item()
        d['chi2_ndof'] = (self.chi2/self.ndof).item()
        d['model'] = self.model_name()

        return d

    def dump_recap(self, output_path):
        d = self._dump_recap()

        with open(output_path, 'w') as f:
            dump(d, f)

    def solve(self):
        method = self.__config['solver']
        model = self.build_model()
        y = self.dp.mag
        weights = 1./self.measure_errors

        if self.config['eq_constraints']:
            constraints = self.eq_constraints(model)
            rows, cols, vals = model.rows.tolist(), model.cols.tolist(), model.vals.tolist()
            for constraint in constraints:
                constraint_cols, constraint_val = constraint
                rows.extend([max(rows)+1]*len(constraint_cols))
                cols.extend(constraint_cols.tolist())
                vals.extend([constraint_val]*len(constraint_cols))

            model = LinearModel(rows, cols, vals, struct=model.struct)
            y = np.concatenate([y, [0.]*len(constraints)])
            weights = np.concatenate([weights, [1.]*len(constraints)])


        start_time = time.perf_counter()
        if method == 'cholesky':
            solver = self._solve_cholesky(model, y, weights)
        elif method == 'cholesky_flip_flop':
            solver = self._solve_flip_flop(model, y, weights)
        else:
            raise NotImplementedError("solve(): {} solve method not implemented!".format(method))
        print("Elapsed time={}s".format(time.perf_counter()-start_time))

        res = solver.get_res(y)
        self.fitted_params = solver.model.params
        self.bads = solver.bads[:len(self.dp.mag)]
        self.res = res[:len(self.dp.mag)]

        # self.cov = solver.get_cov() # Getting cov matrix leads to crash
        # self.diag_cov = solver.get_diag_cov()
        self.cov = None
        self.diag_cov = None
        if(self.config['eq_constraints']):
            self.eq_constraints_res = res[-3:]

    def _solve_cholesky(self, model, y, weights):
        if not self.config['eq_constraints']:
            self.fix_params(model)
        solver = RobustLinearSolver(model, y, weights=weights)
        solver.model.params.free = solver.robust_solution(local_param='m')
        return solver

    def _solve_flip_flop(self, model, y, weights):
        def _fix_params(model, params):
            for param in params:
                model.params[param].fix()

        def _release_params(model, params):
            for param in params:
                model.params[param].release()

        max_iter = self.__config['flip_flop_max_iter']
        bads = None
        local_param = None

        fields = model.params.full.struct.slices.keys()
        flip_fields = ['m']
        flop_fields = list(fields-flip_fields)

        flip = False
        for i in range(max_iter):
            print("Iteration {}".format(i))
            if flip:
                params_to_flip = flop_fields
                local_param = 'm'
                flip = False
            else:
                params_to_flip = flip_fields
                local_param = None
                self.fix_params(model)
                flip = True

            print("Parameters to fix: {}".format(params_to_flip))

            _fix_params(model, params_to_flip)
            solver = RobustLinearSolver(model, self.dp.mag, weights=1./self.measure_errors)
            solver.bads = bads
            model.params.free = solver.robust_solution(local_param=local_param)
            model.params[:].release()
            bads = solver.bads

        self.fix_params(model)
        solver = RobustLinearSolver(model, self.dp.mag, weights=1./self.measure_errors)
        solver.bads = bads
        model.params.free = solver.robust_solution(local_param='m')

        return solver

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

        self.chi2 = np.sum(self.wres[~self.bads]**2).item()



Models = {}

def register_model(model):
    Models[model.model_name()] = model
