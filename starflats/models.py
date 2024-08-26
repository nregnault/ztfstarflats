#!/usr/bin/env python3

import pickle
import time
import yaml
import pathlib
from yaml import load, Loader, dump

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from ztfquery.fields import ccdid_qid_to_rcid

from linearmodels import RobustLinearSolver, LinearModel
from dataproxy import DataProxy
from utils import binplot, idx2markerstyle, make_index_from_array, quadrant_width_px, quadrant_height_px, sanitize_data
from mask import FocalPlaneMask


photometry_choices = ['psf'] + ['apfl{}'.format(i) for i in range(10)]

photometry_choice_to_key = {'psf': 'psfflux'}
photometry_choice_to_key.update(dict([('apfl{}'.format(i), 'apfl{}'.format(i)) for i in range(10)]))
photometry_error_choice_to_key = {'psf': 'epsfflux'}
photometry_error_choice_to_key.update(dict([('apfl{}'.format(i), 'eapfl{}'.format(i)) for i in range(10)]))


class StarflatModel:
    def __init__(self, config=None, mask=None):
        self.__config = config
        if mask:
            self.__mask = mask
        else:
            self.__mask = FocalPlaneMask()

    @classmethod
    def from_config(cls, config_path):
        print("Loading settings from {}".format(config_path))
        with open(config_path, 'r') as f:
            config = load(f, Loader=Loader)

        return cls(config=config, mask=None)

    @classmethod
    def from_result(cls, config_path, result_path):
        model = cls.from_config(config_path)
        model.load_result(result_path)
        return model

    def load_mask(self, mask_path):
        self.__mask = FocalPlaneMask.from_yaml(mask_path)

    def load_data(self, dataset_path):
        photo_key = self.__config['photometry']
        photo_err_key = self.__config['photometry_error']
        photo_color_lhs = self.__config['color_lhs']
        photo_color_rhs = self.__config['color_rhs']
        photo_ext_cat = self.__config['photometry_ext_cat']

        print("Loading sequence dataset from {}".format(dataset_path))
        df = pd.read_parquet(dataset_path)
        print("")

        if 'starid' not in df.columns:
            print("No \'starid\' column, renaming  \'gaiaid\'")
            df = df.rename(columns={'gaiaid': 'starid'})

        print("Measure count={}".format(len(df)))
        print("Star count={}".format(len(list(set(df['starid'].tolist())))))
        print("")

        df = sanitize_data(df, self.__config['photometry'])

        data_mask = FocalPlaneMask.from_data(df)
        self.__mask = self.__mask & data_mask

        measure_count = len(df)
        df = df.loc[self.__mask.mask_from_data(df)]
        if len(df) < measure_count:
            print("Removed {} masked measures".format(measure_count-len(df)))
            print("Measure count={}".format(len(df)))
            print("")

        if 'max_mag' in self.__config.keys():
            print("Filtering faint stars (> {:.1f} mag)".format(self.__config['max_mag']))
            n = len(df)
            df = df.loc[df[photo_ext_cat]<self.__config['max_mag']]
            print("Removed {} measures".format(n-len(df)))
            print("Measure count={}".format(len(df)))
            print("")

        df['col'] = (df[photo_color_lhs] - df[photo_color_rhs]) - np.mean(df[photo_color_lhs] - df[photo_color_rhs])

        if 'max_col' in self.__config.keys():
            print("Filtering by star color ({}-{} > {} mag)".format(self.__config['color_lhs'], self.__config['color_rhs'], self.__config['max_col']))
            n = len(df)
            df = df.loc[df['col']<self.__config['max_col']]
            print("Removed {} measures".format(n-len(df)))
            print("Measure count={}".format(len(df)))
            print("")

        if 'min_col' in self.__config.keys():
            print("Filtering by star color ({}-{} < {} mag)".format(self.__config['color_lhs'], self.__config['color_rhs'], self.__config['min_col']))
            n = len(df)
            df = df.loc[df['col']>self.__config['min_col']]
            print("Removed {} measures".format(n-len(df)))
            print("Measure count={}".format(len(df)))
            print("")

        df['mag'] = -2.5*np.log10(df[photo_key])
        df['emag'] = 1.08*df[photo_err_key]/df[photo_key]

        df['rcid'] = ccdid_qid_to_rcid(df['ccdid'], df['qid']+1)

        df['ext_cat_mag'] = df[photo_ext_cat]

        if 'min_measures' in self.__config.keys():
            print("Removing stars that have less than {} measures...".format(self.__config['min_measures']))
            starid_index_map, starid_index = make_index_from_array(df['starid'].to_numpy())
            starid_mask = (np.bincount(starid_index) < self.__config['min_measures'])
            to_remove_mask = starid_mask[starid_index]
            df = df.loc[~to_remove_mask]
            print("Removed {} measures".format(sum(to_remove_mask)))
            print("Measure count={}".format(len(df)))
            print("Star count={}".format(len(list(set(df['starid'].tolist())))))
            print("")

        # Might not be great to create a dataproxy from the full dataset....
        kwargs = dict([(col_name, col_name) for col_name in df.columns])
        self.dp = DataProxy(df.to_records(), **kwargs)
        self.__dataset_name = dataset_path.stem

        self.dp.make_index('qid')
        self.dp.make_index('ccdid')
        self.dp.make_index('rcid')
        self.dp.make_index('mjd')
        self.dp.make_index('starid')

        if 'sequenceid' in self.dp.nt.dtype.names:
            print("\'sequenceid\' field found, making an index of it")
            self.dp.make_index('sequenceid')

    @property
    def config(self):
        return self.__config

    @property
    def mask(self):
        return self.__mask

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def wres(self):
        return self.res/self.measure_errors

    @property
    def chi2(self):
        return np.sum(self.wres[~self.bads]**2).item()

    @property
    def ndof_full(self):
        return len(self.dp.nt) - len(self.fitted_params.full)

    @property
    def ndof(self):
        return len(self.dp) - len(self.fitted_params.full) - sum(self.bads)

    @property
    def measure_errors(self):
        return np.sqrt(self.dp.emag**2+self.config['piedestal']**2)

    def _build_model(self):
        raise NotImplementedError

    def build_model(self):
        models = self._build_model()
        return sum(models[1:], start=models[0])

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
        raise NotImplementedError

    def plot(self, output_path):
        wres = self.wres

        # Plot dithering pattern
        plt.subplots(figsize=(8., 8.))
        plt.suptitle("Dithering pattern - {}\n {}".format(self.config['photometry'], self.dataset_name))
        mask = (self.dp.starid_index == np.argmax(np.bincount(self.dp.starid_index)))
        rcids = list(set(self.dp.rcid[mask]))

        for i, rcid in enumerate(rcids):
            rcid_mask = (self.dp.rcid[mask] == rcid)
            plt.plot(self.dp.x[mask][rcid_mask], self.dp.y[mask][rcid_mask], idx2markerstyle[i], label=rcid)

        # Dithering pattern in pixel space
        plt.grid()
        plt.legend()
        plt.axis('equal')
        plt.xlabel("$x$ [pixel]")
        plt.ylabel("$y$ [pixel]")
        plt.savefig(output_path.joinpath("dithering.png"), dpi=300.)
        plt.close()

        # Residual wrt magnitude
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

        # Residual wrt color
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
        d['star_count'] = len(self.dp.starid_map)
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
            print("Adding equality constraints")
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

        print("Solving model...")
        start_time = time.perf_counter()
        if method == 'cholesky':
            solver = self._solve_cholesky(model, y, weights)
        elif method == 'cholesky_flip_flop':
            solver = self._solve_flip_flop(model, y, weights)
        else:
            raise NotImplementedError("solve(): {} solve method not implemented!".format(method))
        print("Elapsed time={}s".format(time.perf_counter()-start_time))
        print("")

        res = solver.get_res(y)
        self.fitted_params = solver.model.params
        self.bads = solver.bads[:len(self.dp.mag)]
        self.res = res[:len(self.dp.mag)]
        # self.cov = solver.get_cov() # Getting cov matrix leads to crash
        # self.diag_cov = solver.get_diag_cov()
        self.cov = None
        self.diag_cov = None
        # if(self.config['eq_constraints']):
        #     self.eq_constraints_res = res[-3:]

    def _solve_cholesky(self, model, y, weights):
        print("Using Cholesky method")

        if not self.config['eq_constraints']:
            self.fix_params(model)

        print("Parameter count=({})".format(", ".join(["{}:{}".format(param, len(model.params[param].free)) for param in model.params._struct])))
        print("Total parameter count={}".format(len(model.params.free)))

        solver = RobustLinearSolver(model, y, weights=weights, use_long=False)
        solver.model.params.free = solver.robust_solution(local_param='starid')
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
                local_param = 'starid'
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
        model.params.free = solver.robust_solution(local_param='starid')

        return solver

    def apply_starflat(self, x, y, ccdid, qid, **kwords):
        raise NotImplementedError

    def _dump_result(self):
        raise NotImplementedError

    def dump_result(self, output_path):
        d = self._dump_result()

        with open(output_path, 'wb') as f:
            d.update({'fitted_params': self.fitted_params,
                      'bads': self.bads,
                      'res': self.res,
                      'cov': self.cov,
                      'mask': self.mask,
                      'dataset_name': self.dataset_name,
                      'config': self.config})
            pickle.dump(d, f)

    def load_result(self, result_path):
        if isinstance(result_path, pathlib.Path) or isinstance(result_path, str):
            with open(result_path, 'rb') as f:
                d = pickle.load(f)
        else:
            d = result_path

        property_keys = ['config', 'mask', 'dataset_name']
        [setattr(self, key, d[key]) for key in d.keys() if key not in property_keys]
        [setattr(self, "_StarflatModel__{}".format(key), d[key]) for key in d.keys() if key in property_keys]

        # self.fitted_params = d['fitted_params']
        # self.bads = d['bads']
        # self.res = d['res']
        # self.cov = d['cov']
        # self.dzp_to_index = d['dzp_to_index']
        # self.__config = d['config']
        # self.__mask = d['mask']


Models = {}

def register_model(model):
    Models[model.model_name()] = model
