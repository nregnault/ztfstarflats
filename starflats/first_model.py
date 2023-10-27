#!/usr/bin/env python3

import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import SuperpixelizedZTFFocalPlan, plot_ztf_focal_plane
from dataproxy import DataProxy
from linearmodels import RobustLinearSolver, indic


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Simple delta ZP starflat fit")
    argparser.add_argument('--dataset-path', type=pathlib.Path, required=True)
    argparser.add_argument('--output', type=pathlib.Path, required=True)

    args = argparser.parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not args.output.joinpath("cache.parquet").exists():
        df = pd.read_parquet(args.dataset_path)
        # df = df.sample(frac=0.2)

    # Remove stars that are measured less than N times
        min_measure_count = 10
        print("Removing stars that have less than {} measures...".format(min_measure_count))
        print(" Total measures: {}".format(len(df)))
        to_remove_mask = np.array([False]*len(df))
        gaiaids = list(set(df['gaiaid'].astype(int)))
        for gaiaid in gaiaids:
            print(".", end="", flush=True)
            gaiaid_mask = np.array(df['gaiaid'].astype(int).to_numpy() == gaiaid)
            if len(df.loc[gaiaid_mask]) < min_measure_count:
                to_remove_mask = np.any([to_remove_mask, gaiaid_mask], axis=0)
        df = df.iloc[~to_remove_mask]
        print("Done.")
        print(" New total measure: {}".format(len(df)))

        df = df.loc[df['psfflux']>=0.]
        df.to_parquet(args.output.joinpath("cache.parquet"))
    else:
        df = pd.read_parquet(args.output.joinpath("cache.parquet"))

    if not args.output.joinpath("model.pickle").exists():
        df['mag'] = -2.5*np.log10(df['psfflux'])
        df['emag'] = 1.08*df['epsfflux']/df['psfflux']

        res = 25
        superpixels = SuperpixelizedZTFFocalPlan(res)
        df['zp_index'] = superpixels.superpixelize(df['x'].to_numpy(), df['y'].to_numpy(), df['ccdid'].to_numpy(), df['qid'].to_numpy())

        kwargs = dict([(col_name, col_name) for col_name in df.columns])
        dp = DataProxy(df.to_records(), **kwargs)

        dp.make_index('gaiaid')

        model = indic(dp.gaiaid_index, name='m') + indic(dp.zp_index, name='zp')
        model.params['zp'].fix(0, 0.)
        piedestal = 0.
        solver = RobustLinearSolver(model, dp.mag, weights=1./np.sqrt(dp.emag**2+piedestal**2))
        solver.model.params.free = solver.robust_solution(local_param='m')

        with open("model.pickle", 'wb') as f:
            pickle.dump({'model': solver.model, 'bads':solver.bads, 'resolution': res}, f)
        bads = solver.bads
    else:
        with open("model.pickle", 'rb') as f:
            d = pickle.load(f)
            model = d['model']
            bads = d['bads']
            res = d['resolution']

    df['mag'] = -2.5*np.log10(df['psfflux'])
    df['emag'] = 1.08*df['epsfflux']/df['psfflux']
    kwargs = dict([(col_name, col_name) for col_name in df.columns])
    dp = DataProxy(df.to_records(), **kwargs)
    dp.make_index('gaiaid')
    dp.make_index('mjd')

    res = dp.mag - model()
    wres = res/dp.emag

    chi2_mjd = np.bincount(dp.mjd_index, weights=wres**2)/np.bincount(dp.mjd_index)
    chi2_stars = np.bincount(dp.gaiaid_index, weights=wres**2)/np.bincount(dp.gaiaid_index)

    plt.plot(chi2_mjd, '.')
    plt.show()

    plt.plot(chi2_stars, '.')
    plt.show()

    plt.subplots(figsize=(12., 5.))
    plt.suptitle("Residual plot wrt $G$ magnitude")
    plt.plot(dp.G[~bads], res[~bads], ',')
    plt.xlabel("$m_G$ [AB mag]")
    plt.ylabel("$m_\mathrm{ADU}-m_\mathrm{model}$ [mag]")
    plt.ylim(-0.75, 0.75)
    plt.grid()
    plt.show()
    plt.close()

    plt.subplots(figsize=(12., 5.))
    plt.suptitle("Residual plot wrt $Bp-Rp$ magnitude")
    plt.plot(dp.BP[~bads] - dp.RP[~bads], res[~bads], ',')
    plt.xlabel("$m_{Bp}-m_{Rp}$ [mag]")
    plt.ylabel("$m_\mathrm{ADU}-m_\mathrm{model}$ [mag]")
    plt.ylim(-0.75, 0.75)
    plt.grid()
    plt.show()
    plt.close()

    plt.subplots(figsize=(12., 5.))
    plt.suptitle("Residual plot wrt MJD")
    plt.plot(dp.mjd[~bads], res[~bads], ',')
    plt.xlabel("MJD")
    plt.ylabel("$m_\mathrm{ADU}-m_\mathrm{model}$ [mag]")
    plt.ylim(-0.75, 0.75)
    plt.grid()
    plt.show()
    plt.close()

    superpixels = SuperpixelizedZTFFocalPlan(res)
    zps = model.params['zp'].full
    fig, axs = plt.subplots(figsize=(12., 12.))
    plt.axis('off')
    plt.axis('equal')
    superpixels.plot(fig, zps)
    plt.tight_layout()
    plt.show()
