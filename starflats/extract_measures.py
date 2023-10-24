#!/usr/bin/env python3

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time

from utils import ztffiltercodes
import matplotlib.pyplot as plt

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Extract relevant starflat measure fields from measurement datasets.")
    argparser.add_argument('--dataset-path', type=pathlib.Path, required=True)
    argparser.add_argument('--output', type=pathlib.Path, required=True)
    argparser.add_argument('--year', type=int)
    argparser.add_argument('--filtercode', type=str, choices=ztffiltercodes+['all'], default='all')
    argparser.add_argument('--min-measure-count', type=int, default=10)
    argparser.add_argument('--max-g-mag', type=float, default=18.)

    args = argparser.parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if args.filtercode == 'all':
        filtercodes = ztffiltercodes
    else:
        filtercodes = [args.filtercode]

    filtercode = 'zg'
    # First retrieve relevant datasets
    dataset_paths = list(args.dataset_path.glob("measures_{}_*-{}-*.parquet".format(args.year, filtercode)))

    dataset_list = []
    print("Loading datasets from {}, in band {}".format(args.dataset_path, filtercode))
    for dataset_path in dataset_paths:
        columns_to_extract = ['gaia_Source', 'psf_x', 'psf_y', 'gaia_RA_ICRS', 'gaia_DE_ICRS', 'psf_flux', 'psf_eflux', 'gaia_Gmag', 'gaia_BPmag', 'gaia_RPmag', 'gaia_e_Gmag', 'gaia_e_BPmag', 'gaia_e_RPmag',
                              'mjd', 'qid', 'ccdid', 'filtercode', 'quadrant', 'seeing']
        [columns_to_extract.extend(['aper_apfl{}'.format(i), 'aper_eapfl{}'.format(i), 'aper_rad{}'.format(i)]) for i in range(10)]
        df = pd.read_parquet(dataset_path, columns=columns_to_extract)
        df.rename(columns={'gaia_Source': 'gaiaid',
                           'psf_x': 'x',
                           'psf_y': 'y',
                           'gaia_RA_ICRS': 'ra',
                           'gaia_DE_ICRS': 'dec',
                           'psf_flux': 'psfflux',
                           'psf_eflux': 'epsfflux',
                           'gaia_Gmag': 'G',
                           'gaia_BPmag': 'BP',
                           'gaia_RPmag': 'RP',
                           'gaia_e_Gmag': 'eG',
                           'gaia_e_RPmag': 'eRP',
                           'gaia_e_BPmag': 'eBP'}, inplace=True)
        [df.rename(columns={'aper_apfl{}'.format(i): 'apfl{}'.format(i), 'aper_eapfl{}'.format(i): 'eapfl{}'.format(i), 'aper_rad{}'.format(i): 'rad{}'.format(i)}, inplace=True) for i in range(10)]

        s = len(df)
        df = df.loc[df['G']<=args.max_g_mag]

        print("Loading dataset from {} - removed {} measures having Gmag > Gmax - left total={}".format(dataset_path.name, s - len(df), len(df)))
        # TODO: ra, dec in ICRS epoch, need to translate to mjd of current exposures

        dataset_list.append(df)

    dataset_df = pd.concat(dataset_list, ignore_index=True)

    print("Removing stars that have less than {} measures...".format(args.min_measure_count))
    s = len(dataset_df)
    to_remove_mask = np.array([False]*len(dataset_df))
    gaiaids = list(set(df['gaiaid'].astype(int)))
    for i, gaiaid in enumerate(gaiaids):
        if i%100 == 0:
            print(".", end="", flush=True)

        gaiaid_mask = np.array(dataset_df['gaiaid'].astype(int).to_numpy() == gaiaid)
        if len(dataset_df.loc[gaiaid_mask]) < args.min_measure_count:
            to_remove_mask = np.any([to_remove_mask, gaiaid_mask], axis=0)
    s = len(dataset_df)
    dataset_df = dataset_df.iloc[~to_remove_mask]
    print("")
    print(" Removed {} measures.".format(s-len(dataset_df)))

    # Detect starflat nights
    unique_night_mjds = list(set(dataset_df['mjd'].astype(int)))
    unique_nights = [Time(unique_night_mjd, format='mjd').to_value('datetime').date().isoformat() for unique_night_mjd in unique_night_mjds]

    print("Detected {} distinct night(s)".format(len(unique_nights)))
    for unique_night_mjd, unique_night in zip(unique_night_mjds, unique_nights):
        night_mask = (dataset_df['mjd'].astype(int)==unique_night_mjd)
        print("Saving night {} - {} measurements".format(unique_night, sum(night_mask)))
        dataset_df.loc[night_mask].to_parquet(args.output.joinpath("starflat_measures_{}-{}.parquet".format(unique_night, filtercode)))
