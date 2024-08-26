#!/usr/bin/env python3

import sys
import pathlib
import argparse

import pandas as pd
import numpy as np

from mask import FocalPlaneMask
from utils import make_index_from_array


def _load_dataset(path, sequenceid, starid_offset=0):
    df = pd.read_parquet(path)
    if path.with_suffix(".mask.yaml").exists():
        mask = FocalPlaneMask.from_yaml(path.with_suffix(".mask.yaml"))
        df = df.loc[mask.mask_from_data(df)]
    df = df.assign(starid=make_index_from_array(df['gaiaid'].to_numpy())[1]+starid_offset,
                   sequenceid=sequenceid)

    return df


def main():
    argparser = argparse.ArgumentParser(description="Starflat model solver.")
    argparser.add_argument('--dataset-folder-path', type=pathlib.Path, help="Starflat sequence dataset folder path.")
    argparser.add_argument('--output-path', type=pathlib.Path)
    argparser.add_argument('--filtercode', type=str)
    argparser.add_argument('--year', type=int)

    args = argparser.parse_args()

    args.dataset_folder_path = args.dataset_folder_path.expanduser().resolve()
    args.output_path = args.output_path.expanduser().resolve()

    if args.year:
        dataset_paths = list(args.dataset_folder_path.glob("starflat_measures_{}-*-{}.parquet".format(args.year, args.filtercode)))
    else:
        dataset_paths = list(args.dataset_folder_path.glob("starflat_measures_*-{}.parquet".format(args.filtercode)))

    sequenceid = 0
    print("Loading {}... ".format(dataset_paths[0].stem), end="", flush=True)
    concat_df = _load_dataset(dataset_paths[0], sequenceid)
    print("Done. Total size={}".format(len(concat_df)))
    for dataset_path in dataset_paths[1:]:
        sequenceid = sequenceid + 1
        starid_offset = concat_df['starid'].max()+1
        print("Loading {}... ".format(dataset_path.stem), end="", flush=True)
        concat_df = pd.concat([concat_df, _load_dataset(dataset_path, sequenceid, starid_offset=starid_offset)])
        print("Done. Total size={}".format(len(concat_df)))

    print("Saving as {}".format(args.output_path))
    concat_df.to_parquet(args.output_path)
    print("Done")


sys.exit(main())
