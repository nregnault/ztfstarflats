#!/usr/bin/env python3

import sys
import pathlib

import pandas as pd

from mask import FocalPlaneMask


def _load_dataset(path):
    df = pd.read_parquet(path)
    if path.with_suffix(".mask.yaml").exists():
        mask = FocalPlaneMask.from_dataset(df)
        df = df.loc[mask]

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

    dataset_paths = list(args.dataset_folder_path.joinpath("starflat_measures_*-{}.parquet".format(args.filtercode)))

    concat_df = _load_dataset(dataset_paths[0])
    print(concat_df)
    for dataset_path in dataset_paths[1:]:
        concat_df = concat_df.append(_load_dataset(dataset_path))
        print(concat_df)

sys.exit(main())
