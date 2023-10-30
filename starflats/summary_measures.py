#!/usr/bin/env python3

import argparse
import pathlib

import pandas as pd


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Output relevant measurement summary.")
    argparser.add_argument('--sequence-path', type=pathlib.Path, required=True)

    args = argparser.parse_args()
    args.sequences_path = args.sequence_path.expanduser().resolve()

    sequence_paths = list(args.sequence_path.glob("*"))

    sequences = {}
    for sequence_path in sequence_paths:
        print("Processing {}...".format(sequence_path.stem))
        sequences[sequence_path] = {}

        date_band = sequence_path.stem.split("_")[-1]
        date = "-".join(date_band.split("-")[:-1])
        band = date_band.split("-")[-1]
        measures_df = pd.read_parquet(sequence_path, columns=['gaiaid', 'mjd', 'seeing', 'ra', 'dec'])
        measure_count = len(measures_df)
        star_count = len(list(set(measures_df['gaiaid'].tolist())))
        exposure_count = len(list(set(measures_df['mjd'].tolist())))
        mean_seeing = measures_df['seeing'].mean()
        mean_ra = measures_df['ra'].mean()
        mean_dec = measures_df['dec'].mean()

        sequences[sequence_path]['date'] = date
        sequences[sequence_path]['band'] = band
        sequences[sequence_path]['measure_count'] = measure_count
        sequences[sequence_path]['star_count'] = star_count
        sequences[sequence_path]['exposure_count'] = exposure_count
        sequences[sequence_path]['mean_seeing'] = mean_seeing
        sequences[sequence_path]['mean_ra'] = mean_ra
        sequences[sequence_path]['mean_dec'] = mean_dec
        sequences[sequence_path]['mean_star_per_ccd'] = star_count/16.

    sequences_df = pd.DataFrame.from_dict(sequences, orient='index')

    sequences_df.to_csv("sequences.csv")
