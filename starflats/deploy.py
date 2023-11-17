#!/usr/bin/env python3

import pathlib
import argparse
import subprocess

all_actions = ['solve', 'plot', 'dump']

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Simple delta ZP starflat fit")
    argparser.add_argument('--dataset-path', type=pathlib.Path)
    argparser.add_argument('--config-path', type=pathlib.Path)
    argparser.add_argument('--output-path', type=pathlib.Path)
    argparser.add_argument('--run-path', type=pathlib.Path)
    argparser.add_argument('--actions', type=str)
    argparser.add_argument('--generate', action='store_true')
    argparser.add_argument('--schedule', action='store_true')

    args = argparser.parse_args()

    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.output_path = args.output_path.expanduser().resolve()
    args.config_path = args.config_path.expanduser().resolve()
    args.run_path = args.run_path.expanduser().resolve()

    if args.actions == "" and args.generate:
        print("Cannot generate jobs without actions!")
        exit()

    # if args.actions != "":
    #     actions = args.actions.split(",")
        # if any([~(action in all_actions) for action in actions]):
        #     print("Unrecognised action in : {}".format(actions))
        #     exit()

        # print("Actions: {}".format(actions))

    dataset_paths = list(args.dataset_path.glob("*2022*.parquet"))
    print(dataset_paths)
    print("Found {} datasets".format(len(dataset_paths)))

    if args.generate:
        for dataset_path in dataset_paths:
            dataset_name = dataset_path.stem.split("_")[-1]
            print(dataset_name)
            job = """#!/bin/sh
export PATH=${{PATH}}:~/projects/ztfstarflats/starflats
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
starflat_all_photometry.sh {dataset_path} {config_path} {output_path}
            """.format(dataset_path=dataset_path, config_path=args.config_path, output_path=args.output_path)

            with open(args.run_path.joinpath("batches/{}.sh".format(dataset_name)), 'w') as f:
                f.write(job)

    if args.schedule:
        for dataset_path in dataset_paths:
            dataset_name = dataset_path.stem.split("_")[-1]
            print(dataset_name)
            cmd = ["sbatch", "--ntasks=1",
                   "-D", "{}".format(args.run_path),
                   "-J", "starflat_{}".format(dataset_name),
                   "-o", args.run_path.joinpath("logs/log_{}".format(dataset_name)),
                   "-A", "ztf",
                   "-L", "sps",
                   "--mem=20G",
                   "-t", "5-0",
                   args.run_path.joinpath("batches/{}.sh".format(dataset_name))]

            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)
            print(out.stdout)
