#!/usr/bin/env python3

import sys
import pickle
import argparse
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from simple_starflat_model import SimpleStarflatModel


def main():
    argparser = argparse.ArgumentParser(description="Starflat model solver.")
    argparser.add_argument('--model-a-path', type=pathlib.Path)
    argparser.add_argument('--model-b-path', type=pathlib.Path)
    argparser.add_argument('--config-path', type=pathlib.Path)

    args = argparser.parse_args()

    args.model_a_path = args.model_a_path.expanduser().resolve()
    args.model_b_path = args.model_b_path.expanduser().resolve()
    args.config_path = args.config_path.expanduser().resolve()

    model_a = SimpleStarflatModel.from_result(args.config_path, args.model_a_path)
    model_b = SimpleStarflatModel.from_result(args.config_path, args.model_b_path)

    res = model_a.fitted_params['dzp'].full - model_b.fitted_params['dzp'].full

    model_compare = SimpleStarflatModel.from_result(args.config_path, pathlib.Path("/home/llacroix/starflat/output_test_apply/model.pickle"))
    # res = res + model_compare.fitted_params['dzp'].full

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
    # plt.suptitle("$\delta ZP(u, v)$ - {}\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
    model_a.superpixels.plot(fig, res, vec_map=model_a.dzp_to_index, cmap='viridis', f=np.median, vlim='mad', cbar_label="$\delta ZP$ [mag]")
    plt.show()
    plt.close()


sys.exit(main())
