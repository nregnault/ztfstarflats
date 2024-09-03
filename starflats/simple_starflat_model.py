#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pathlib

import models
from utils import SuperpixelizedZTFFocalPlane, plot_ztf_focal_plane, quadrant_width_px, quadrant_height_px
from linearmodels import indic

class SimpleStarflatModel(models.StarflatModel):
    def __init__(self, config=None, mask=None):
        super().__init__(config=config, mask=mask)
        self.superpixels = SuperpixelizedZTFFocalPlane(self.config['zp_resolution'])
        self.fit_gain = config.get('fit_gain', False)

        if self.fit_gain:
            self.gain_plane = SuperpixelizedZTFFocalPlane(1)

    def load_data(self, dataset_path):
        super().load_data(dataset_path)

        self.dp.add_field('dzp', self.superpixels.superpixelize(self.dp.x, self.dp.y, self.dp.ccdid, self.dp.qid))
        self.dp.make_index('dzp')

        self.dzp_to_index = -np.ones(64*self.superpixels.resolution**2, dtype=int)
        np.put_along_axis(self.dzp_to_index, np.array(list(self.dp.dzp_map.keys())), np.array(list(self.dp.dzp_map.values())), axis=0)

        if self.fit_gain:
            self.dp.add_field('gain', self.dp.sequenceid*64+self.dp.rcid)
            self.dp.make_index('gain')

            self.gain_to_index = -np.ones((self.sequence_count*64), dtype=int).flatten()
            np.put_along_axis(self.gain_to_index, np.array(list(self.dp.gain_map.keys())), np.array(list(self.dp.gain_map.values())), axis=0)

    def _build_model(self):
        models = [indic(self.dp.starid_index, name='starid'), indic(self.dp.dzp_index, name='dzp')]
        if self.fit_gain:
            # models.append(indic(self.dp.rcid_index, name='gain'))
            models.append(indic(self.dp.gain_index, name='gain'))

        return models

    @staticmethod
    def model_desc():
        return "Simplest starflat model. Fit star magnitude and superpixelized ZP over the focal plane."

    @staticmethod
    def model_math():
        return "$m_\mathrm{ADU}=m_s+\delta ZP(u, v)$"

    @staticmethod
    def model_name():
        return 'simple'

    def fix_params(self, model):
        if self.fit_gain:
            for ccdid in range(1, 17):
                for qid in range(4):
                    if self.mask[ccdid, qid]:
                        if qid == 0:
                            model.params['dzp'].fix(self.dp.dzp_map[self.superpixels.vecrange(ccdid, qid).stop - 1], 0.)
                        elif qid == 1:
                            model.params['dzp'].fix(self.dp.dzp_map[self.superpixels.vecrange(ccdid, qid).stop - self.superpixels.resolution], 0.)
                        elif qid == 2:
                            model.params['dzp'].fix(self.dp.dzp_map[self.superpixels.vecrange(ccdid, qid).start], 0.)
                        else:
                            model.params['dzp'].fix(self.dp.dzp_map[self.superpixels.vecrange(ccdid, qid).start + self.superpixels.resolution-1], 0.)
            if 'sequenceid' in self.dp.nt.dtype.names:
                idx = 0
                for sequenceid in range(self.sequence_count):
                    model.params['gain'].fix(idx, 0.)
                    idx = idx + sum(self.gain_to_index[64*sequenceid:64*(sequenceid+1)]!=-1)
            else:
                model.params['gain'].fix(0, 0.)
        else:
            model.params['dzp'].fix(self.superpixels.vecrange(7, 0).start, 0.)

    def eq_constraints(self, model, mu=0.1):
        if self.fit_gain:
            # const = [[model.params['gain'].indexof(), mu]]
            # for ccdid in range(1, 17):
            #     for qid in range(4):
            #         const.append([model.params['dzp'].indexof(self.superpixels.vecrange(ccdid, qid)), mu])
            # return const
            raise NotImplementedError()
        else:
            return [[model.params['dzp'].indexof(), mu]]

    def _dump_recap(self):
        d = super()._dump_recap()
        d['dzp'] = self.superpixels.vecsize
        d['dzp_resolution'] = self.superpixels.resolution
        d['dzp_parameter'] = len(self.dp.dzp_set)
        d['dzp_sum'] = np.sum(self.fitted_params['dzp'].full).item()

        return d

    def plot(self, output_path):
        super().plot(output_path)

        chi2_ndof = np.sum(self.wres[~self.bads]**2)/self.ndof

        # Measurement count per superpixel
        fig, axs = plt.subplots(figsize=(12., 12.))
        plt.suptitle("Measure count per superpixel")
        self.superpixels.plot(fig, np.bincount(self.dp.dzp), cbar_label="Measure count")
        plt.savefig(output_path.joinpath("superpixel_count.png"), dpi=300.)
        plt.close()

        # Per quadrant median centered Delta ZP
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("$\delta ZP(u, v)$ (per quadrant median centered) - {}\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
        self.superpixels.plot(fig, self.fitted_params['dzp'].full, vec_map=self.dp.dzp_map, cmap='viridis', f=np.median, vlim=(-0.02, 0.02), cbar_label="$\delta ZP$ [mag]")
        plt.savefig(output_path.joinpath("centered_dzp.png"))
        plt.close()

        if self.fit_gain:
            gain_plane = SuperpixelizedZTFFocalPlane(1)

            # Gain distribution
            s = 0
            for sequenceid in range(self.sequence_count):
                fig = plt.figure(figsize=(5., 5.))
                plt.suptitle("Gain - SequenceID={}".format(sequenceid))
                vec_map = np.where(self.gain_to_index[64*sequenceid:64*(sequenceid+1)]==-1, -1, self.gain_to_index[64*sequenceid:64*(sequenceid+1)]-s)
                s = s + 64 - sum(self.gain_to_index[64*sequenceid:64*(sequenceid+1)]==-1)
                gain_plane.plot(fig, self.fitted_params['gain'].full[len(self.dp.rcid_set)*sequenceid:(len(self.dp.rcid_set))*(sequenceid+1)], vec_map=vec_map)
                plt.savefig(output_path.joinpath("gain_seq{}.png".format(sequenceid)))
                plt.close()

            # Delta ZP plane with gain
            fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
            plt.suptitle("$\delta ZP(u, v)$ with gain - {}\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
            # vec = self.fitted_params['dzp'].full.copy()
            vec = self.get_gain_dzp_vec()
            self.superpixels.plot(fig, self.get_gain_dzp_vec(), vec_map=self.dp.dzp_map, cmap='viridis', cbar_label="$\delta ZP$ [mag]")
            plt.savefig(output_path.joinpath("dzp_gain.png"))
            plt.close()

        # Raw Delta ZP plane
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("$\delta ZP(u, v)$ - {}\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
        self.superpixels.plot(fig, self.fitted_params['dzp'].full, vec_map=self.dp.dzp_map, cmap='viridis', cbar_label="$\delta ZP$ [mag]", vlim=(-0.02, 0.02))
        plt.savefig(output_path.joinpath("dzp.png"))
        plt.close()

        wres_dzp = np.bincount(self.dp.dzp_index, weights=self.wres**2)/np.bincount(self.dp.dzp_index)

        # Chi2 per superpixel
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        plt.suptitle("Stacked $\chi^2$ - {}\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, self.model_math(), chi2_ndof))
        self.superpixels.plot(fig, wres_dzp, vec_map=self.dp.dzp_map, vlim='sigma_clipping')
        plt.savefig(output_path.joinpath("chi2_superpixels.png"))
        plt.close()

        # Outlier count per Delta ZP superpixel
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12))
        plt.suptitle("Outlier count per superpixel")
        self.superpixels.plot(fig, np.bincount(self.dp.dzp_index, weights=self.bads), vec_map=self.dzp_to_index, cbar_label="Outlier count")
        plt.savefig(output_path.joinpath("superpixel_outlier.png"), dpi=300.)
        plt.close()

        # for mjd in self.dp.mjd_map.keys():
        #     mjd_index = self.dp.mjd_map[mjd]
        #     mask = (self.dp.mjd_index == mjd_index)
        #     wres_dzp = np.bincount(self.dp.dzp_index[mask], weights=self.wres[mask]**2, minlength=self.superpixels.vecsize)/np.bincount(self.dp.dzp_index[mask], minlength=self.superpixels.vecsize)

        #     fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12., 12.))
        #     plt.suptitle("Stacked $\chi^2$ - {} - MJD={}\n {} \n {} \n $\chi^2/\mathrm{{ndof}}$={}".format(self.config['photometry'], self.dataset_name, mjd, self.model_math(), chi2_ndof))
        #     self.superpixels.plot(fig, wres_dzp, vlim='mad_positive')
        #     plt.show()
        #     # plt.savefig(output_path.joinpath("chi2_superpixels.png"))
        #     plt.close()

    def apply_model(self, x, y, ccdid, qid, mag, **kwords):
        zp_index = self.dzp_to_index[self.superpixels.superpixelize(x, y, ccdid, qid)]
        return mag - self.fitted_params['dzp'].full[zp_index]

    def _dump_result(self):
        d = {}
        d['dzp_to_index'] = self.dzp_to_index
        return d

    def get_gain_dzp_vec(self):
        if self.sequence_count == 1:
            return self.fitted_params['dzp'].full + np.repeat(self.fitted_params['gain'].full, self.superpixels.resolution**2)
        else:
            vec = np.where(self.dzp_to_index==-1, np.nan, self.fitted_params['dzp'].full[self.dzp_to_index])

            for ccdid in range(1, 17):
                for qid in range(4):
                    if self.mask[ccdid, qid]:
                        gains = []
                        for sequenceid in range(self.sequence_count):
                            gain_to_index = self.gain_to_index[64*sequenceid:64*(sequenceid+1)]
                            if gain_to_index[self.gain_plane.vecrange(ccdid, qid)] != -1:
                                gains.append(self.fitted_params['gain'].full[gain_to_index[self.gain_plane.vecrange(ccdid, qid)]])

                        gain = np.mean(gains)
                        vec[self.superpixels.vecrange(ccdid, qid)] = vec[self.superpixels.vecrange(ccdid, qid)] + gain*np.ones(self.superpixels.resolution**2)

            return np.delete(vec, np.where(self.dzp_to_index==-1))

models.register_model(SimpleStarflatModel)
