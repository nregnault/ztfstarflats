#!/usr/bin/env python3

import yaml
import numpy as np


class FocalPlaneMask:
    def __init__(self, mask=None):
        if mask is not None:
            self.mask = mask
        else:
            self.fill(True)

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            return cls(yaml.load(f, Loader=yaml.Loader))

    @classmethod
    def from_data(cls, df):
        a = (np.bincount((df['ccdid']-1)*4+df['qid']) > 0)
        m = dict([(ccdid, dict([(qid, a[(ccdid-1)*4+qid]) for qid in range(4)])) for ccdid in range(1, 17)])
        return cls(m)

    def __getitem__(self, idx):
        return self.mask[idx[0]][idx[1]]

    def fill(self, b):
        assert isinstance(b, bool)
        self.mask = {}
        for ccdid in range(1, 17):
            self.mask[ccdid] = {}
            for qid in range(4):
                self.mask[ccdid][qid] = b

    def __and__(self, other):
        mask = FocalPlaneMask()
        for ccdid in range(1, 17):
            for qid in range(0, 4):
                mask.mask[ccdid][qid] = (self.mask[ccdid][qid] & other.mask[ccdid][qid])

        return mask

    def to_yaml(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.mask, f)

    def mask_from_data(self, df):
        a = np.array([[self.mask[ccdid][qid] for qid in range(4)] for ccdid in range(1, 17)]).flatten()
        m = (df['ccdid']-1)*4+df['qid']
        return a[m]
