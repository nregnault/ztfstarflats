#!/usr/bin/python
import numpy as np


def make_index(index):
    s = np.argsort(index)
    n = np.bincount(index.astype('int'))
    n = n[n != 0]
    l = []
    p = 0
    for i in n:
        l.append(s[p:p + i])
        p = p + i
    return l


def first_two_cols(ntuple):
    """ Return the first two columns of an array or NTuple
    """
    if len(ntuple.shape) == 1:
        col_x, col_y = ntuple.dtype.names[:2]
        return ntuple[col_x], ntuple[col_y]
    else:
        return ntuple[:, 0], ntuple[:, 1]


def find_closest(data, target, sorted=True):
    """
    Stolen from stakoverflow to find index closest to target from a sorted list A.

    If sorted is False, takes care of it
    """
    
    if not sorted:
        idx_sorted = np.argsort(data)
        A = data[idx_sorted]
    else:
        A = data
        idx_sorted = np.arange(len(data) - 1)
        
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    if idx > len(idx_sorted) -1:
        idx = -1    
    return idx_sorted[idx]
    
