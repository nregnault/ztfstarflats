""" Minimalist Python wrapper to the SelInv sparse matrix inversion code
from L. Lin, C. Yang, J. Meza, J. Lu, L. Ying and W. E, SelInv -- An algorithm
for selected inversion of a sparse symmetric matrix, ACM Trans. Math.
Software 37, 40, 2011. See the ReadMe notice in the selinv directory. 
"""
import numpy as np
import os
import ctypes
import scipy.sparse as sp
import starflats

_lib = np.ctypeslib.load_library('libselinv', starflats.__path__[0])
_selinv = _lib.selinv
_selinv.restype = None
_selinv.argtypes = [
    np.ctypeslib.ndpointer("i4", ndim=1, flags='aligned, contiguous'),
    np.ctypeslib.ndpointer("i4", ndim=1, flags='aligned, contiguous'),
    np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous'),
    np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous, writeable'),
    ctypes.c_int,
    ]

_selinvp = _lib.selinvp
_selinvp.restype = None
_selinvp.argtypes = [
    np.ctypeslib.ndpointer("i4", ndim=1, flags='aligned, contiguous'),
    np.ctypeslib.ndpointer("i4", ndim=1, flags='aligned, contiguous'),
    np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous'),
    np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous, writeable'),
    ctypes.c_int,
    np.ctypeslib.ndpointer("i4", ndim=1, flags='aligned, contiguous')
    ]

_preproc = _lib.preproc
_preproc.restype = ctypes.c_int
_preproc.argtypes = [
    np.ctypeslib.ndpointer("i4", ndim=1, flags='aligned, contiguous'),
    np.ctypeslib.ndpointer("i4", ndim=1, flags='aligned, contiguous'),
    np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous'),
    ctypes.c_int,
    np.ctypeslib.ndpointer("i4", ndim=1, flags='aligned, contiguous'),
    ]

_selinvpreprocessed = _lib.selinvpreprocessed
_selinvpreprocessed.restype = None
_selinvpreprocessed.argtypes = [
    np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous, writeable'),
    np.ctypeslib.ndpointer('i4', ndim=1, flags='aligned, contiguous, writeable'),
    np.ctypeslib.ndpointer('i4', ndim=1, flags='aligned, contiguous, writeable'),
    np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous, writeable'),
    ctypes.c_int,
    np.ctypeslib.ndpointer('i4', ndim=1, flags='aligned, contiguous, writeable'),
    np.ctypeslib.ndpointer('i4', ndim=1, flags='aligned, contiguous, writeable'),
    np.ctypeslib.ndpointer('i4', ndim=1, flags='aligned, contiguous, writeable'),
    np.ctypeslib.ndpointer(float, ndim=1, flags='aligned, contiguous, writeable'),
    ]

def selinv(mat, P=None):
    n = mat.shape[0]
    mat = mat.tocsc()
    print((repr(mat)))
    requires = ['CONTIGUOUS', 'ALIGNED']
    colptr = np.asanyarray(mat.indptr + 1, dtype='i4')
    colptr = np.require(colptr, 'i4', requires)
    rowind = np.asanyarray(mat.indices + 1, dtype='i4')
    rowind = np.require(rowind, 'i4', requires)
    nzvals = np.asanyarray(mat.data)
    nzvals = np.require(nzvals, float, requires)
    diag = np.zeros(n)
    if P is not None:
        P = np.asanyarray(P + 1, dtype='i4')
        P = np.require(P, 'i4', requires)
        _selinvp(colptr, rowind, nzvals, diag, n, P)
    else:
        _selinv(colptr, rowind, nzvals, diag, n)
    return diag

def selinvfull(mat, P):
    n = mat.shape[0]
    mat = mat.tocsc()
    print((repr(mat)))
    requires = ['CONTIGUOUS', 'ALIGNED']
    colptr = np.asanyarray(mat.indptr + 1, dtype='i4')
    colptr = np.require(colptr, 'i4', requires)
    rowind = np.asanyarray(mat.indices + 1, dtype='i4')
    rowind = np.require(rowind, 'i4', requires)
    nzvals = np.asanyarray(mat.data)
    nzvals = np.require(nzvals, float, requires)
    diag = np.zeros(n)
    
    P = np.asanyarray(P + 1, dtype='i4')
    P = np.require(P, 'i4', requires)
    nnz = _preproc(colptr, rowind, nzvals, n, P)
    print(nnz)
    Acolptr = np.zeros(n+1, dtype='i4')
    Acolptr = np.require(Acolptr, 'i4', requires)
    perm = np.zeros(n, dtype='i4')
    perm = np.require(perm, 'i4', requires)
    Arowind = np.zeros(nnz, dtype='i4')
    Arowind = np.require(Arowind, 'i4', requires)
    Ainv = np.zeros(nnz)
    Ainv = np.require(Ainv, 'float', requires)
    lcolptr = np.require(np.zeros(n+1), 'i4', requires)
    lrowind = np.require(np.zeros(nnz), 'i4', requires)
    lnz = np.require(np.zeros(nnz), float, requires)
    
    _selinvpreprocessed(diag, Acolptr, Arowind, Ainv, nnz, perm, lcolptr, lrowind, lnz)
    L = sp.csc_matrix((lnz, lrowind, lcolptr))
    A = sp.csc_matrix((Ainv, Arowind, Acolptr))
    return L, A, perm - 1

class fullselinv():
    def __init__(self, L, Ai, perm):
        self.L = L
        self.L.data[L.indptr[:-1]] = 0
        self.L.eliminate_zeros()
        self.Ai = Ai
        self.perm = perm
        
    def diag(self):
        return self.Ai.diagonal()[perm]

    def __getitem__(self, index):
        subcols = perm[index]
        sortindex = np.argsort(subcols)
        subcols = subcols[sortindex]
        subA = np.tril((self.Ai[np.ix_(subcols, subcols)]).todense())
        subA += np.triu(subA.T, 1)
        invsort = np.zeros_like(sortindex)
        invsort[sortindex] = np.arange(len(sortindex))
        return subA[np.ix_(invsort, invsort)]
    
def readmat(fname):
    fid = open(fname, 'rb')
    fid.seek(0)
    header = np.fromfile(fid, dtype='<i4', count=1)
    cols, entries = np.fromfile(fid, dtype='<i4', count=2)
    header = np.fromfile(fid, dtype='<i4', count=1)
    header = np.fromfile(fid, dtype='<i4', count=1)
    colptr = np.fromfile(fid, dtype='<i4', count=cols + 1)
    header = np.fromfile(fid, dtype='<i4', count=1)
    header = np.fromfile(fid, dtype='<i4', count=1)
    indx = np.fromfile(fid, dtype='<i4', count=entries)
    header = np.fromfile(fid, dtype='<i4', count=1)
    header = np.fromfile(fid, dtype='<i4', count=1)
    rval = np.fromfile(fid, dtype='<f8', count=entries)
    header = np.fromfile(fid, dtype='<i4', count=1)
    return sp.csc_matrix((rval, indx - 1, colptr -1))

if __name__ == '__main__':
    M = readmat('bcsstk14.ccf')
    print((repr(selinv(M))))
