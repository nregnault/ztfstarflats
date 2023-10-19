#!/usr/bin/env python

import os
import os.path as op
import sys
import re

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO)

import numpy as np
import pylab as pl

import pandas
from croaks import DataProxy
import saunerie.linearmodels as lm
from saunerie import bspline


def convert(fn='data_fits_f_10.h5', ofn=None):
    """
    """
    data = pandas.read_hdf(fn)
    #    expnums = np.array([int(expn.split('_')[1]) for expn in data.index.get_level_values(0)])
    #    mapping = {v:i for i, v in enumerate(data.index.get_level_values(0).unique())}
    mapping = {v:int(v.split('_')[1]) for v in data.index.get_level_values(0).unique()}
    data = data.rename(index=mapping, level=0)
    data = data[["qid", "ccdid", "x", "y", "u", "v", "ra", "dec", "f_10", "f_10_e", "f_10_ratio", "f_10_f", "psfcat", "psfcat_e", "psfcat_ratio", "isolated", "colormag"]].reset_index().rename({"level_0":"img_id"}, axis=1)
    d = data.to_records(column_dtypes={'qid':np.int, 'isolated': np.int})
    if ofn is not None:
        np.save(ofn, d)
    return d


def convert_psf(fn='data_fits_psfcat.h5', ofn=None):
    """
    """
    data = pandas.read_hdf(fn)
    #    expnums = np.array([int(expn.split('_')[1]) for expn in data.index.get_level_values(0)])
    #    mapping = {v:i for i, v in enumerate(data.index.get_level_values(0).unique())}
    mapping = {v:int(v.split('_')[1]) for v in data.index.get_level_values(0).unique()}
    data = data.rename(index=mapping, level=0)
    data = data[["qid", "ccdid", "x", "y", "u", "v", "ra", "dec", "psfcat", "psfcat_e", "psfcat_ratio", "isolated", "colormag"]].reset_index().rename({"level_0":"img_id"}, axis=1)
    d = data.to_records(column_dtypes={'qid':np.int, 'isolated': np.int})
    if ofn is not None:
        np.save(ofn, d)
    return d


def get_mjd(exp_id):
    """
    """
    from astropy.time import Time
    import datetime
    
    # let's decompose the encoding of the exp_id
    # this should correspond to the date in UTC
    # (UTC is what has replaced the astronomical GMT)
    yyyy = np.floor(exp_id * 1.E-10).astype(int)
    mm   = np.floor(exp_id * 1.E-8 - yyyy*100).astype(int)
    dd   = np.floor(exp_id * 1.E-6 - yyyy*10000 - mm*100).astype(int)

    # c'est assez moche - je n'ai pas de meilleure idee pour le moment
    mjd = Time([datetime.datetime(yyyy[i], mm[i], dd[i]) \
                for i in range(len(yyyy))]).mjd 
    
    # this is the remainder, is days since midnight UTC
    mjd += (exp_id - yyyy*1E10 - mm*1E8 - dd*1.E6) * 1.E-6
    
    return mjd


def load(fn='ztfstarflat.npy', mag_cuts=False, psf_flux=False):
    """
    load and clean dataset 
    """
    logging.info('load and clean data')
    
    d = np.load(fn, allow_pickle=True)
    if not psf_flux:
        mag = -2.5 * np.log10(d['f_10'])
        emag = (2.5 / np.log(10)) * d['f_10_e'] / d['f_10']
    else:
        mag = -2.5 * np.log10(d['psfcat'])
        emag = (2.5 / np.log(10)) * d['psfcat_e'] / d['psfcat']
        
    #    ok = np.isfinite(mag) & np.isfinite(emag) & (~np.isnan(mag)) & (~np.isnan(emag)) & (d['f_10']>1000) & (d['f_10_f'] == 0) & (d['isolated'] == 1)
    ok = (d['isolated'] == 1) & (~np.isnan(mag)) & (~np.isnan(emag)) & np.isfinite(emag) & np.isfinite(mag)
    if not psf_flux and 'f_10_f' in d.dtype.names:
        ok &= (d['f_10_f'] == 0)
    ok &= mag<-6
    #    ok = (d['f_10_f'] == 0) & (d['isolated'] == 1) & (~np.isnan(mag)) & (~np.isnan(emag)) & np.isfinite(emag) & np.isfinite(mag)
    
    logging.info('{}/{} measurements discarded (flux/mag/emag cuts)'.format((~ok).sum(), len(ok)))

    mjd = get_mjd(d[ok]['img_id'])
    
    dp = DataProxy(d[ok],
                   exp='img_id', ccd='ccdid', qid='qid', star='Source',
                   x='x', y='y', u='u', v='v', ra='ra', dec='dec',
                   colormag='colormag')
    dp.colormag -= dp.colormag.mean()
    dp.add_field('mag', mag[ok])
    dp.add_field('emag', np.sqrt(emag[ok]**2 + 0.005**2))
    dp.add_field('mjd', mjd)
    dp.make_index('star')
    dp.make_index('exp')
    dp.make_index('mjd')
    
    return dp


def plot_zp_vs_exp(dp, solver, model):
    """
    """
    pl.figure()
    if 'zp' not in model.params._struct:
        print('no ZP in model')
    pl.plot(dp.exp_set, model.params['zp'].full, 'b.')
    pl.xlabel('Exp ID')
    pl.ylabel('ZP')


def plot_zp_vs_mjd(dp, solver, model):
    """
    """
    d = np.unique(np.rec.fromarrays((dp.mjd, dp.exp_index),
                  names=['mjd', 'exp_index']))
    pl.figure()
    if 'zp' not in model.params._struct:
        print('no ZP in model')
    pl.plot(d['mjd'], model.params['zp'].full[d['exp_index']], 'b.')
    pl.xlabel('MJD')
    pl.ylabel('ZP')
    


def pixellize(dp, nx, ny, Nx=3100, Ny=3100):
    """
    Superpixellize the focal plane, and assign a cell to each star.
    
    It looks to me that ZTF quadrants are something like 3080x3080 
    pixels wide
    """
    # define slightly wider bins
    xbins = np.linspace(-10., Nx, nx+1)
    ybins = np.linspace(-10., Ny, ny+1)

    # bin centers
    xc = 0.5 * (xbins[1:] + xbins[:-1])
    yc = 0.5 * (ybins[1:] + ybins[:-1])
    
    # bin coordinates per star
    ix = np.digitize(dp.x, xbins)-1
    iy = np.digitize(dp.y, ybins)-1
    
    # detect the stars which are not assigned to a valid bin
    valid = (ix>=0) & (ix<nx) & (iy>=0) & (iy<ny)
    logging.info('{} stars in non-valid bins'.format((~valid).sum()))
    
    quad = dp.ccd*10 + dp.qid
    cell = iy*nx + ix + quad * 1000
    
    dp.add_field('ix', ix)
    dp.add_field('iy', iy)

    dp.add_field('xc', xc[ix])
    dp.add_field('yc', yc[iy])
    
    dp.add_field('quad', quad)
    dp.make_index('quad')
    
    dp.add_field('cell', cell)
    dp.make_index('cell')
    
    dp.add_field('valid', valid)

    # build a map, that allows to associate
    # the cell index with the position of the cell
    # in the quad 
    cell_map = []
    for ccd in range(1,17):
        for qid in range(1,5):
            for ix in range(nx):
                for iy in range(ny):
                    quad = ccd*10 + qid
                    cell = iy*nx + ix + quad * 1000
                    cell_map.append((ccd, qid, quad, cell, dp.cell_map[cell], ix, iy, xc[ix], yc[iy], 0., 0.))
                    
    cell_map = np.rec.fromrecords(cell_map, names=['ccd', 'qid', 'quad', 'cell', 'icell', 'ix', 'iy', 'xc', 'yc', 'dzp', 'edzp'])
    
    return dp, cell_map


def starflat_model_dzp(dp, fixed_cell=71515):
    """
    """
    mstar = lm.indic(dp.star_index, name='mstar', valid=dp.valid)
    dzp = lm.indic(dp.cell_index, name='dzp', valid=dp.valid)
    model = mstar + dzp

    model.params['dzp'].fix(dp.cell_map[fixed_cell])
    
    return model


def starflat_model_zpexp(dp, fixed_cell=71515):
    """
    """
    mstar = lm.indic(dp.star_index, name='mstar', valid=dp.valid)
    dzp = lm.indic(dp.cell_index, name='dzp', valid=dp.valid)
    zp = lm.indic(dp.exp_index, name='zp', valid=dp.valid)
    model = mstar + dzp + zp 

    
    model.params['dzp'].fix(dp.cell_map[fixed_cell])
    model.params['zp'].fix(0)
    
    return model


def starflat_model_zpexpspline(dp, fixed_cell=71515, nb_nodes=5):
    """
    """
    mstar = lm.indic(dp.star_index, name='mstar', valid=dp.valid)
    dzp = lm.indic(dp.cell_index, name='dzp', valid=dp.valid)

    # smooth zp 
    mjd_mean = dp.mjd.mean()
    mjd_grid = np.linspace(dp.mjd.min() - mjd_mean,
                           dp.mjd.max() - mjd_mean,
                           nb_nodes)
    basis = bspline.BSpline(mjd_grid)
    J = basis.eval(dp.mjd - mjd_mean)
    zp = lm.LinearModel(J.row, J.col, J.data, name='zp')
    
    model = mstar + dzp + zp 

    model.params['dzp'].fix(dp.cell_map[fixed_cell])
    model.params['zp'].fix(0)
    
    return model
    


def starflat_model_dk_zpexp(dp, fixed_cell=71515):
    """
    """
    mstar = lm.indic(dp.star_index, name='mstar', valid=dp.valid)
    dzp = lm.indic(dp.cell_index, name='dzp', valid=dp.valid)
    zp = lm.indic(dp.exp_index, name='zp', valid=dp.valid)
    dk = lm.indic(dp.cell_index, name='dk', val=dp.colormag, valid=dp.valid)
    model = mstar + dzp + dk + zp 

    model.params['dzp'].fix(dp.cell_map[fixed_cell])
    model.params['zp'].fix(0)
    model.params['dk'].fix(dp.cell_map[fixed_cell])
    
    return model


def starflat_model_dzpquad_dzpcell_zpexp(dp):
    """
    """
    mstar = lm.indic(dp.star_index, name='mstar', valid=dp.valid)
    dzp_cell = lm.indic(dp.cell_index, name='dzp_cell', valid=dp.valid)
    dzp_quad = lm.indic(dp.quad_index, name='dzp_quad', valid=dp.valid)
    zp = lm.indic(dp.exp_index, name='zp', valid=dp.valid)
    model = mstar + dzp_cell + dzp_quad + zp

    # fix 1 quad
    model.params['dzp_quad'].fix(27)
    
    # fix 1 central cell / quadrant
    nx, ny = dp.ix.max()+1, dp.iy.max()+1
    offset = int(nx*ny/2)
    ifix = np.arange(offset, len(model.params['dzp_cell'].full), nx*ny).astype(int)
    model.params['dzp_cell'].fix(ifix)

    # fix 1 zp 
    model.params['zp'].fix(0)
    
    return model
    



def fit(dp, model, nsig=4.5):
    solver = lm.RobustLinearSolver(model, dp.mag, weights = 1./dp.emag, ordering_method='metis')
    x = solver.robust_solution(nsig=nsig, local_param='mstar')
    model.params.free = x
    return dp, model, solver, x


def plot_measurements(dp, model, solver):
    """
    measurement/outlier density on the focal plane
    """
    ok = ~solver.bads
    
    pl.figure(figsize=(10,8))
    pl.title('outliers')
    pl.hexbin(dp.u[~ok], dp.v[~ok])
    pl.xlabel('U')
    pl.ylabel('V')    
    pl.colorbar()

    pl.figure(figsize=(10,8))
    pl.title('good measurements')    
    pl.hexbin(dp.u[ok], dp.v[ok])
    pl.colorbar()
    pl.xlabel('U')
    pl.ylabel('V')    
    

    
def plot_model(dp, model, solver, ccd=None, subtract_gains=False, vmin=-0.1, vmax=0.1, hexbin=False):
    """
    model the spatial grid variations
    """

    pars = model.params.copy()
    #    pars['mstar'].full[:] = 0.
    pars.full[:] = 0.
    pars['dzp'].full[:] = model.params['dzp'].full[:]
    #    pars['dk'].full[:] = model.params['dk'].full[:]
    # evaluate the model
    v = model(pars.full)
    ok = ~solver.bads
    r_val, r_gains = None, None
    
    if ccd is not None:
        ok &= (dp.ccd == ccd)

    def add_labels():
        pl.xlabel('U')
        pl.ylabel('V')    
        pl.colorbar()
        if ccd is None:
            pl.title('Starflat')
        else:
            pl.title('Starflat ccd={}'.format(ccd))        
        
    if subtract_gains:
        uu, vv, val, gains = [], [], [], []
        for q in dp.quad_set:
            idx = (dp.quad == q) & ok
            m = np.median(v[idx])
            uu.append(dp.u[idx])
            vv.append(dp.v[idx])
            val.append(v[idx] - m)
            gains.append(np.full(idx.sum(), m))
        if hexbin:
            pl.figure(figsize=(10,8))
            pl.hexbin(np.hstack(uu), np.hstack(vv), C=np.hstack(val))
            add_labels()
            #        pl.scatter(np.hstack(uu), np.hstack(vv), c=np.hstack(val), s=0.05)
        else:
            pl.figure(figsize=(10,8))
            h, _, _ = np.histogram2d(np.hstack(vv), np.hstack(uu), weights=np.hstack(val), bins=[1000,1000])
            c, _, _ = np.histogram2d(np.hstack(vv), np.hstack(uu), bins=[1000,1000])
            r = h/c
            r[np.isnan(r)] = 0.
            pl.imshow(-r[::1, ::1], vmin=vmin, vmax=vmax, origin='lower')
            add_labels()
            r_val = r

            pl.figure(figsize=(10,8))
            h, _, _ = np.histogram2d(np.hstack(uu), np.hstack(vv), weights=np.hstack(gains), bins=[1000,1000])
            c, _, _ = np.histogram2d(np.hstack(uu), np.hstack(vv), bins=[1000,1000])
            r = h/c
            r[np.isnan(r)] = 0.
            pl.imshow(r[::1, ::1], origin='lower')
            add_labels()
            r_gains = r
            
    else:
        if hexbin:
            pl.figure(figsize=(10,8))
            pl.hexbin(dp.u[ok], dp.v[ok], C=v[ok], vmin=vmin, vmax=vmax)
            add_labels()
        else:
            pl.figure(figsize=(10,8))
            h, _, _ = np.histogram2d(dp.u[ok], dp.v[ok], weights=v[ok], bins=[500,500])
            c, _, _ = np.histogram2d(dp.u[ok], dp.v[ok], bins=[500,500])
            r = (h/c).T
            r[np.isnan(r)] = 0.
            pl.imshow(r[::1, ::1], origin='lower', vmin=vmin, vmax=vmax)
            add_labels()
            r_val = r
    
    return r_val, r_gains

#        pl.xlabel('U')
#        pl.ylabel('V')    
#        pl.colorbar()
#        if ccd is None:
#            pl.title('Starflat')
#        else:
#            pl.title('Starflat ccd={}'.format(ccd))

        
    

# def plot_residuals(dp, model, solver):
#     """
#     mde
#     """
#     v = model()
#     ok = ~solver.bads
#     r = dp.mag - v
    
#     pl.figure(figsize=(10,8))
#     pl.title('Starflat')
#     val = pl.hexbin(dp.u[ok], dp.v[ok], C=r[ok])
    
#     pl.xlabel('U')
#     pl.ylabel('V')    
#     pl.colorbar()    

    
#     pl.figure(figsize=(10,10))
#     pl.hist(r[ok]/dp.emag[ok], bins=1000)
#     pl.xlabel(residuals)
    
    
def plot_residuals(dp, model, solver):
    """
    mde
    """
    v = model()
    ok = ~solver.bads
    r = dp.mag - v
    
    pl.figure(figsize=(10,8))
    pl.title('Starflat residuals')
    val = pl.hexbin(dp.u[ok], dp.v[ok], C=r[ok], vmin=-0.01, vmax=0.01)
    pl.xlabel('U')
    pl.ylabel('V')    
    pl.colorbar()    

    pl.figure(figsize=(10,8))
    pl.title('Starflat residuals [weighted]')
    val = pl.hexbin(dp.u[ok], dp.v[ok], C=r[ok]/dp.emag[ok])    
    pl.xlabel('U')
    pl.ylabel('V')    
    pl.colorbar()    

    pl.figure(figsize=(10,8))
    pl.title('Starflat residuals [partial $\chi^2$]')
    val = pl.hexbin(dp.u[ok], dp.v[ok], C=(r[ok]/dp.emag[ok])**2, vmin=0, vmax=3)    
    pl.xlabel('U')
    pl.ylabel('V')    
    pl.colorbar()    

    pl.figure(figsize=(10,10))
    okok = ok & (dp.mag<-10)
    pl.hist(r[okok]/dp.emag[okok], bins=1000)
    pl.xlabel('residuals')

    pl.figure(figsize=(10,10))
    okok = ok & (dp.mag<-10)
    pl.hist(r[okok], bins='auto')
    pl.xlabel('residuals')

    pl.figure(figsize=(10,10))
    pl.plot(dp.mag[ok], r[ok]/dp.emag[ok], 'k,')
    pl.xlabel('mag')
    pl.ylabel('residuals [weighted]')
    pl.title('weighted residuals')

    pl.figure(figsize=(10,10))
    pl.plot(dp.mag[ok], r[ok], 'k,')
    pl.xlabel('mag')
    pl.ylabel('residuals [unweighted]')    
    pl.title('unweighted residuals')    
    


def cell_index_to_xy(dp, pars):
    """
    slow method
    """
    
    ncells = len(dp.cell_set)
    l = []
    for cell_index in range(ncells):
        if cell_index%100 == 0:
            print(' (*) cell_index: ', cell_index)
        idx = dp.cell_index == cell_index
        xx = dp.x[idx].mean()
        yy = dp.y[idx].mean()
        uu = dp.u[idx].mean()
        vv = dp.v[idx].mean()
        ccd = dp.ccd[idx].mean()
        assert(dp.ccd[idx].std() == 0)
        quad = dp.quad[idx].mean()
        assert(dp.quad[idx].std() == 0)
        dzp = pars['dzp'].full[cell_index]
        l.append((cell_index, ccd, quad, 
                  xx, yy, uu, vv,
                  dzp, 0.))
    d = np.rec.fromrecords(l, names=['icell', 'ccd', 'quad',
                                     'x', 'y', 'u', 'v', 'dzp', 'gain_offset'])

    for quad in dp.quad_set:
        idx = d['quad'] == quad
        offset = np.median(d['dzp'][idx])
        d['gain_offset'][idx] = offset
    
    return d


def pars_to_starflats(m, pars, plot=False):
    """
    """
    dzp = pars['dzp'].full[m.icell]
    m['dzp'] = dzp

    mm = {}
    for d in m:
        mm[d['cell']] = d['dzp']

    if plot:
        for quad in np.unique(m.quad):
            idx = m.quad == quad
            offset = np.median(dzp[idx])
            pl.figure()
            pl.scatter(m.xc[idx], m.yc[idx], c=dzp[idx]-offset, s=50, vmin=-0.01, vmax=0.01)
            pl.xlabel('x [pixels]')
            pl.ylabel('y [pixels]')
            pl.title('CCD {} QUAD: {}'.format(m.ccd[idx][0], m.qid[idx][0]))
            pl.colorbar()


    return m, mm
    
    
def main(filename='zg_2019.npy', control_plots=False, nx=30, ny=30):
    """
    """
    dp = load(filename)
    dp, m = pixellize(dp, nx, ny)
    model = starflat_model_zpexp(dp, fixed_cell=71013)
    #    model = starflat_model_dzp(dp)
    #    model = starflat_model_zpexpspline(dp)
    _,_,solver,x = fit(dp, model)
    _, mm = pars_to_starflats(m, model.params, plot=control_plots)

    # ici, la structure m contient tout ce dont tu as besoin
    # Pour interpoler.

    return dp, solver, model, m, mm


def fit_test_model(filename, grid_map, control_plots=False, nsig=5.):
    """
    """
    dp = load(filename)
    dp, m = pixellize(dp, 30, 30)

    # c'est la que ca ne va pas.
    # l'indexation peut varier d'un dataset a l'autre
    #    dzp = grid_pars['dzp'].full[dp.cell_index]
    dzp = np.array([grid_map[c] for c in dp.cell])
    dp.mag -= dzp

    model = starflat_model_zpexp(dp)
    #    model = starflat_model_dzp(dp)    
    _,_,solver,x = fit(dp, model, nsig=nsig)

    _, mm = pars_to_starflats(m, model.params, plot=control_plots)

    return dp, solver, model, m, mm


    
