

import setuptools
from setuptools import setup, Extension

import os
import subprocess
import glob
selinv = glob.glob(os.path.join('selinv','*.f'))

def fortran_objects(l):
    for n in l:
        subprocess.check_call(['gfortran', '-O3', '-fPIC',
                               '-c', n,
                               '-o', n.replace('.f', '.o')])
    return [n.replace('.f', '.o') for n in l]


extensions = [
    Extension('starflats.libselinv',
              define_macros = [('MAJOR_VERSION', '1'),
                               ('MINOR_VERSION', '0')],
              include_dirs = [],
              libraries = ['lapack', 'blas', 'm', 'gfortran'],
              sources = ['selinv/wrapselinv.c', 'selinv/C2Finterface.c'],#+ selinv,
              extra_compile_args = ['-g'],
              #extra_f90_compile_args = ['-03'],
              #f2py_options = ['--no-wrap-functions'],
              #fortran_compiler = 'gfortran',
              extra_objects=fortran_objects(selinv),
              )
]


setup(
    ext_modules=extensions,
)
