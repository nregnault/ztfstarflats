#!/usr/bin/env bash

# $1 dataset path
# $2 config path
# $3 output folder

echo "PSF photometry"
# bash starflat_all_models.sh $1 $2 $3 "psf"

echo "Aper4 photometry"
bash starflat_all_models.sh $1 $2 $3 "apfl4"

echo "Aper6 photometry"
bash starflat_all_models.sh $1 $2 $3 "apfl7"
