#!/usr/bin/env bash

# $1 dataset path
# $2 config path
# $3 output folder
# $4 photometry

dataset_name=`echo $1 | awk -F '.' -e '{print $1}' | awk -F '_' -e '{print $3}'`
mkdir -p $3/$dataset_name

mkdir -p $3/$dataset_name/$4
mkdir -p $3/$dataset_name/$4/simple
mkdir -p $3/$dataset_name/$4/zp
mkdir -p $3/$dataset_name/$4/color
mkdir -p $3/$dataset_name/$4/full

echo "Simple model"
starflat --dataset-path=$1 --config-path=$2/config_$4.yaml --output-path=$3/$dataset_name/$4/simple --model=simple --solve --plot

echo "ZP model"
starflat --dataset-path=$1 --config-path=$2/config_$4.yaml --output-path=$3/$dataset_name/$4/zp --model=zp --solve --plot

echo "Color model"
starflat --dataset-path=$1 --config-path=$2/config_$4.yaml --output-path=$3/$dataset_name/$4/color --model=color --solve --plot

echo "Full model"
starflat --dataset-path=$1 --config-path=$2/config_$4.yaml --output-path=$3/$dataset_name/$4/full --model=full --solve --plot
