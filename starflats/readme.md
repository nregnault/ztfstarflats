# ZTF starflat toolkit

## Preparing the dataset
The starflat fitting software `starflat` expect one dataset per sequence, as a DataFrame saved in the parquet format. Each row describe one measurement, that is, a star detection (or any point like sources). For each measurement, apperture and PSF photometry, GaiaID and photometry, quadrant/CCD informations are expected, as well as position in the quadrant frame.

Columns are as follows:
```
gaiaid x y ra dec psfflux epsfflux G BP RP eG eBP eRP mjd qid ccdid filtercode quadrant seeing apfl0 eapfl0 rad0 apfl1 eapfl1 rad1 apfl2 eapfl2 rad2 apfl3 eapfl3 rad3 apfl4 eapfl4 rad4 apfl5 eapfl5 rad5 apfl6 eapfl6 rad6 apfl7 eapfl7 rad7 apfl8 eapfl8 rad8 apfl9 eapfl9 rad9
```

### `extract_measures`
The ZTF Scene Modeling Photometry (SMP) pipeline can be used to get such a sequence dataset, using the `concat_catalog` operation. This gives a dataset, per `rcid`, concatening all relevant catalogs, i.e. aperture and PSF photometry, external catalogs (PS1, Gaia) and most quadrant header informations. Currently, starflat photometry measurements, implemented using the SMP pipeline, process starflats per year. Since there a years with several starflats, a tool has been written to split these big 1 year datasets into individual starflat sequences.

```
> extract_measures --help
usage: extract_measures [-h] --dataset-path DATASET_PATH --output OUTPUT [--year YEAR] [--filtercode {zg,zr,zi}]
                        [--min-measure-count MIN_MEASURE_COUNT] [--max-g-mag MAX_G_MAG]

Extract relevant starflat measure fields from measurement datasets.

optional arguments:
  -h, --help            show this help message and exit
  --dataset-path DATASET_PATH
                        Path where all rcid datasets are saved.
  --output OUTPUT       Output folder where each sequences get stored.
  --year YEAR           Which year to process.
  --filtercode {zg,zr,zi}
                        Which filter to process
  --min-measure-count MIN_MEASURE_COUNT
                        Filter out stars having less than a certain amount of detections/measurements.
  --max-g-mag MAX_G_MAG
                        Filter out stars having an Gaia G band magnitude higher than the set amount.
```
## Fitting starflats
Given a starflat sequence dataset, a starflat model can be fit.
```
> starflat --help
usage: starflat [-h] [--dataset-path DATASET_PATH] [--config-path CONFIG_PATH] [--output-path OUTPUT_PATH] [--model {simple,zp,color,full}]
                [--list-models] [--plot] [--solve]

Starflat model solver.

optional arguments:
  -h, --help            show this help message and exit
  --dataset-path DATASET_PATH
                        Starflat sequence dataset path.
  --config-path CONFIG_PATH
                        Configuration file path.
  --output-path OUTPUT_PATH
                        Output path where model solution and plot are saved.
  --model {simple,zp,color,full}
                        Model to solve.
  --list-models         List all implemented models.
  --plot                Generate control plots.
  --solve               Solve the model.
```

### Starflat models
Implemented models list:
```
> starflat --list-model
============================== simple ==============================
Simplest starflat model. Fit star magnitude and superpixelized ZP over the focal plane.
$m_\mathrm{ADU}=m_s+\delta ZP(u, v)$

============================== zp ==============================
ZP starflat model. Fit star magnitude, superpixelized ZP and ZP wrt mjd.
$m_\mathrm{ADU}=m_s+\delta ZP(u, v) + ZP(mjd)$

============================== color ==============================
Color starflat model. Fit star magnitude and superpixelized ZP, centered color term over the focal plane, ZP wrt mjd.
$m_\mathrm{ADU}^{s,q}=m^s+\delta ZP(u, v) + ZP^q + \delta k(u, v) C_{Bp-Rp}^s$

============================== full ==============================
Full starflat model. Fit star magnitude and superpixelized ZP, color term over the focal plane, ZP wrt mjd and differential airmass term.
$m_\mathrm{ADU}=m_s+\delta ZP(u, v) + ZP(mjd) + \delta k C_{Bp-Rp} + k(X(u, v)-1)$
```

### Configuration file
Configuration file example, in yaml format:
```
photometry: psf
piedestal: 0.005
zp_resolution: 15
color_resolution: 2
solve_method: cholesky
flip_flop_max_iter: 4
eq_constraints: false
```

Available photometry are `psf` and apperture `apflX` with `X` a number between 0 and 9 (smaller to bigger apperture).

2 solve method are implemented, `cholesky` and `cholesky_flip_flop`.
