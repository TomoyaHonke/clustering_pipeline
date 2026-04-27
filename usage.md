## Config

### CorrfuncConfig

```python
cfg = CorrfuncConfig(
    nthreads=56,
    mu_max=1.0,
    nmu_bins=20,
    binfile=np.linspace(20, 200, 46),
    mode="radec",   # "radec" or "xyz"
)
```

- nthreads : number of threads  
- mu_max : maximum μ  
- nmu_bins : number of μ bins  
- binfile : s bin edges [Mpc/h]  
- mode : coordinate system ("radec" or "xyz")

---

### AnalysisConfig

```python
acfg = AnalysisConfig(
    zbins={
        "z1": (0.8, 1.1),
        "z2": (1.1, 1.6),
    },
    regions=("A", "B"),
    combine_regions=True,
    use_jackknife=False,
    n_jack=64,
)
```

- zbins : dictionary of redshift bins  
  - "z1": 0.8 < z < 1.1  
  - "z2": 1.1 < z < 1.6  
- regions : arbitrary dataset splits  
- combine_regions :  
  - True → combine all regions  
  - False → return per-region results  
- use_jackknife : enable jackknife error  
- n_jack : number of jackknife regions

## Input

Provide data and random catalogs as dictionaries keyed by region names.

```python
data_path = {
    "A": "/path/to/data_A.fits",
    "B": "/path/to/data_B.fits",
}

random_path = {
    "A": "/path/to/random_A.fits",
    "B": "/path/to/random_B.fits",
}
```

- Keys must match `AnalysisConfig.regions`
- Each file should contain:
  - RA
  - DEC
  - redshift (z)
  - weight

## Run

Execute the pipeline as follows:

```python
results = compute_monopole(
    data_path,
    random_path,
    cosmo,
    cfg,
    acfg,
)
```

## Notes

- - Distances are expressed in units of [Mpc/h].

- The keys of `data_path` and `random_path` must match `AnalysisConfig.regions`.

- When `combine_regions=True`, all regions are combined into a single result.
  When `False`, results are returned separately for each region.

- The output format depends on the configuration:
  - With jackknife: `(s, xi0, err)`
  - Without jackknife: `(s, xi0)`

- The binning in `binfile` defines the separation bins used in the analysis.
  The output `s` corresponds to the bin centers.

- The estimator for `xi0` is computed using the internal pipeline (Corrfunc-based),
  while the error (if enabled) is computed independently via jackknife.

- Jackknife error estimation is currently supported only for `mode="xyz"`.
- `mode="radec"` is not supported for jackknife at this stage.