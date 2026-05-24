# AeroCalib

Code accompanying the paper

> M. S. Sadak. *Data-Efficient Calibration Transfer Across Operational
> Weather Regimes for Probabilistic Airport Hazard Prediction.*
> International Journal of Data Science and Analytics, under revision (2026).

The scripts train XGBoost models for four airport hazards (low
visibility, low ceiling, crosswind above 15 kt, and thunderstorm) on
15 years of METAR data from 51 airports, then study how the predicted
probabilities behave when the model is applied outside its training
regime.

## Repository layout

```
config.py                     constants, task definitions, helpers
01_download_data.py           pull METAR from IEM
02_preprocess_data.py         build hourly feature matrix and labels
03_train_evaluate.py          train XGBoost, in-domain evaluation
04_baselines.py               climatology, persistence, LR, RF, XGB
05_calibration.py             Platt scaling, in-domain ECE / NLL / slope
06_airport_heldout.py         leave-9-airports-out generalization
07_t2_ablation.py             T2 label leakage check
08_dl_comparison.py           LSTM and Transformer baselines
09_cost_benefit.py            cost-optimal thresholds
10_transfer_learning.py       continued boosting across zones
11_nwp_proxy.py               METAR vs METAR+ERA5 (sensitivity)
13_calibration_transfer.py    main 2x2 ablation, 120 transfer-task runs
13_ece_sensitivity.py         ECE bin-count sensitivity
14_regime_prevalence.py       per-regime hazard prevalence table
15_brier_decomposition.py     Brier reliability / resolution split
16_koppen_crosswalk.py        Koppen-Geiger overlap with regimes
17_chronological_adaptation.py    rolling-origin calibration build-up
18_chrono_to_latex.py         tabulate output of 17
19_temp_vs_platt.py           one-parameter temperature scaling head-to-head
20_regime_input_stats.py      per-regime feature distribution summary
21_airport_level_recal.py     Platt recalibration at individual airports
22_holistic_eval.py           ridge LR baseline, F1, PR-AUC, variable importance
```

Each script reads from `data/processed/` and writes to `results/`.

## Requirements

Python 3.10 or newer. See `requirements.txt`. The main dependencies are
`numpy`, `pandas`, `scikit-learn`, `xgboost`, `scipy`, and (for
script 08) `torch`.

```
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to reproduce the paper

The temporal split is fixed in `config.py`: 2010-2020 for training,
2021-2022 for validation and calibration, 2023-2024 for testing.

Run the scripts in order. Steps 01 and 02 produce the data files used
by everything else, so they need to run first. The rest are independent
of each other and can be run in any order.

```
python 01_download_data.py
python 02_preprocess_data.py
python 03_train_evaluate.py
python 04_baselines.py
python 05_calibration.py
python 06_airport_heldout.py
python 07_t2_ablation.py
python 08_dl_comparison.py            # needs PyTorch; can be skipped
python 09_cost_benefit.py
python 10_transfer_learning.py
python 11_nwp_proxy.py
python 13_calibration_transfer.py     # main experiment, ~6 hours on a 32-GB workstation
python 13_ece_sensitivity.py
python 14_regime_prevalence.py
python 15_brier_decomposition.py
python 16_koppen_crosswalk.py
python 17_chronological_adaptation.py
python 18_chrono_to_latex.py
python 19_temp_vs_platt.py
python 20_regime_input_stats.py
python 21_airport_level_recal.py
python 22_holistic_eval.py
```

Outputs land in `results/` as JSON or CSV. The mapping between scripts
and tables in the paper is documented in the `output_file` field of each
JSON.

## Data

METAR data comes from the Iowa Environmental Mesonet:
<https://mesonet.agron.iastate.edu/request/download.phtml>.
The list of 51 airports is in `config.py` under `AIRPORTS`. The
download step calls the IEM endpoint directly; running the full
download once takes about half an hour and writes about 1.8 GB of
CSV to `data/raw/`.

The preprocessed parquet file (`data/processed/aerocalib_hourly.parquet`)
is roughly 1.5 GB. It is not included in the repository for size
reasons. Step 02 builds it from the raw CSVs.

## License

MIT. See `LICENSE`.

## Citation

If you use this code, please cite the paper:

```
@article{sadak2026aerocalib,
  author  = {Sadak, Mustafa Semih},
  title   = {Data-Efficient Calibration Transfer Across Operational
             Weather Regimes for Probabilistic Airport Hazard Prediction},
  journal = {International Journal of Data Science and Analytics},
  year    = {2026},
  note    = {Under revision}
}
```
