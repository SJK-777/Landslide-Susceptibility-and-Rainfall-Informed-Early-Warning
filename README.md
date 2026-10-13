# India Landslide Watch — Streamlit Community Cloud

## Files
- `app.py`
- `requirements.txt`
- `INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv` (required unless uploaded in the sidebar)

## Expected CSV
Use the existing project file:
`FINAL_TRAINING_DATASETS/data/INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv`

The app expects a binary `label` field containing both 1 (recorded event) and 0 (sampled background), plus environmental predictor columns.

## Deploy
1. Upload `app.py`, `requirements.txt`, and the training CSV to the root of a GitHub repository.
2. Open https://share.streamlit.io/
3. Create app, select the repository/branch, and set main file to `app.py`.
4. Deploy and inspect logs if the first startup fails.

## Scientific limitations
The app fixes the ambiguous-NA bug by explicit null-safe label parsing and avoids `pd.NA` truth testing. It fits a new Random Forest from the supplied CSV at startup; this is not the same as loading the exact prior fitted model. Coordinate inputs do not automatically retrieve raster predictors: users must enter predictor values, and defaults are training medians/modes. Forecast rainfall is shown separately and is not an input to the classifier. Rainfall bands are illustrative and unvalidated. Background samples are not verified non-landslide observations. Do not use for real-world warnings or evacuation decisions.

