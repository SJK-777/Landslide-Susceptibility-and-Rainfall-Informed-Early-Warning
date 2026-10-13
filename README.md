# India Landslide Watch — Streamlit Community Cloud

## What this prototype does
- Fits a Random Forest classifier from the provided historical susceptibility training CSV.
- Lets the user select a coordinate and edit predictor values.
- Fetches forecast precipitation from Open-Meteo for that coordinate.
- Displays a separate, explicitly provisional rainfall screening advisory.
- Explains the limits of the historical event/background model.

## Important scientific limitations
This is a research prototype, not an official warning service. Background samples are not confirmed non-landslide observations. The model score is not a calibrated real-world landslide probability. The rainfall bands are provisional interface rules, not validated thresholds. The current training dataset uses static terrain/land-cover/NDVI features; forecast rainfall is shown separately and is NOT fed into the fitted susceptibility model. Do not use this app for evacuation or life-safety decisions.

## Files to add
Place `app.py`, `requirements.txt`, and the file below in the repository root:
`INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv`

Use the existing file from:
`/content/drive/MyDrive/GROUND-R/NILGIRIS_PATH_B_REAL/FINAL_TRAINING_DATASETS/data/INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv`

Do not upload private credentials. The forecast service is called without an API key.

## Deploy
1. Upload the three files to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set main file path to `app.py`.
4. Deploy. First startup can take several minutes because the model is trained from the CSV.
5. Test a known coordinate and verify the displayed forecast dates and units.

## What is not included
A nationwide susceptibility raster/map is not generated here because a verified nationwide raster stack was not available in the project context. The coordinate prediction uses user-entered/default predictor values, not automatic raster extraction at that coordinate. For a scientifically defensible operational early-warning service, add nationwide predictor rasters and validated event-aligned rainfall thresholds, then independently validate the full warning logic.
