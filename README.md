# GraphCast Precipitation Forecasting for Bangladesh

This repository contains the complete workflow for evaluating GraphCast model's performance in precipitation forecasting over Bangladesh, compared against traditional physics-based models (ECMWF and GFS/NCEP). This work was developed as part of a bachelor's thesis project.

## Overview

GraphCast is a machine learning-based weather forecasting model developed by Google DeepMind. This project assesses its capability in predicting precipitation patterns specifically for Bangladesh and benchmarks its performance against established numerical weather prediction models.

## Repository Structure

### Data Download Scripts
- **`Era5_data_download.ipynb`** - Downloads ERA5 reanalysis data, which serves as ground truth/observational data for model validation
- **`ecmwf+ncep_forecast_download.ipynb`** - Downloads control forecast data from ECMWF and NCEP (GFS) models for comparison

### Data Processing Scripts
- **`accumulate_6h_precip.py`** - Converts hourly precipitation data to 6-hourly accumulated precipitation
- **`pre-processing.ipynb`** - Prepares the downloaded data into the format required by the GraphCast model

### Model Execution
- **`GraphCast_model_run.py`** - Runs the GraphCast model to generate precipitation forecasts

### Output Processing
- **`Post-processing.ipynb`** - Processes and analyzes the model output for evaluation and visualization

## Workflow

1. **Data Acquisition**
   - Run `Era5_data_download.ipynb` to obtain ERA5 reanalysis data
   - Run `ecmwf+ncep_forecast_download.ipynb` to download ECMWF and NCEP forecast data

2. **Data Preparation**
   - Use `accumulate_6h_precip.py` to aggregate hourly precipitation to 6-hourly totals
   - Execute `pre-processing.ipynb` to format data for GraphCast compatibility

3. **Model Execution**
   - Run `GraphCast_model_run.py` to generate GraphCast precipitation forecasts

4. **Analysis**
   - Use `Post-processing.ipynb` to analyze results and compare model performances

## Research Focus

This project evaluates:
- GraphCast model's precipitation forecasting skill over Bangladesh
- Comparative performance against ECMWF and GFS models
- Machine learning vs. physics-based approaches for regional precipitation prediction


## Citation

If you use this work, please cite appropriately and acknowledge the original GraphCast model by Google DeepMind.


## Contact

For questions or collaboration opportunities, please open an issue in this repository.
