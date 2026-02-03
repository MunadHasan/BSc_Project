# Imports

import dataclasses
import datetime
import functools
import math
import re
from typing import Optional
import gc

import cartopy.crs as ccrs
from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray as xr
import pandas as pd

def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))

# Authenticate with Google Cloud Storage

gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "graphcast/"

# Choose and load model
from graphcast import checkpoint, graphcast

ckpt_file_name = "params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz"

# --- Load the checkpoint ---
with gcs_bucket.blob(f"{dir_prefix}{ckpt_file_name}").open("rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)

# Extract model params and state
params = ckpt.params
state = {}

# Extract model and task configs
model_config = ckpt.model_config
task_config = ckpt.task_config

# Optional: print info
print("Model description:\n", ckpt.description, "\n")
print("Model license:\n", ckpt.license, "\n")

model_config

file_path = 'gcs-bucket/Input files/february_23.nc'

ds = xr.open_dataset(file_path)

ds_sliced = ds.isel(time=slice(3, None))

start_date = '2023-03-01 00:00:00'
end_date = '2023-03-03 00:00:00'

new_time_index = pd.date_range(start=start_date, end=end_date, freq='6h')

new_data_vars_data = {}

for var_name, var_data in ds_sliced.data_vars.items():
    if 'time' in var_data.dims:
        new_shape = list(var_data.shape)
        time_axis = var_data.dims.index('time')
        new_shape[time_axis] = len(new_time_index)
        new_values = np.zeros(new_shape, dtype=var_data.dtype)
        new_data_vars_data[var_name] = (var_data.dims, new_values)
    else:
        new_data_vars_data[var_name] = (var_data.dims, var_data.values)


new_coords = {d: c for d, c in ds_sliced.coords.items() if d != 'time'}
new_coords['time'] = new_time_index

new_ds = xr.Dataset(new_data_vars_data, coords=new_coords)

ds = xr.concat([ds_sliced, new_ds], dim='time', combine_attrs="override")

print(f"New dataset shape - Time dimension: {ds.sizes['time']}")

num_batches = 29
timesteps_per_batch = 14
offset = 4

# Chunk dataset for memory efficiency
ds = ds.chunk({'time': 1, 'lat': 180, 'lon': 360}) 

step_ns = np.timedelta64(6, 'h')                  
last_time = np.timedelta64(3*24, 'h')               
new_time_coord_values = np.linspace(
    start=-step_ns.astype('timedelta64[ns]').astype('int64'), 
    stop=last_time.astype('timedelta64[ns]').astype('int64'),   
    num=timesteps_per_batch
).astype('timedelta64[ns]')

first_batch_slice = ds.isel(time=slice(0, timesteps_per_batch))
final_dims = ('batch',) + tuple(first_batch_slice.dims)
final_coords = {d: c for d, c in first_batch_slice.coords.items() if d != 'time'}
final_coords['batch'] = np.arange(num_batches)
final_coords['time'] = new_time_coord_values
final_coords['datetime'] = (('batch', 'time'), np.empty((num_batches, timesteps_per_batch), dtype='datetime64[ns]'))

final_data_vars = {}
for var_name, var_data in first_batch_slice.data_vars.items():
    if 'time' in var_data.dims:
        var_dims = ('batch',) + tuple(d for d in var_data.dims)
        var_shape = (num_batches,) + var_data.shape
        final_data_vars[var_name] = (var_dims, np.zeros(var_shape, dtype=var_data.dtype))
    else:
        final_data_vars[var_name] = (var_data.dims, var_data.values)

ds_batched = xr.Dataset(final_data_vars, coords=final_coords)

for i in range(num_batches):
    start_time_index = i * offset
    end_time_index = start_time_index + timesteps_per_batch

    ds_batch_slice = ds.isel(time=slice(start_time_index, end_time_index))
    ds_batch_computed = ds_batch_slice.compute()

    for var_name, var_data in ds_batch_computed.data_vars.items():
        if 'time' in var_data.dims:
            final_slice = {dim: slice(None) for dim in ds_batched[var_name].dims}
            final_slice['batch'] = i
            final_slice['time'] = slice(0, timesteps_per_batch)
            ds_batched[var_name][final_slice] = var_data.values
        else:
            ds_batched[var_name][i] = var_data.values

    ds_batched['datetime'][i, :] = ds_batch_computed.time.values.astype('datetime64[ns]')

    del ds_batch_slice
    del ds_batch_computed
    gc.collect()

ds = ds_batched

print(f"New dataset shape - Batch: {ds.sizes['batch']}, Time: {ds.sizes['time']}")

variables_to_modify = ['geopotential_at_surface', 'land_sea_mask']

for var_name in variables_to_modify:
    if var_name in ds.data_vars and 'time' in ds[var_name].dims:
        ds[var_name] = ds[var_name].isel(time=0, drop=True)
    if var_name in ds.data_vars and 'batch' in ds[var_name].dims:
        ds[var_name] = ds[var_name].isel(batch=0, drop=True)

if 'total_precipitation_6h' in ds.data_vars:
    ds = ds.rename({'total_precipitation_6h': 'total_precipitation_6hr'})


bd_list = []

for b in range(ds.sizes["batch"]):
    print(f"Processing batch {b+1}/{ds.sizes['batch']}...")
    
    ds1 = ds.sel(batch=[b])

    # Extract inputs/targets/forcings
    inputs, _, _ = data_utils.extract_inputs_targets_forcings(
        ds1,
        target_lead_times=slice("-6h", "0h"),
        **dataclasses.asdict(task_config)
    )
    _, targets, forcings = data_utils.extract_inputs_targets_forcings(
        ds1,
        target_lead_times=slice("6h", "72h"),
        **dataclasses.asdict(task_config)
    )

    with gcs_bucket.blob(dir_prefix+"stats/diffs_stddev_by_level.nc").open("rb") as f:
      diffs_stddev_by_level = xr.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/mean_by_level.nc").open("rb") as f:
      mean_by_level = xr.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/stddev_by_level.nc").open("rb") as f:
      stddev_by_level = xr.load_dataset(f).compute()

    
    # Build jitted functions for prediction
    
    def construct_wrapped_graphcast(
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig):
      """Constructs and wraps the GraphCast Predictor."""
      predictor = graphcast.GraphCast(model_config, task_config)
      predictor = casting.Bfloat16Cast(predictor)
      predictor = normalization.InputsAndResiduals(
          predictor,
          diffs_stddev_by_level=diffs_stddev_by_level,
          mean_by_level=mean_by_level,
          stddev_by_level=stddev_by_level)
      predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
      return predictor
    
    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
      predictor = construct_wrapped_graphcast(model_config, task_config)
      return predictor(inputs, targets_template=targets_template, forcings=forcings)
    
    def with_configs(fn):
      return functools.partial(fn, model_config=model_config, task_config=task_config)
    
    def with_params(fn):
      return functools.partial(fn, params=params, state=state)
    
    def drop_state(fn):
      return lambda **kw: fn(**kw)[0]
    
    init_jitted = jax.jit(with_configs(run_forward.init))
    
    if params is None: 
      params, state = init_jitted(
          rng=jax.random.PRNGKey(0),
          inputs=inputs,
          targets_template=inputs * np.nan,
          forcings=forcings)
    
    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))
    assert model_config.resolution in (0, 360. / inputs.sizes["lon"]), (
        "Model resolution doesn't match the data resolution (expected 1° for GraphCast_small)."
    )

    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets * np.nan,
        forcings=forcings,
    )

    # Slice Bangladesh from predictions; 
    bd_slice = predictions.sel(lat=slice(27, 20), lon=slice(87, 93))

    bd_list.append(bd_slice)

    del ds1, inputs, targets, forcings, predictions, bd_slice
    gc.collect()

# Merge all batches into a single dataset
bd = xr.concat(bd_list, dim="batch")

print("Final Bangladesh dataset shape:", bd.sizes)

# Save to bucket
output_path = "gcs-bucket/Output/february_23.nc"
bd.to_netcdf(output_path)
print(f"Saved predictions to {output_path}")
