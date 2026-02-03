import xarray as xr
import numpy as np
import pandas as pd
import subprocess
import os

# Paths
gcs_bucket = "gs://411_project_data"
input_gcs_file = f"{gcs_bucket}/precipitation.nc"
local_input_file = "/home/monad/precipitation.nc"
local_output_file = "/home/monad/precipitation_6h_accum.nc"
output_gcs_file = f"{gcs_bucket}/precipitation_6h_accum.nc"

# Download from GCS
print("Downloading precipitation.nc from GCS...")
subprocess.run(["gsutil", "cp", input_gcs_file, local_input_file], check=True)

# Open dataset
print("Opening dataset...")
ds = xr.open_dataset(local_input_file, engine='h5netcdf')
precip = ds['tp']
time = ds['valid_time']

# Compute 6-hour accumulated precipitation
print("Computing 6-hour accumulated precipitation...")
new_times = [time.values[0]]
chunks = []

for i in range(1, len(time) - 5, 6):
    chunk = precip.isel(valid_time=slice(i, i+6))
    acc = chunk.sum(dim='valid_time')
    
    chunk_start_time = pd.to_datetime(time.values[i])
    label_hour = ((chunk_start_time.hour // 6 + 1) * 6) % 24
    if label_hour == 0:
        new_time = (chunk_start_time + pd.Timedelta(hours=6)).replace(hour=0)
    else:
        new_time = chunk_start_time.replace(hour=label_hour)
    acc = acc.expand_dims(valid_time=[np.datetime64(new_time)])
    chunks.append(acc)

first_hour = precip.isel(valid_time=0).expand_dims(valid_time=[time.values[0]])
chunks = [first_hour] + chunks
new_tp = xr.concat(chunks, dim='valid_time')

new_tp.attrs = precip.attrs
new_tp.name = "tp"

# Save to NetCDF
print("Saving to local NetCDF...")
new_ds = xr.Dataset(
    {"tp": new_tp},
    coords={
        "valid_time": new_tp['valid_time'],
        "latitude": ds['latitude'],
        "longitude": ds['longitude']
    }
)

new_ds.to_netcdf(local_output_file, engine='h5netcdf')
print(f"Saved locally: {local_output_file}")


print("Uploading to GCS...")
subprocess.run(["gsutil", "cp", local_output_file, output_gcs_file], check=True)
print(f"Uploaded to: {output_gcs_file}")

