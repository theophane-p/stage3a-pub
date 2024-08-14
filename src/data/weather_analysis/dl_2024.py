import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import download_met


lonlat_crs = ccrs.PlateCarree()
wind_crs = ccrs.LambertConformal(central_longitude=15, central_latitude=63, standard_parallels=(63, 63))

my_locations = [
    (67.5267, 12.1038),
    (68.26283, 14.24983),
    (67.34, 13.57),
    (68.327, 14.1777),
    (67.4048, 13.8958), 
    (68.6072, 14.4347)
]
my_x_y = wind_crs.transform_points(lonlat_crs, np.array(my_locations)[:,1], np.array(my_locations)[:,0])[:, :2]
margin = 2.e3
x_min = my_x_y[:,0].min() - margin
x_max = my_x_y[:,0].max() + margin
y_min = my_x_y[:,1].min() - margin
y_max = my_x_y[:,1].max() + margin


def processing(
        ds: xr.Dataset
    ) -> xr.Dataset:

    r"""Processing function for MET Norway analysis files."""

    small_ds = ds.copy()
    small_ds = small_ds.loc[dict(x=slice(x_min, x_max), y=slice(y_min, y_max))]
    small_ds = small_ds.rio.set_crs(wind_crs)
    return small_ds


def main():

    out_name = "jan10_med.nc"
    start_time = pd.Timestamp('2024-01-01 00:00:00')
    end_time = pd.Timestamp('2024-02-01 00:00:00')
    hours = pd.date_range(start=start_time, end=end_time, freq='h')

    download_met.download_process_merge(
        timestamps=hours,
        output_name=out_name,
        processing_function=processing,
        delete_original=True,
        delete_intermediate=False,
        verbose=True
    )


if __name__ == '__main__':
    main()


