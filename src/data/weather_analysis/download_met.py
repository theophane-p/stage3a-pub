import sys
sys.path.append("src")
from visu import map

import os
import requests
import time

import pandas as pd
import xarray as xr
import cartopy.crs as ccrs


def download_file(
        url: str, 
        filepath: str,
        timeout: int=60,
        allow_redirects:bool=False
    ) -> bool:

    r"""Download a file from a given URL and save it to a given filepath.
    
    Parameters
    ----------
    url : str
        The URL of the file to download.
    filepath : str
        The path to save the downloaded file.
    timeout : int, optional
        The number of seconds to wait for a response from the server. Default is 60.
    allow_redirects : bool, optional
        Whether or not to follow redirects. Default is False.
        
    Returns
    -------
    bool
        True if the file was downloaded successfully, False otherwise.
    """

    try:
        response = requests.get(url, timeout=timeout, allow_redirects=allow_redirects)
        response.raise_for_status()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as file:
            file.write(response.content)
    except requests.exceptions.RequestException as e:
        print(e)
        print(f"Failed to download file {filepath} from {url}")
        return False
    return True


def timestamp_to_filename(
        timestamp: pd.Timestamp
    ) -> str:

    r"""Convert a timestamp to a MET Norway analysis filename.

    Parameters
    ----------
    timestamp : pd.Timestamp
        The timestamp to convert.
    """

    return timestamp.strftime("met_analysis_1_0km_nordic_%Y%m%dT%HZ.nc")


def filename_to_timestamp(
        filename: str
    ) -> pd.Timestamp:

    r"""Convert a MET Norway analysis filename to a timestamp.

    Parameters
    ----------
    filename : str
        The filename to convert.
    """

    return pd.Timestamp(filename[26:38])


def download_met_file(
        timestamp: pd.Timestamp,
        output_dir: str="."
    ) -> bool:

    r"""Download a MET Norway analysis file for a given timestamp (meteorology, netcdf4 format).

    Parameters
    ----------
    timestamp : pd.Timestamp
        The timestamp for the MET Norway analysis file.
    output_dir : str, optional
        The directory to save the downloaded file. Default is ".".
    
    Returns
    -------
    bool
        True if the file was downloaded successfully, False otherwise.
    """

    base_url = "https://thredds.met.no/thredds/fileServer/metpparchive/"
    day_folder = timestamp.strftime("%Y/%m/%d/")
    filename = timestamp_to_filename(timestamp)
    url = base_url + day_folder + filename

    filepath = os.path.join(output_dir, filename)
    return download_file(url, filepath)


def default_processing(
        ds: xr.Dataset
    ) -> xr.Dataset:

    r"""Default processing function for MET Norway analysis files.

    Parameters
    ----------
    ds : xr.Dataset
        The MET Norway analysis file as an xarray Dataset.

    Returns
    -------
    xr.Dataset
        The processed MET Norway analysis file.
    """

    longitude_min, longitude_max, latitude_min, latitude_max = map.default_extents["lofoten"]
    wind_crs = ccrs.LambertConformal(central_longitude=15, central_latitude=63, standard_parallels=(63, 63))
    x_min, y_min = wind_crs.transform_point(longitude_min, latitude_min, ccrs.PlateCarree())
    x_max, y_max = wind_crs.transform_point(longitude_max, latitude_max, ccrs.PlateCarree())

    small_ds = ds.loc[dict(x=slice(x_min, x_max), y=slice(y_min, y_max))].copy()
    small_ds = small_ds.rio.set_crs(wind_crs)
    return small_ds


def process_met_file(
        path: str, 
        processing_function=default_processing,
        output_dir: str=".",
        delete_original: bool=False
    ) -> None:

    r"""Process a MET Norway analysis file using a given processing function.

    Parameters
    ----------
    path : str
        The path to the MET Norway analysis file.
    processing_function : function, optional
        The function to process the MET Norway analysis file. Default is default_processing.
        Signature: function(ds: xr.Dataset) -> xr.Dataset
    output_dir : str, optional
        The directory to save the processed file. Default is ".".
    delete_original : bool, optional
        Whether or not to delete the original file. Default is False.
    """

    with xr.open_dataset(path) as ds:
        processed_ds = processing_function(ds)
    head, tail = os.path.split(path)
    processed_path = os.path.join(output_dir, tail)
    processed_ds.to_netcdf(processed_path)

    if delete_original:
        os.remove(path)


def merge_met_files(
        paths: list[str], 
        output_filepath: str="merged_met.nc",
        delete_original: bool=False
    ) -> None:

    r"""Merge multiple MET Norway analysis files into a single file.

    Parameters
    ----------
    paths : list[str]
        The paths to the MET Norway analysis files to merge.
    output_filepath : str, optional
        The path to save the merged file. Default is "merged_met.nc".
    delete_original : bool, optional
        Whether or not to delete the original files. Default is False.
    """

    datasets = [xr.open_dataset(path) for path in paths]
    merged_ds = xr.concat(datasets, dim="time", data_vars=["forecast_reference_time"])
    merged_ds.to_netcdf(output_filepath)

    if delete_original:
        for path in paths:
            os.remove(path)


def download_process(
        timestamp: pd.Timestamp,
        original_dir: str,
        processed_dir: str,
        processing_function=default_processing,
        delete_original: bool=True,
        n_tries: int=5,
        sleep_time: int=5
    ) -> bool:

    r"""Download, process, and save a MET Norway analysis file for a given timestamp.
    
    Parameters
    ----------
    timestamp : pd.Timestamp
        The timestamp for the MET Norway analysis file.
    original_dir : str
        The directory to save the original file.
    processed_dir : str
        The directory to save the processed file.
    processing_function : function, optional
        The function to process the MET Norway analysis file. Default is default_processing.
        Signature: function(ds: xr.Dataset) -> xr.Dataset
    delete_original : bool, optional
        Whether or not to delete the original file. Default is True.
    n_tries : int, optional
        The number of download attempts. Default is 5.
    sleep_time : int, optional
        The number of seconds to sleep between download attempts. Default is 5.
    """

    downloaded = False
    for _ in range(n_tries):
        if download_met_file(timestamp, original_dir):
            downloaded = True
            break
        time.sleep(sleep_time)
    
    if not downloaded:
        return False
    
    filename = timestamp_to_filename(timestamp)
    original_path = os.path.join(original_dir, filename)
    process_met_file(
        original_path, 
        processing_function=processing_function, 
        output_dir=processed_dir, 
        delete_original=delete_original
    )
    return True


def download_process_merge(
        timestamps: list[pd.Timestamp],
        output_name: str="met_ds.nc",
        processing_function=default_processing,
        delete_original: bool=True,
        delete_intermediate: bool=True,
        tag: str | None = None, 
        verbose: bool=False
    ) -> None:

    r"""Download, process, and merge MET Norway analysis files for a list of timestamps.

    Parameters
    ----------
    timestamps : list[pd.Timestamp]
        The timestamps for the MET Norway analysis files.
    output_name : str, optional
        The name of the output file. Default is "met_ds.nc".
    processing_function : function, optional
        The function to process the MET Norway analysis files. Default is default_processing.
        Signature: function(ds: xr.Dataset) -> xr.Dataset
    delete_original : bool, optional
        Whether or not to delete the original files. Default is True.
    delete_intermediate : bool, optional
        Whether or not to delete intermediate files. Default is True.
    tag : str, optional
        A tag to append to the output directory. Default will be call time.
    """

    out_dir, _ = os.path.split(output_name)
    if tag is None:
        tag = time.strftime("%Y%m%d_%H%M%S")
    original_dir = os.path.join(out_dir, "original_" + tag)
    processed_dir = os.path.join(out_dir, "processed_" + tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    if verbose:
        print("=============================================")
        print(f"Directories ok : \n\t{original_dir}, \n\t{processed_dir}, \n\t{out_dir}")
        print("=============================================")

    # Download and process files
    failures = []
    downloads = []
    for timestamp in timestamps:
        if not download_process(timestamp, original_dir, processed_dir, processing_function, delete_original):
            failures.append(timestamp)
        elif verbose:
            print(f"{timestamp} : Downloaded and processed")
    if failures:
        raise Exception(f"Failed to download files for timestamps: {failures}")
    
    if verbose:
        print("=============================================")
        print("Downloaded and processed all files, merging...")

    # Merge files
    processed_paths = [os.path.join(processed_dir, timestamp_to_filename(timestamp)) for timestamp in timestamps]
    merge_met_files(processed_paths, output_name, delete_original=delete_intermediate)

    # Clean up
    if delete_original:
        os.rmdir(original_dir)
    if delete_intermediate:
        os.rmdir(processed_dir)


    




