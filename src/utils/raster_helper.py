import xarray as xr
import numpy as np
import pandas as pd

lat_range = np.arange(-90, 90.1, 0.25)
lon_range = np.arange(-180, 180.1, 0.25)


def time_resolution_to_freq(time_resolution):
    if time_resolution == "hour":
        return "h"
    elif time_resolution == "day":
        return "D"
    elif time_resolution == "month":
        return "ME"
    elif time_resolution == "year":
        return "YE"
    else:
        raise ValueError("Invalid time_resolution")


def gen_xarray_from_query(query):
    lat_start = lat_range.searchsorted(query["min_lat"], side="left")
    lat_end = lat_range.searchsorted(query["max_lat"], side="right")
    lon_start = lon_range.searchsorted(query["min_lon"], side="left")
    lon_end = lon_range.searchsorted(query["max_lon"], side="right")
    ds_query = xr.Dataset()
    ds_query["latitude"] = lat_range[lat_start:lat_end]
    ds_query["longitude"] = lon_range[lon_start:lon_end]
    ds_query["time"] = pd.date_range(
        start=query["start_datetime"], end=query["end_datetime"], freq=time_resolution_to_freq(query["time_resolution"])
    )
    return ds_query


def gen_xarray_from_meta(row):
    lat_start = lat_range.searchsorted(row["min_lat"], side="left")
    lat_end = lat_range.searchsorted(row["max_lat"], side="right")
    lon_start = lon_range.searchsorted(row["min_lon"], side="left")
    lon_end = lon_range.searchsorted(row["max_lon"], side="right")
    ds = xr.Dataset()
    ds["latitude"] = lat_range[lat_start:lat_end]
    ds["longitude"] = lon_range[lon_start:lon_end]
    ds["time"] = pd.date_range(
        start=row["start_datetime"],
        end=row["end_datetime"],
        freq=time_resolution_to_freq(row["resolution"]),
    )
    return ds


def get_relevant_meta(df, query):
    df_relevant = df[
        (df["resolution"] == query["time_resolution"])
        & (df["min_lat"] <= query["max_lat"])
        & (df["max_lat"] >= query["min_lat"])
        & (df["min_lon"] <= query["max_lon"])
        & (df["max_lon"] >= query["min_lon"])
        & (pd.to_datetime(df["start_datetime"]) <= pd.to_datetime(query["end_datetime"]))
        & (pd.to_datetime(df["end_datetime"]) >= pd.to_datetime(query["start_datetime"]))
    ]
    return df_relevant


# def is_overlap(q_min, q_max, m_min, m_max):
#     """return if q overlaps with m"""
#     return q_min <= m_max and m_min <= q_max


def mask_query_with_meta(ds_query, ds_meta):
    return (
        ds_query["latitude"].isin(ds_meta["latitude"])
        & ds_query["longitude"].isin(ds_meta["longitude"])
        & ds_query["time"].isin(ds_meta["time"])
    )