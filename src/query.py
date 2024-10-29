import xarray as xr
import pandas as pd
import numpy as np
from utils.get_whole_period import (
    get_whole_period_between,
    get_last_date_of_month,
    get_total_hours_in_year,
    get_total_hours_in_month,
    get_total_hours_between,
)
from utils.const import long_short_name_dict, RAW_DATA_PATH, AGG_DATA_PATH


def gen_file_list(
    variable: str,
    start_datetime: str,
    end_datetime: str,
    time_resolution: str,  # e.g., "hour", "day", "month", "year"
    time_agg_method: str,  # e.g., "mean", "max", "min"
):
    file_list = []
    if time_resolution == "hour":
        start_year = start_datetime[:4]
        end_year = end_datetime[:4]
        for year in range(int(start_year), int(end_year) + 1):
            file_path = f"{RAW_DATA_PATH}/{variable}/{variable}-{year}.nc"
            file_list.append(file_path)
    else:
        file_path = f"{AGG_DATA_PATH}/{variable}/{variable}-{time_resolution}-{time_agg_method}.nc"
        file_list.append(file_path)
    print(file_list)
    return file_list


def get_raster(
    variable: str,
    start_datetime: str,
    end_datetime: str,
    time_resolution: str,  # e.g., "hour", "day", "month", "year"
    time_agg_method: str,  # e.g., "mean", "max", "min"
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    # spatial_resolution: float,  # e.g., 0.25, 0.5, 1.0, 2.5, 5.0
):
    file_list = gen_file_list(variable, start_datetime, end_datetime, time_resolution, time_agg_method)
    ds_list = []
    for file in file_list:
        ds = xr.open_dataset(file, engine="netcdf4").sel(
            time=slice(start_datetime, end_datetime),
            latitude=slice(max_lat, min_lat),
            longitude=slice(min_lon, max_lon),
        )
        ds_list.append(ds)
    ds = xr.concat([i.chunk() for i in ds_list], dim="time")
    return ds


def get_timeseries(
    variable: str,
    start_datetime: str,
    end_datetime: str,
    time_resolution: str,  # e.g., "hour", "day", "month", "year"
    time_agg_method: str,  # e.g., "mean", "max", "min"
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    time_series_aggregation_method: str,  # e.g., "mean", "max", "min"
):
    ds = get_raster(
        variable,
        start_datetime,
        end_datetime,
        time_resolution,
        time_agg_method,
        min_lat,
        max_lat,
        min_lon,
        max_lon,
    )
    if time_series_aggregation_method == "mean":
        ts = ds.mean(dim=["latitude", "longitude"])
    elif time_series_aggregation_method == "max":
        ts = ds.max(dim=["latitude", "longitude"])
    elif time_series_aggregation_method == "min":
        ts = ds.min(dim=["latitude", "longitude"])
    return ts


def get_mean_heatmap(
    variable: str,
    start_datetime: str,
    end_datetime: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
):
    years, months, days, hours = get_whole_period_between(start_datetime, end_datetime)
    year_hours = []
    month_hours = []
    day_hours = []
    hour_hours = []
    xrds_list = []

    if years:
        year = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-year-mean.nc")
        year_match = [f"{y}-12-31 00:00:00" for y in years]
        year_selected = year.sel(time=year_match, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
        year_hours = [get_total_hours_in_year(y) for y in years]
        xrds_list.append(year_selected)

    if months:
        month = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-month-mean.nc")
        month_match = [f"{m}-{get_last_date_of_month(pd.Timestamp(m))} 00:00:00" for m in months]
        month_selected = month.sel(
            time=month_match, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        )
        month_hours = [get_total_hours_in_month(m) for m in months]
        xrds_list.append(month_selected)

    if days:
        day = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-day-mean.nc")
        day_match = [f"{d} 00:00:00" for d in days]
        day_selected = day.sel(time=day_match, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
        day_hours = [24 for _ in days]
        xrds_list.append(day_selected)

    if hours:
        year_hour_dict = {}
        for h in hours:
            year = h.split("-")[0]
            if year not in year_hour_dict:
                year_hour_dict[year] = []
            year_hour_dict[year].append(h)

        ds_list = []
        for y in year_hour_dict:
            file_path = f"{RAW_DATA_PATH}/{variable}/{variable}-{y}.nc"
            ds = xr.open_dataset(file_path, engine="netcdf4").sel(
                time=year_hour_dict[y], latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
            )
            ds_list.append(ds)
        hour_selected = xr.concat(ds_list, dim="time")
        hour_hours = [1 for _ in hours]
        xrds_list.append(hour_selected)

    xrds_concat = xr.concat(xrds_list, dim="time")
    nd_array = xrds_concat["t2m"].to_numpy()
    weights = np.array(year_hours + month_hours + day_hours + hour_hours)
    total_hours = get_total_hours_between(start_datetime, end_datetime)
    weights = weights / total_hours
    average = np.average(nd_array, axis=0, weights=weights)
    res = xr.Dataset(
        {long_short_name_dict[variable]: (["latitude", "longitude"], average)},
        coords={"latitude": xrds_concat.latitude, "longitude": xrds_concat.longitude},
    )
    return res


def get_min_heatmap(
    variable: str,
    start_datetime: str,
    end_datetime: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
):
    years, months, days, hours = get_whole_period_between(start_datetime, end_datetime)
    xrds_list = []

    if years:
        year = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-year-min.nc")
        year_match = [f"{y}-12-31 00:00:00" for y in years]
        year_selected = year.sel(time=year_match, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
        xrds_list.append(year_selected)

    if months:
        month = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-month-min.nc")
        month_match = [f"{m}-{get_last_date_of_month(pd.Timestamp(m))} 00:00:00" for m in months]
        month_selected = month.sel(
            time=month_match, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        )
        xrds_list.append(month_selected)

    if days:
        day = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-day-min.nc")
        day_match = [f"{d} 00:00:00" for d in days]
        day_selected = day.sel(time=day_match, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
        xrds_list.append(day_selected)

    if hours:
        year_hour_dict = {}
        for h in hours:
            year = h.split("-")[0]
            if year not in year_hour_dict:
                year_hour_dict[year] = []
            year_hour_dict[year].append(h)

        ds_list = []
        for y in year_hour_dict:
            file_path = f"{RAW_DATA_PATH}/{variable}/{variable}-{y}.nc"
            ds = xr.open_dataset(file_path, engine="netcdf4").sel(
                time=year_hour_dict[y], latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
            )
            ds_list.append(ds)
        hour_selected = xr.concat(ds_list, dim="time")
        xrds_list.append(hour_selected)

    xrds_concat = xr.concat(xrds_list, dim="time")
    res = xrds_concat.min(dim="time")

    return res


def get_max_heatmap(
    variable: str,
    start_datetime: str,
    end_datetime: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
):
    years, months, days, hours = get_whole_period_between(start_datetime, end_datetime)
    xrds_list = []

    if years:
        year = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-year-max.nc")
        year_match = [f"{y}-12-31 00:00:00" for y in years]
        year_selected = year.sel(time=year_match, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
        xrds_list.append(year_selected)

    if months:
        month = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-month-max.nc")
        month_match = [f"{m}-{get_last_date_of_month(pd.Timestamp(m))} 00:00:00" for m in months]
        month_selected = month.sel(
            time=month_match, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        )
        xrds_list.append(month_selected)

    if days:
        day = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-day-max.nc")
        day_match = [f"{d} 00:00:00" for d in days]
        day_selected = day.sel(time=day_match, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
        xrds_list.append(day_selected)

    if hours:
        year_hour_dict = {}
        for h in hours:
            year = h.split("-")[0]
            if year not in year_hour_dict:
                year_hour_dict[year] = []
            year_hour_dict[year].append(h)

        ds_list = []
        for y in year_hour_dict:
            file_path = f"{RAW_DATA_PATH}/{variable}/{variable}-{y}.nc"
            ds = xr.open_dataset(file_path, engine="netcdf4").sel(
                time=year_hour_dict[y], latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
            )
            ds_list.append(ds)
        hour_selected = xr.concat(ds_list, dim="time")
        xrds_list.append(hour_selected)

    xrds_concat = xr.concat(xrds_list, dim="time")
    res = xrds_concat.max(dim="time")
    return res


def get_heatmap(
    variable: str,
    start_datetime: str,
    end_datetime: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    heatmap_aggregation_method: str,  # e.g., "mean", "max", "min"
):
    if heatmap_aggregation_method == "mean":
        return get_mean_heatmap(
            variable,
            start_datetime,
            end_datetime,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        )
    elif heatmap_aggregation_method == "max":
        return get_max_heatmap(
            variable,
            start_datetime,
            end_datetime,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        )
    elif heatmap_aggregation_method == "min":
        return get_min_heatmap(
            variable,
            start_datetime,
            end_datetime,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        )
    else:
        raise ValueError("Invalid heatmap_aggregation_method")


def find_time_baseline(
    variable: str,
    start_datetime: str,
    end_datetime: str,
    time_resolution: str,  # e.g., "hour", "day", "month", "year"
    time_agg_method: str,  # e.g., "mean", "max", "min"
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    time_series_aggregation_method: str,  # e.g., "mean", "max", "min"
    filter_predicate: str,  # e.g., ">", "<", "==", "!=", ">=", "<="
    filter_value: float,
):
    ts = get_timeseries(
        variable,
        start_datetime,
        end_datetime,
        time_resolution,
        time_agg_method,
        min_lat,
        max_lat,
        min_lon,
        max_lon,
        time_series_aggregation_method,
    )
    if filter_predicate == ">":
        res = ts.where(ts > filter_value, drop=False)
    elif filter_predicate == "<":
        res = ts.where(ts < filter_value, drop=False)
    elif filter_predicate == "==":
        res = ts.where(ts == filter_value, drop=False)
    elif filter_predicate == "!=":
        res = ts.where(ts != filter_value, drop=False)
    elif filter_predicate == ">=":
        res = ts.where(ts >= filter_value, drop=False)
    elif filter_predicate == "<=":
        res = ts.where(ts <= filter_value, drop=False)
    res = res.fillna(False)
    res = res.astype(bool)
    return res


def find_time_pyramid(
    variable: str,
    start_datetime: str,
    end_datetime: str,
    time_resolution: str,  # e.g., "hour", "day", "month", "year"
    time_agg_method: str,  # e.g., "mean", "max", "min"
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    time_series_aggregation_method: str,  # e.g., "mean", "max", "min"
    filter_predicate: str,  # e.g., ">", "<", "==", ">=", "<="
    filter_value: float,
):
    """
    Optimizations hueristics:
        - find hour >  x: if year-min >  x, return True ; if year-max <= x, return False
        - find hour <  x: if year-min >= x, return False; if year-max <  x, return True
        - find hour == x: if year-min >  x, return False; if year-max <  x, return False
        - find hour >= x: if year-min >= x, return True ; if year-max <  x, return False
        - find hour <= x: if year-min >  x, return False; if year-max <= x, return True
    """
    if time_resolution == "year" or time_resolution == "month" or time_resolution == "day" or filter_predicate == "!=":
        return find_time_baseline(
            variable,
            start_datetime,
            end_datetime,
            time_resolution,  # e.g., "hour", "day", "month", "year"
            time_agg_method,  # e.g., "mean", "max", "min"
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            time_series_aggregation_method,  # e.g., "mean", "max", "min"
            filter_predicate,  # e.g., ">", "<", "==", "!=", ">=", "<="
            filter_value,
        )
    pass


def find_time_pyramid_hour(
    variable: str,
    start_datetime: str,
    end_datetime: str,
    time_resolution: str,  # e.g., "hour", "day", "month", "year"
    time_agg_method: str,  # e.g., "mean", "max", "min"
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    time_series_aggregation_method: str,  # e.g., "mean", "max", "min"
    filter_predicate: str,  # e.g., ">", "<", "==", ">=", "<="
    filter_value: float,
):
    short_variable = long_short_name_dict[variable]
    """when time resolution is hour"""
    years, months, days, hours = get_whole_period_between(start_datetime, end_datetime)
    time_points = pd.date_range(start=start_datetime, end=end_datetime, freq="h")
    result = xr.Dataset(
        data_vars={short_variable: (["time"], [None] * len(time_points))},
        coords=dict(time=time_points),
    )

    if years:
        year_min = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-year-min.nc")
        year_max = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-year-max.nc")
        for year in years:
            year_determined = False
            year_datetime = f"{year}-12-31 00:00:00"
            curr_year_min = year_min.sel(
                time=year_datetime, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
            )[short_variable].min()
            curr_year_max = year_max.sel(
                time=year_datetime, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
            )[short_variable].max()
            if filter_predicate == ">":
                if curr_year_min > filter_value:
                    print(f"{year}: min > filter, True")
                    year_determined = True
                    result[short_variable].loc[str(year) : str(year)] = True
                elif curr_year_max <= filter_value:
                    print(f"{year}: max <= filter, False")
                    year_determined = True
                    result[short_variable].loc[str(year) : str(year)] = False
            elif filter_predicate == "<":
                if curr_year_min >= filter_value:
                    print(f"{year}: min >= filter, False")
                    year_determined = True
                    result[short_variable].loc[str(year) : str(year)] = False
                elif curr_year_max < filter_value:
                    print(f"{year}: max < filter, True")
                    year_determined = True
                    result[short_variable].loc[str(year) : str(year)] = True
            elif filter_predicate == "==":
                if curr_year_min > filter_value or curr_year_max < filter_value:
                    print(f"{year}: min > filter or max < filter, False")
                    year_determined = True
                    result[short_variable].loc[str(year) : str(year)] = False
            if not year_determined:
                # add monthes to months
                months = months + [f"{year}-{month:02d}" for month in range(1, 13)]

    if months:
        print(months)
        month_min = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-month-min.nc")
        month_max = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-month-max.nc")
        for month in months:
            month_determined = False
            month_datetime = f"{month}-{get_last_date_of_month(pd.Timestamp(month))} 00:00:00"
            curr_month_min = month_min.sel(
                time=month_datetime, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
            )[short_variable].min()
            curr_month_max = month_max.sel(
                time=month_datetime, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
            )[short_variable].max()
            if filter_predicate == ">":
                if curr_month_min > filter_value:
                    print(f"{month}: min > filter, True")
                    month_determined = True
                    result[short_variable].loc[month:month] = True
                elif curr_month_max <= filter_value:
                    print(f"{month}: max <= filter, False")
                    month_determined = True
                    result[short_variable].loc[month:month] = False
            elif filter_predicate == "<":
                if curr_month_min >= filter_value:
                    print(f"{month}: min >= filter, False")
                    month_determined = True
                    result[short_variable].loc[month:month] = False
                elif curr_month_max < filter_value:
                    print(f"{month}: max < filter, True")
                    month_determined = True
                    result[short_variable].loc[month:month] = True
            elif filter_predicate == "==":
                if curr_month_min > filter_value or curr_month_max < filter_value:
                    print(f"{month}: min > filter or max < filter, False")
                    month_determined = True
                    result[short_variable].loc[month:month] = False
            if not month_determined:
                # add days to days
                days = days + [
                    f"{month}-{day:02d}" for day in range(1, get_last_date_of_month(pd.Timestamp(month)) + 1)
                ]

    if days:
        print(days)
        day_min = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-day-min.nc")
        day_max = xr.open_dataset(f"{AGG_DATA_PATH}/{variable}/{variable}-day-max.nc")
        for day in days:
            day_datetime = f"{day} 00:00:00"
            curr_day_min = day_min.sel(
                time=day_datetime, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
            )[short_variable].min()
            curr_day_max = day_max.sel(
                time=day_datetime, latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
            )[short_variable].max()
            if filter_predicate == ">":
                if curr_day_min > filter_value:
                    print(f"{day}: min > filter, True")
                    result[short_variable].loc[day:day] = True
                elif curr_day_max <= filter_value:
                    print(f"{day}: max <= filter, False")
                    result[short_variable].loc[day:day] = False
            elif filter_predicate == "<":
                if curr_day_min >= filter_value:
                    print(f"{day}: min >= filter, False")
                    result[short_variable].loc[day:day] = False
                elif curr_day_max < filter_value:
                    print(f"{day}: max < filter, True")
                    result[short_variable].loc[day:day] = True
            elif filter_predicate == "==":
                if curr_day_min > filter_value or curr_day_max < filter_value:
                    print(f"{day}: min > filter or max < filter, False")
                    result[short_variable].loc[day:day] = False

    result_undetermined = result["time"].where(result[short_variable].isnull(), drop=True)
    if result_undetermined.size > 0:
        min_hour = result_undetermined.min().values
        max_hour = result_undetermined.max().values
        min_hour = pd.Timestamp(min_hour).strftime("%Y-%m-%d %H:%M:%S")
        max_hour = pd.Timestamp(max_hour).strftime("%Y-%m-%d %H:%M:%S")
        print("Check hour: ", min_hour, max_hour)

        rest = find_time_baseline(
            variable,
            min_hour,
            max_hour,
            time_resolution,  # e.g., "hour", "day", "month", "year"
            time_agg_method,  # e.g., "mean", "max", "min"
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            time_series_aggregation_method,  # e.g., "mean", "max", "min"
            filter_predicate,  # e.g., ">", "<", "==", "!=", ">=", "<="
            filter_value,
        )

        result[short_variable].loc[f"{min_hour}":f"{max_hour}"] = rest[short_variable]
        result[short_variable] = result[short_variable].astype(bool)
    return result


def find_area_baseline(
    variable: str,
    start_datetime: str,
    end_datetime: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    heatmap_aggregation_method: str,  # e.g., "mean", "max", "min"
    filter_predicate: str,  # e.g., ">", "<", "==", "!=", ">=", "<="
    filter_value: float,
):
    hm = get_heatmap(
        variable,
        start_datetime,
        end_datetime,
        min_lat,
        max_lat,
        min_lon,
        max_lon,
        heatmap_aggregation_method,
    )
    if filter_predicate == ">":
        res = hm.where(hm > filter_value, drop=False)
    elif filter_predicate == "<":
        res = hm.where(hm < filter_value, drop=False)
    elif filter_predicate == "==":
        res = hm.where(hm == filter_value, drop=False)
    elif filter_predicate == "!=":
        res = hm.where(hm != filter_value, drop=False)
    elif filter_predicate == ">=":
        res = hm.where(hm >= filter_value, drop=False)
    elif filter_predicate == "<=":
        res = hm.where(hm <= filter_value, drop=False)
    res = res.fillna(False)
    res = res.astype(bool)
    return res
