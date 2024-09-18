import pandas as pd
import calendar


def get_last_date_of_month(dt):
    return calendar.monthrange(dt.year, dt.month)[1]


def get_whole_year_between(start_dt, end_dt):
    assert start_dt.year <= end_dt.year
    start_year = start_dt.year
    end_year = end_dt.year
    first_hour_start_year = pd.Timestamp(f"{start_year}-01-01 00:00:00")
    last_hour_end_year = pd.Timestamp(f"{end_year}-12-31 23:00:00")
    years = list(range(start_year, end_year + 1))
    if start_dt != first_hour_start_year:
        years = years[1:]
    if end_dt != last_hour_end_year:
        years = years[:-1]
    # residual
    residual = []
    last_hour_start_year = pd.Timestamp(f"{start_year}-12-31 23:00:00")
    first_hour_end_year = pd.Timestamp(f"{end_year}-01-01 00:00:00")
    if end_dt <= last_hour_start_year:
        residual.append([start_dt, end_dt])
    else:
        if start_dt != first_hour_start_year:
            residual.append([start_dt, last_hour_start_year])
        if end_dt < last_hour_end_year:
            residual.append([first_hour_end_year, end_dt])
    # print("years:", years)
    # print("month residual:", residual)
    return years, residual


def get_whole_month_between(start_dt, end_dt):
    assert start_dt.year == end_dt.year
    year = start_dt.year
    start_month = start_dt.month
    end_month = end_dt.month
    first_hour_start_month = pd.Timestamp(f"{year}-{start_month:02d}-01 00:00:00")
    last_hour_end_month = pd.Timestamp(f"{year}-{end_month:02d}-{get_last_date_of_month(end_dt)} 23:00:00")
    months = [f"{year}-{month:02d}" for month in range(start_month, end_month + 1)]
    if start_dt != first_hour_start_month:
        months = months[1:]
    if end_dt != last_hour_end_month:
        months = months[:-1]
    # residual
    residual = []
    last_hour_start_month = pd.Timestamp(f"{year}-{start_month:02d}-{get_last_date_of_month(start_dt)} 23:00:00")
    first_hour_end_month = pd.Timestamp(f"{year}-{end_month:02d}-01 00:00:00")
    if end_dt <= last_hour_start_month:
        residual.append([start_dt, end_dt])
    else:
        if start_dt != first_hour_start_month:
            residual.append([start_dt, last_hour_start_month])
        if end_dt < last_hour_end_month:
            residual.append([first_hour_end_month, end_dt])
    # print("months:", months)
    # print("day residual:", residual)
    return months, residual


def get_whole_day_between(start_dt, end_dt):
    assert start_dt.year == end_dt.year
    assert start_dt.month == end_dt.month
    year = start_dt.year
    month = start_dt.month
    start_day = start_dt.day
    end_day = end_dt.day
    first_hour_start_day = pd.Timestamp(f"{year}-{month:02d}-{start_day:02d} 00:00:00")
    last_hour_end_day = pd.Timestamp(f"{year}-{month:02d}-{end_day:02d} 23:00:00")
    days = [f"{year}-{month:02d}-{day:02d}" for day in range(start_day, end_day + 1)]
    if start_dt != first_hour_start_day:
        days = days[1:]
    if end_dt != last_hour_end_day:
        days = days[:-1]
    # residual
    residual = []
    last_hour_start_day = pd.Timestamp(f"{year}-{month:02d}-{start_day:02d} 23:00:00")
    first_hour_end_day = pd.Timestamp(f"{year}-{month:02d}-{end_day:02d} 00:00:00")
    if end_dt <= last_hour_start_day:
        residual.append([start_dt, end_dt])
    else:
        if start_dt != first_hour_start_day:
            residual.append([start_dt, last_hour_start_day])
        if end_dt < last_hour_end_day:
            residual.append([first_hour_end_day, end_dt])
    # print("days:", days)
    # print("hour residual:", residual)
    return days, residual


def get_whole_hour_between(start_dt, end_dt):
    assert start_dt.year == end_dt.year
    assert start_dt.month == end_dt.month
    assert start_dt.day == end_dt.day
    year = start_dt.year
    month = start_dt.month
    day = start_dt.day
    start_hour = start_dt.hour
    end_hour = end_dt.hour
    hours = [f"{year}-{month:02d}-{day:02d} {hour:02d}:00:00" for hour in range(start_hour, end_hour + 1)]
    # print("hours:", hours)
    return hours


def get_whole_period_between(start, end):
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    whole_months = []
    whole_days = []
    whole_hours = []
    whole_years, residual = get_whole_year_between(start_dt, end_dt)
    for res in residual:
        months, residual = get_whole_month_between(pd.Timestamp(res[0]), pd.Timestamp(res[1]))
        whole_months.extend(months)
        for res in residual:
            days, residual = get_whole_day_between(pd.Timestamp(res[0]), pd.Timestamp(res[1]))
            whole_days.extend(days)
            for res in residual:
                hours = get_whole_hour_between(pd.Timestamp(res[0]), pd.Timestamp(res[1]))
                whole_hours.extend(hours)
    print("******************")
    print("whole year")
    for y in whole_years:
        print(y)
    print("whole month")
    for m in whole_months:
        print(m)
    print("whole day")
    for d in whole_days:
        print(d)
    print("whole hour")
    for h in whole_hours:
        print(h)
    return whole_years, whole_months, whole_days, whole_hours


def get_total_hours_in_year(year):
    return 24 * 365 if calendar.isleap(year) else 24 * 366


def get_total_hours_in_month(month):
    return 24 * get_last_date_of_month(pd.Timestamp(month))


def get_total_hours_between(start, end):
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    return int((end_dt - start_dt) / pd.Timedelta("1 hour")) + 1
