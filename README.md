# iharp-queries

To init venv
```
bash init_venv.sh 
```

---

| Query              |       Status       |
| ------------------ | :----------------: |
| get raster         | :white_check_mark: |
| get time series    | :white_check_mark: |
| get heatmap (mean) | :white_check_mark: |
| get heatmap (min)  | :white_check_mark: |
| get heatmap (max)  | :white_check_mark: |
| get heatmap (sum)  |        todo        |
| find time          | under optimization |
| find area          |  go with baseline  |

---

| Query           | Input                                                                                                                                                                                     |             Example              |
| --------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------: |
| get raster      | variable</br>start_datetime</br>end_datetime</br>time_resolution(hour, day, month, year)</br>time_aggregation_method(mean, min, max, sum)</br>min_lat</br>max_lat</br>min_lon</br>max_lon |   ![](figures/get_raster.png)    |
| get time series | everything in **[get raster]**</br>time_series_aggregation_method(mean, min, max, sum)                                                                                                    | ![](figures/get_time_series.png) |
| get heatmap     | variable</br>start_datetime</br>end_datetime</br>min_lat</br>max_lat</br>min_lon</br>max_lon</br>heatmap_aggregation_method(mean, min, max, sum)                                          |   ![](figures/get_heatmap.png)   |
| find time       | everything in **[get time series]**</br>filter_predicate(>, <, ==, !=, >=, <=)</br>filter_value                                                                                           |    ![](figures/find_time.png)    |
| find area       | everything in **[get heatmap]**</br>filter_predicate(>, <, ==, !=, >=, <=)</br>filter_value                                                                                               |    ![](figures/find_area.png)    |

---

### Query variants

*"Which day is hottest for Greenland in last 20 years?"*