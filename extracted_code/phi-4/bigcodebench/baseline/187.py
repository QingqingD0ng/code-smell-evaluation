import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def task_func(dic={'Lon': (-180, 180), 'Lat': (-90, 90)}, cities=['New York', 'London', 'Beijing', 'Tokyo', 'Sydney']):
    if 'Lon' not in dic or 'Lat' not in dic:
        raise ValueError("Dictionary must contain 'Lon' and 'Lat' keys.")
    
    lon_range = dic['Lon']
    lat_range = dic['Lat']
    
    if not isinstance(lon_range, tuple) or not isinstance(lat_range, tuple):
        raise ValueError("'Lon' and 'Lat' values must be tuples.")
    
    lons = np.random.uniform(lon_range[0], lon_range[1], len(cities))
    lats = np.random.uniform(lat_range[0], lat_range[1], len(cities))
    
    points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    
    data = {'City': cities, 'Coordinates': points}
    gdf = gpd.GeoDataFrame(data, geometry='Coordinates')
    
    return gdf