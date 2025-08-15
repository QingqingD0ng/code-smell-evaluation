import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def task_func(dic={'Lon': (-180, 180), 'Lat': (-90, 90)}, cities=['New York', 'London', 'Beijing', 'Tokyo', 'Sydney']):
    # Validate dictionary keys
    if 'Lon' not in dic or 'Lat' not in dic:
        raise ValueError("Dictionary must contain 'Lon' and 'Lat' keys.")
    if not (isinstance(dic['Lon'], tuple) and isinstance(dic['Lat'], tuple)):
        raise ValueError("'Lon' and 'Lat' values must be tuples.")

    # Generate random coordinates
    lons = np.random.uniform(dic['Lon'][0], dic['Lon'][1], len(cities))
    lats = np.random.uniform(dic['Lat'][0], dic['Lat'][1], len(cities))
    
    # Create GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    gdf = gpd.GeoDataFrame({'City': cities, 'Coordinates': geometry})
    
    return gdf