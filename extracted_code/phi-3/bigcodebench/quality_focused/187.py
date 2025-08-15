import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def task_func(dic={'Lon': (-180, 180), 'Lat': (-90, 90)}, cities=['New York', 'London', 'Beijing', 'Tokyo', 'Sydney']):
    # Check if the required keys exist and have the correct tuple format
    if 'Lon' not in dic or 'Lat' not in dic or not isinstance(dic['Lon'], tuple) or not isinstance(dic['Lat'], tuple):
        raise ValueError("Dictionary must contain 'Lon' and 'Lat' keys with tuple values for coordinate ranges.")
    
    # Generate random coordinates within the specified ranges
    lons = np.random.uniform(dic['Lon'][0], dic['Lon'][1], len(cities))
    lats = np.random.uniform(dic['Lat'][0], dic['Lat'][1], len(cities))
    
    # Create a GeoDataFrame with City and Coordinates (Point objects)
    data = {'City': cities, 'Coordinates': [Point(lon, lat) for lon, lat in zip(lons, lats)]}
    gdf = gpd.GeoDataFrame(data, geometry='Coordinates')
    
    return gdf