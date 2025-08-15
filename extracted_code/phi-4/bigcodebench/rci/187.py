import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, List, Tuple

DEFAULT_COORD_RANGE = {'Lon': (-180, 180), 'Lat': (-90, 90)}

def task_func(dic: Dict[str, Tuple[float, float]] = DEFAULT_COORD_RANGE, 
              cities: List[str] = ['New York', 'London', 'Beijing', 'Tokyo', 'Sydney']) -> gpd.GeoDataFrame:
    
    required_keys = ['Lon', 'Lat']
    for key in required_keys:
        if key not in dic:
            raise ValueError(f"Dictionary must contain '{key}' key.")
        
        coord_range = dic[key]
        if not isinstance(coord_range, tuple) or len(coord_range)!= 2:
            raise ValueError(f"'{key}' value must be a tuple with two elements.")
        
        if coord_range[0] >= coord_range[1]:
            raise ValueError(f"Invalid range for '{key}': {coord_range}")
    
    lons = np.random.uniform(dic['Lon'][0], dic['Lon'][1], len(cities))
    lats = np.random.uniform(dic['Lat'][0], dic['Lat'][1], len(cities))
    
    points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    
    data = {'City': cities, 'Coordinates': points}
    gdf = gpd.GeoDataFrame(data, geometry='Coordinates')
    
    return gdf