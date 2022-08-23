import pyproj
import numpy as np
from math import radians, cos, sin, asin, sqrt


def ecef2lla(pnt_ecef):
    M = np.array([[-0.20053982643488838, -0.72975436304732511,  0.653637780140389870, 4177139.442822214700], 
                  [0.979685550579095120, -0.14937937302342402,  0.133798448801411370, 855052.7445900742900],
                  [0.000000000000000000,  0.667191406216027790, 0.744886318488585660, 4728408.463954962800],
                  [0.000000000000000000,  0.000000000000000000, 0.000000000000000000, 1.000000000000000000]])
    x = np.append(pnt_ecef.T, np.array([1], ndmin=2), axis=0)
    
    pnt  = M @ x
    
    transformer = pyproj.Transformer.from_crs(
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    )
    
    lon, lat, alt = transformer.transform(pnt[0],pnt[1],pnt[2],radians=False)
    return lon[0], lat[0], alt[0]

def lla2utm(lon, lat):
    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=32, preserve_units=False)
    east, north = proj(lon, lat)
    return east, north

def pnt2utm(pnt):
    lon, lat, _ = ecef2lla(pnt.reshape(1,-1))
    east, north = lla2utm(lon, lat)
    return north, east

# Code From: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r * 1000