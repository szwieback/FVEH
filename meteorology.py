'''
Created on Mar 28, 2018

@author: zwieback
'''
import numpy as np
from scipy.constants import Stefan_Boltzmann
import datetime

# humidity conversion
def mixingRatioFromSpecificHumidity(q):
    r = q / (1 - q)
    return r

def mixingRatioFromVapourPressure(p, e):
    r = 0.622 * e / (p - e)
    return r

def specificHumidityFromMixingRatio(r):
    q = r / (1 + r)
    return q

def relativeHumidityFromSpecificHumidity(T, q, p, Pascal=False):
    # T in C; # p in Pa
    rh = (vapourPressureFromMixingRatio(mixingRatioFromSpecificHumidity(q), p)
          / saturationVapourPressure(T, Pascal=Pascal))
    return rh

def vapourPressureFromRelativeHumidity(rh, T, Pascal=False):
    vp = rh * saturationVapourPressure(T, Pascal=Pascal)
    return vp

def saturationVapourPressure(T, Pascal=False):
    # T is in Celsius
    # output in hPa unless Pascal
    es = 6.1094 * np.exp((17.625 * T) / (T + 243.04))
    if Pascal:
        es = es * 100
    return es

def vapourPressureFromMixingRatio(r, p):
    # r: mixing ratio; p: pressure
    e = r * p / (r + 0.622)
    return e

def solarAngles(lat=68.75, lon=-133.5,
                   timeutc=[datetime.datetime(2016, 6, 21, 6, 0)]):
    # timeutc can also be an array
    import ephem
    observer = ephem.Observer()
    observer.lon, observer.lat = lon / 180.0 * np.pi, lat / 180.0 * np.pi
    sun = ephem.Sun()
    try:  # single timeutc
        observer.date = timeutc
        sun.compute(observer)
        elevation = float(sun.alt)
        azimuth = float(sun.az)
    except:  # list-like
        elevation = []
        azimuth = []
        for timeutcj in timeutc:
            observer.date = timeutcj
            sun.compute(observer)
            elevation.append(float(sun.alt))
            azimuth.append(float(sun.az))  # az from north, clockwise
    return np.array(elevation), np.array(azimuth)

def diffuseShortWave(S_horizontal_observed, solar_elevation):
    # Reindl et al. approach
    solar_constant = 1367.0
    E0 = 1.0  # no eccentricity correction
    S_extraterrestrial = E0 * solar_constant * np.sin(solar_elevation)
    kt = S_horizontal_observed / S_extraterrestrial
    ind = np.nonzero(kt > 1.0)[0]
    kt[ind] = 1.0
    kd = 0 * kt
    ind = np.nonzero(kt <= 0.3)[0]
    kd[ind] = 1.02 - 0.254 * kt[ind] + 0.0123 * np.sin(solar_elevation[ind])
    ind = np.nonzero(np.logical_and(kt > 0.3, kt <= 0.78))[0]
    kd[ind] = 1.4 - 1.749 * kt[ind] + 0.177 * np.sin(solar_elevation[ind])
    ind = np.nonzero(kt > 0.78)[0]
    kd[ind] = 0.486 * kt[ind] - 0.182 * np.sin(solar_elevation[ind])
    S_diffuse = kd * S_horizontal_observed
    ind = np.nonzero(solar_elevation <= 0)[0]
    S_diffuse[ind] = S_horizontal_observed[ind]
    ind = np.nonzero(S_diffuse > S_horizontal_observed)[0]
    S_diffuse[ind] = S_horizontal_observed[ind]
    return S_diffuse

def normalVector(elevation, azimuth=None):
    # N, E, z
    return np.stack([np.cos(azimuth) * np.cos(elevation), np.sin(azimuth) * np.cos(elevation),
              np.sin(elevation)])

# shortwave: direct/diffuse
# only diffuse radiation is multiplied by sky_view (direct: no obstacle [lake])
# from doi:10.3189/002214310791968430
# but per unit surface (rather than horizontal surface), as in doi:10.3189/2015JoG14J194
def receivedShortwaveSurface(S_horizontal_observed, lat=68.0, lon=-120.0,
                            azimuth_surface=0.0, azimuth_surface_degrees=True,
                            tilt_surface=90.0, tilt_surface_degrees=True,
                            timeutc=[datetime.datetime(2016, 6, 21, 6, 0)],
                            view_sky=0.5, albedo_terrain=0.1,
                            taper_direct_elevation_rad=10.0 * np.pi / 180):
    # azimuth_surface: from north
    # tilt_surface: 0: horizontal, 90: vertical
    # need taper (although can be set to None)
    # because diffuse partitioning gets very inaccurate for low zenith angles
    if azimuth_surface_degrees:
        azimuth_surface_rad = azimuth_surface * np.pi / 180.0
    else:
        azimuth_surface_rad = azimuth_surface
    if tilt_surface_degrees:
        tilt_surface_rad = tilt_surface * np.pi / 180.0
    else:
        tilt_surface_rad = tilt_surface
    solar_elevation, solar_azimuth = solarAngles(lat=lat, lon=lon, timeutc=timeutc)
    S_diffuse = diffuseShortWave(S_horizontal_observed, solar_elevation)
    S_direct_perpendicular = (S_horizontal_observed - S_diffuse) / np.sin(solar_elevation)
    S_direct_perpendicular[np.isnan(S_direct_perpendicular)] = 0.0
    if taper_direct_elevation_rad is not None:
        ind = np.nonzero(solar_elevation < taper_direct_elevation_rad)[0]
        S_direct_perpendicular[ind] = S_direct_perpendicular[ind] * (
            solar_elevation[ind] / taper_direct_elevation_rad)
    normal_vector_surface = normalVector(
        np.pi / 2 - tilt_surface_rad, azimuth=azimuth_surface_rad)
    normal_vector_sun = normalVector(
        solar_elevation, azimuth=solar_azimuth)
    cos_factor = np.sum(normal_vector_surface[:, np.newaxis] * normal_vector_sun, axis=0)
    cos_factor[cos_factor < 0.0] = 0.0
    S_direct = S_direct_perpendicular * cos_factor
    # rough approximation; no shadowing, directional effects, sun glint over lake, etc.
    S_terrain = albedo_terrain * S_horizontal_observed * (1 - view_sky)
    S_surface = S_diffuse * view_sky + S_terrain + S_direct
    return S_surface

# longwave
def emittedLongwave(T_surf, celsius=True, emissivity=0.96):
    offset = 273.15 if celsius else 0.0
    return emissivity * Stefan_Boltzmann * (T_surf + offset) ** 4

def receivedLongwavePartition(
        L_sky, T_surf_terrain, celsius=True, emissivity_terrain=0.96, view_sky=0.5):
    # simple, as in doi:10.3189/2014JoG13J045
    L_terrain = emittedLongwave(T_surf_terrain, celsius=celsius,
                                    emissivity=emissivity_terrain)
    return view_sky * L_sky + (1 - view_sky) * L_terrain
