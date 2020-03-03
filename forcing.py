'''
Created on May 1, 2018

@author: zwieback
'''

import pandas as pd
import pytz
import numpy as np
import csv
from datetime import datetime, timedelta
from scipy.constants import Stefan_Boltzmann
import _strptime  # for some reason needed for multithreading

from meteorology import (vapourPressureFromRelativeHumidity, receivedLongwavePartition,
                         receivedShortwaveSurface)
from paths import fncsv

tzTVC = pytz.timezone('Etc/GMT+7')

def readTVCForcingDataFrame(
        fncsv=fncsv, tz=tzTVC, mindate=None, maxdate=None, emissivity_terrain=0.96):

    outpfields = [('Snow Depth Corrected (m)', 'snow'),
                  ('Platinum Resistance Thermometer (degC)', 'Tplat'),
                  ('Air Temp (degC)', 'Tair'),
                  ('Relative Humidity Corrected (%)', 'RH'),
                  ('Precipitation Tipping Bucket (mm)', 'Pt'),
                  ('Wind Speed at 10m (m/s)', 'wind'),
                  ('Incoming Shortwave Radiation (W*m-2)', 'Sinhorizontal'),
                  ('Outgoing Shortwave Radiation (W*m-2)', 'Sout'),
                  ('Incoming Longwave Radiation (W*m-2)', 'Linhorizontal'),
                  ('Outgoing Longwave Radiation (W*m-2)', 'Lout'),
                  ('Weighing Precip Gauge VW123 Corrected (mm)', 'Ptwc'),
                  ('Barometric Pressure (mb)', 'p'),
                  ('Relative Humidity Corrected (%)', 'rh')]
    outpfieldsdiff = {'Ptwc':'Ptw'}
    nanvalues = {'Ptwc':-99999.0}
    listdicts = []
    with open(fncsv, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            year = int(row['Year'])
            jd = int(row['Day of Year'])
            hourstring = str((int(row['Hour & Minute']))).zfill(4)
            hours = int(hourstring[0:2])
            minutes = int(hourstring[2:4])
            dt1 = (datetime(year, 1, 1) + timedelta(jd - 1) +
                   timedelta(hours=hours, minutes=minutes))
            dt2 = tz.localize(dt1, is_dst=False)
            dt_utc = dt2.astimezone(pytz.utc)
            outpdict = {'time':dt_utc}
            if mindate is None or pytz.utc.localize(mindate) <= dt_utc:
                for opf in outpfields:
                    try:
                        outpdict[opf[1]] = float(row[opf[0]])
                        if opf[1] in nanvalues.keys():
                            if nanvalues[opf[1]] == outpdict[opf[1]]:
                                outpdict[opf[1]] = np.nan
                        if opf[1] in outpfieldsdiff.keys():
                            if len(listdicts) == 0:
                                outpdict[outpfieldsdiff[opf[1]]] = np.nan
                            else:
                                outpdict[outpfieldsdiff[opf[1]]] = (
                                    outpdict[opf[1]] - listdicts[-1][opf[1]])
                    except:
                        outpdict[opf[1]] = np.nan

                outpdict['Linhorizontal'] = (
                    outpdict['Linhorizontal']
                    +Stefan_Boltzmann * (273.15 + outpdict['Tplat']) ** 4)
                outpdict['Lout'] = (
                    outpdict['Lout'] + 5.67e-8 * (273.15 + outpdict['Tplat']) ** 4)
                outpdict['Tsurfhorizontal'] = (
                    (outpdict['Lout'] / (emissivity_terrain * Stefan_Boltzmann)) ** (0.25)
                    -273.15)
                if np.isnan(outpdict['Tsurfhorizontal']):
                    outpdict['Tsurfhorizontal'] = outpdict['Tair']
                outpdict['p'] = outpdict['p'] * 100  # Pascal
                outpdict['VPair'] = vapourPressureFromRelativeHumidity(
                    0.01 * outpdict['rh'], outpdict['Tair'], Pascal=True)
                listdicts.append(outpdict)
            if maxdate is not None and dt_utc > pytz.utc.localize(maxdate):
                break
    df = pd.DataFrame(listdicts)
    df['time'] = df.time.dt.tz_localize(None)  # already in UTC, but avoids pandas trouble
    df = df.interpolate(limit_direction='both')
    return df

def forcingFromDataFrame(
        df, emissivity_terrain=0.96, view_sky=None, lat=68.75, lon=-133.5,
        azimuth_surface=0.0, azimuth_surface_degrees=True, tilt_surface=90.0,
        tilt_surface_degrees=True, albedo_terrain=0.1,
        taper_direct_elevation_rad=10.0 * np.pi / 180, perturb_forcing=False,
        perturb_deltaT=0.0):
    if view_sky is None:
        tilt_surface_rad = tilt_surface * np.pi / 180.0 if tilt_surface_degrees else tilt_surface
        view_sky = 1 - tilt_surface_rad / np.pi

    if perturb_forcing:
        df = perturbForcing(df, deltaT=perturb_deltaT, Pascal=True)

    # longwave
    df['Lin'] = receivedLongwavePartition(
        df['Linhorizontal'], df['Tsurfhorizontal'], emissivity_terrain=emissivity_terrain,
        view_sky=view_sky, celsius=True)
    # shortwave
    df['Sin'] = receivedShortwaveSurface(
        df['Sinhorizontal'], lat=lat, lon=lon, azimuth_surface=azimuth_surface,
        azimuth_surface_degrees=azimuth_surface_degrees, tilt_surface=tilt_surface,
        tilt_surface_degrees=tilt_surface_degrees, timeutc=df['time'], view_sky=view_sky,
        albedo_terrain=albedo_terrain,
        taper_direct_elevation_rad=taper_direct_elevation_rad)

    values_forcing = ({ind:np.array(df[ind]) for ind in df.columns})
    values_forcing['time'] = [
        datetime.strptime(str(t), '%Y-%m-%dT%H:%M:%S.%f000')
        for t in values_forcing['time']]
    return values_forcing

def perturbForcing(df, deltaT=0.0, Pascal=True):
    df_p = df.copy()
    df_p['Tair'] = df['Tair'] + deltaT
    df_p['Tsurfhorizontal'] = df['Tsurfhorizontal'] + deltaT
    df_p['Linhorizontal'] = Stefan_Boltzmann * (((
        df['Linhorizontal'] / Stefan_Boltzmann) ** 0.25) + deltaT) ** 4
    df_p['VPair'] = vapourPressureFromRelativeHumidity(
        0.01 * df['rh'], df_p['Tair'], Pascal=Pascal)
    return df_p

if __name__ == '__main__':
    df = readTVCForcingDataFrame(mindate=datetime(2018, 6, 1), maxdate=datetime(2018, 9, 1))

#     for tilt_surface in [90, 75, 60, 45, 30, 15, 0]:
#         values_forcing = forcingFromDataFrame(df, tilt_surface = tilt_surface, azimuth_surface = 0.0,
#                                               view_sky = None)
#         print(tilt_surface, np.nansum(values_forcing['Sin']))
#     print(np.nansum(values_forcing['Sinhorizontal']))
