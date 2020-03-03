'''
Created on 11 Nov 2017

@author: Simon
'''
from __future__ import print_function, division
from fipy import Viewer
from paths import path0
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os

from slump import ThawSlumpEnthalpy, TimeStepCFLSources, OutputStepHourly, \
    ForcingInterpolation, SlumpResults
from thermal import ThermalPropertiesSoilPhaseChange, \
    thermal_constants_default, ThermalPropertiesSoilPhaseChangeLinearCharacteristic
from boundary import BoundaryCondition1DNoFlux, \
        BoundaryCondition1DEnergyBalance
from forcing import forcingFromDataFrame, readTVCForcingDataFrame

# parameters
constituents_def = [(0.0, {'theta_t':0.4, 'theta_m':0.6, 'theta_o':0.0, 'theta_n':0.0}),
                    (1.0, {'theta_t':0.3, 'theta_m':0.7, 'theta_o':0.0, 'theta_n':0.0})]
slump_parameters = {
    'albedo' : 0.15, 'beta' : 0.95, 'emissivity' : 0.96, 'z0m' : 0.005, 'z0a' : 0.005,
    'd0' : 0.00, 'lat':68.75, 'lon':-133.5, 'mesh_type': 'finecoarse', 'mesh_dx_fine': 0.05,
    'mesh_dx_coarse': 0.25, 'mesh_length_fine': 2.0, 'mesh_length': 6.0,
    'emissivity_terrain': 0.96, 'view_sky': None, 'albedo_terrain':0.15,
    'azimuth_surface':0.0, 'azimuth_surface_degrees': True, 'tilt_surface': 30.0,
    'tilt_surface_degrees': True, 'taper_direct_elevation_rad': 10.0 * np.pi / 180,
    'thermal_model': 'LinearCharacteristic', 'thermal_constants': thermal_constants_default,
    'constituents': constituents_def, 'perturb_forcing': False, 'perturb_deltaT': 0.0 }

integration_parameters = {'safety': 0.9, 'relative_enthalpy_change': 5e-2}
T_initial = np.array(-3.0)
T_initial2 = np.array(-15.0)
T_initialmelt = np.array(0.0)
date_initial = datetime.datetime(2017, 6, 1, 12, 10, 0)
date_final = datetime.datetime(2017, 6, 15, 12, 10, 0)

def create_constituent_profiles(slump_parameters):
    meshtemp = create_mesh(slump_parameters)
    faces_top = meshtemp.mesh.faceCenters[0]
    faces_top_depth = -(faces_top - meshtemp.mesh.faceCenters[0][-1])
    indices_top = np.array([np.argmin(np.abs(faces_top_depth - stratum[0]))
                            for stratum in slump_parameters['constituents'][::-1]])
    indices = np.concatenate(([0], indices_top))
    constituent_profiles = {}
    for jstratum, stratum in enumerate(slump_parameters['constituents'][::-1]):
        for constituent in stratum[1]:
            constituent_stratum = (stratum[1][constituent]
                                   * np.ones(indices[jstratum + 1] - indices[jstratum]))
            if jstratum == 0:
                constituent_profiles[constituent] = constituent_stratum
            else:
                constituent_profiles[constituent] = np.concatenate(
                    (constituent_profiles[constituent], constituent_stratum))
    return constituent_profiles

def create_mesh(slump_parameters):
    if slump_parameters['mesh_type'] == 'regular':
        from mesh import OneDMeshRegular
        tsmesh = OneDMeshRegular(slump_parameters['mesh_nx'], slump_parameters['mesh_dx'])
    elif slump_parameters['mesh_type'] == 'finecoarse':  # 2.5 cm surface cell too slow
        n_fine = int((slump_parameters['mesh_length_fine'])
                     / slump_parameters['mesh_dx_fine'])
        mesh_fine = slump_parameters['mesh_dx_fine'] * np.ones(n_fine)
        n_coarse = int((slump_parameters['mesh_length']
                        -np.sum(mesh_fine)) / slump_parameters['mesh_dx_coarse'])
        dx_coarse = (slump_parameters['mesh_length'] - np.sum(mesh_fine)) / n_coarse
        mesh_coarse = dx_coarse * np.ones(n_coarse)
        from mesh import OneDMeshIrregular
        tsmesh = OneDMeshIrregular(np.concatenate([mesh_coarse, mesh_fine]))
    else:
        raise NotImplementedError
    return tsmesh

def setup_slump(
        slump_parameters=slump_parameters, integration_parameters=integration_parameters,
        T_initial=T_initial, date_initial=date_initial, date_final=date_final,
        forcing_MERRA=False):
    # mesh
    tsmesh = create_mesh(slump_parameters)

    # stratigraphy
    slump_constituent_profiles = create_constituent_profiles(slump_parameters)
    # time step module
    time_step_module = TimeStepCFLSources(
        safety=integration_parameters['safety'],
        relative_enthalpy_change=integration_parameters['relative_enthalpy_change'])

    # forcing
    # consider passing df directly
    if forcing_MERRA:
        import pandas as pd
        from paths import fndfmerra
        df = pd.read_pickle(fndfmerra)
    else:
        df = readTVCForcingDataFrame(mindate=date_initial - datetime.timedelta(days=10),
                                     maxdate=date_final + datetime.timedelta(days=10))

    values_forcing = forcingFromDataFrame(
        df, emissivity_terrain=slump_parameters['emissivity_terrain'],
        albedo_terrain=slump_parameters['albedo_terrain'],
        view_sky=slump_parameters['view_sky'], lat=slump_parameters['lat'],
        lon=slump_parameters['lon'], azimuth_surface=slump_parameters['azimuth_surface'],
        azimuth_surface_degrees=slump_parameters['azimuth_surface_degrees'],
        tilt_surface=slump_parameters['tilt_surface'],
        tilt_surface_degrees=slump_parameters['tilt_surface_degrees'],
        taper_direct_elevation_rad=slump_parameters['taper_direct_elevation_rad'],
        perturb_forcing=slump_parameters['perturb_forcing'],
        perturb_deltaT=slump_parameters['perturb_deltaT'])

    forcing_module = ForcingInterpolation(values_forcing)

    # thermal_properties
    if (slump_parameters.has_key('thermal_properties')
         and slump_parameters['thermal_properties'] is not None):
        thermal_properties = slump_parameters['thermal_properties']
    elif slump_parameters['thermal_model'] == 'PhaseChange':
        thermal_properties = ThermalPropertiesSoilPhaseChange(
                            parameters=slump_parameters['thermal_constants'],
                            theta_t=slump_constituent_profiles['theta_t'],
                            theta_m=slump_constituent_profiles['theta_m'],
                            theta_o=slump_constituent_profiles['theta_o'],
                            theta_n=slump_constituent_profiles['theta_n'])
    elif slump_parameters['thermal_model'] == 'LinearCharacteristic':
        thermal_properties = ThermalPropertiesSoilPhaseChangeLinearCharacteristic(
                            parameters=slump_parameters['thermal_constants'],
                            theta_t=slump_constituent_profiles['theta_t'],
                            theta_m=slump_constituent_profiles['theta_m'],
                            theta_o=slump_constituent_profiles['theta_o'],
                            theta_n=slump_constituent_profiles['theta_n'])
    else:
        raise NotImplementedError
    # boundary conditions
    bc_inside = BoundaryCondition1DNoFlux(side='inside')
    # bc_headwall = BoundaryCondition1DNoFlux(side = 'headwall')
    # bc_headwall = BoundaryCondition1DConstantTemperature(side = 'headwall')
    # bc_headwall = BoundaryCondition1DNewtonLawCooling(coeff_newton = 0.1, side = 'headwall')
    bc_headwall = BoundaryCondition1DEnergyBalance(slump_parameters, side='headwall')

    # initialize thaw slump and viewer
    if len(T_initial.shape) == 0:  # only 0-length array provided
        T_initial = T_initial * np.ones_like(tsmesh.cell_volumes)
    ts = ThawSlumpEnthalpy(
        tsmesh, output_step_module=OutputStepHourly(), time_step_module=time_step_module,
        forcing_module=forcing_module, thermal_properties=thermal_properties,
        time_initial=date_initial, T_initial=T_initial, bc_inside=bc_inside,
        bc_headwall=bc_headwall)
    ts.addStoredVariable(['Tair', 'VPair'])
    return ts

def integrate_slump(ts, date_final=date_final, fnout=None, fnoutfig=None,
                    viewer_variable='T', viewer_min=-5, viewer_max=30):
    if viewer_variable is not None:
        viewer = Viewer(vars=(ts.variables[viewer_variable]), datamin=viewer_min,
                        datamax=viewer_max, title=ts.date)
    else:
        viewer = None
    # integrate
    ts.integrate(date_final, viewer=viewer)
    if viewer is not None:
        plt.close(viewer.id)
    if fnoutfig is not None:
        plot_time_series(ts, fnout=fnoutfig)
    if fnout is not None:
        ts.exportOutput(fnout)
    return ts

def plot_time_series(ts, fnout=None, depths=[0.35, 0.2, 0.05]):
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(nrows=3, sharex=True)
    plt.subplots_adjust(right=0.79, top=0.98, bottom=0.05)
    outputtuple = ts._output_module.read()
    sr = SlumpResults(*outputtuple)
    outp = outputtuple[1]
    outpdates = ts._output_module.read()[0]
    lw = 0.5
    cols = {'red': '#993333', 'grey': '#999999', 'blue':'#333399',
            'dark': '#000000', 'lightred': '#cc9999'}
    colslist = ['dark', 'red', 'lightred']
    ax[0].plot(outpdates, outp['flux_sensible_headwall'], lw=lw,
               c=cols['red'], label='sensible')
    ax[0].plot(outpdates, outp['flux_radiation_headwall'], lw=lw,
               c=cols['grey'], label='radiation')
    ax[0].plot(outpdates, outp['flux_latent_headwall'], lw=lw,
               c=cols['blue'], label='latent')
    ax[0].plot(outpdates, outp['flux_headwall'], lw=lw,
               c=cols['dark'], label='total')
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0, 0.25, 1))
    ax[1].plot(outpdates, outp['Tair'], lw=lw, c=cols['grey'], label='air')
    ax[1].plot(outpdates, outp['Tsurf_headwall'], lw=lw,
               c=cols['dark'], label='surface')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0, 0.25, 1))
    # temperature profile
    ax[2].plot(outpdates, outp['Tair'], lw=lw, c=cols['grey'], label='air')
    for jdepth, depth in enumerate(depths):
        ax[2].plot(outpdates, sr.readVariable(variable_name='T',
                interp_depths=depth), c=cols[colslist[jdepth]], lw=lw,
                label=str(depth))
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0, 0.25, 1))
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    # ax[1].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1,15], interval=1))
    # print(np.sum(outp['flux_latent_headwall'])/np.sum(outp['flux_radiation_headwall']))
    # print(np.sum(outp['flux_sensible_headwall'])/np.sum(outp['flux_radiation_headwall']))
    # print(np.sum(outp['flux_headwall'])/np.sum(outp['flux_radiation_headwall']))
    if fnout is None:
        plt.show(block=True)
    else:
        fig.savefig(fnout)

def integrate_slump_helper(index):
    fnout = os.path.join(path0, 'output_' + str(index) + '.p')
    fnoutfig = os.path.join(path0, 'output_' + str(index) + '.pdf')
    ts = setup_slump(
        slump_parameters=slump_parameters, integration_parameters=integration_parameters,
        T_initial=T_initial, date_initial=date_initial, date_final=date_final)
    ts = integrate_slump(
        ts, date_final=date_final, viewer_variable=None, fnout=fnout, fnoutfig=fnoutfig)

def integrate_slump_parallel(helper_function=integrate_slump_helper, n=4,
                             n_jobs=32):
    from joblib import Parallel, delayed
    Parallel(n_jobs=n_jobs)(delayed(helper_function)(i) for i in range(n))

if __name__ == '__main__':
    import time
    fnout = os.path.join(path0, 'output.p')
    fnoutfig = os.path.join(path0, 'output.pdf')

    date_initial = datetime.datetime(2017, 6, 1, 12, 10, 0)
    date_final = datetime.datetime(2017, 8, 30, 12, 10, 0)
    from slump_properties import slump_parameters_instrumented
    slump_parameters = slump_parameters_instrumented('Top', replace_thermal_properties=True)

    # tic = time.time()
    # ts = setup_slump(slump_parameters = slump_parameters,
    #                date_initial = date_initial, date_final = date_final)
    # ts = integrate_slump(ts, date_final = date_final,
    #                      viewer_variable = None, viewer_min = -5, viewer_max = 30,
    #                      fnout = fnout, fnoutfig = fnoutfig)
    # print(ts.variables['theta_m'].value)
    # print(ts.variables['h'].value)
    # print(ts.variables['T'].value)
    # print(time.time() - tic)

    sr = SlumpResults.fromFile(fnout)
    print(np.mean(sr.data['flux_latent_headwall']))
    print(np.mean(sr.data['flux_sensible_headwall']))

    # tic = time.time()
    # integrate_slump_parallel(n = 32)
    # print(time.time() - tic)
