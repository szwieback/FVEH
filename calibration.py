'''
Created on Nov 8, 2018

@author: Zwieback
'''

import numpy as np
import copy, os, datetime, string
import itertools
import collections

from paths import pathcalibration
from model import (setup_slump, slump_parameters, integration_parameters, T_initial,
                   integrate_slump)
from slump import SlumpResults

misfit_parameters_null = [{'variable': 'T', 'depth': 0.05, 'type': 'least_squares',
                           'obs_values': None, 'obs_dates': None}]


class CalibrationModule(object):

    def __init__(
            self, baseline_slump_parameters=slump_parameters,
            integration_parameters=integration_parameters, T_initial=T_initial,
            misfit_parameters=misfit_parameters_null,
            date_initial=datetime.datetime(2018, 6, 1, 12, 10),
            date_final=datetime.datetime(2018, 8, 28, 12, 10),
            calibration_parameters={'beta': (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)},
            same_roughness_length=False, forcing_MERRA=False, output_path=None,
            name='calibrationstd', save_figures=True):
        
        self.baseline_slump_parameters = baseline_slump_parameters
        self.integration_parameters = integration_parameters
        self.T_initial = T_initial
        self.date_initial = date_initial
        self.date_final = date_final
        self.same_roughness_length = same_roughness_length
        self.misfit_parameters = misfit_parameters
        self.calibration_parameters = collections.OrderedDict(calibration_parameters)
        self._calibration_grid = tuple(
            itertools.product(*self.calibration_parameters.values()))
        self.forcing_MERRA = forcing_MERRA
        self.name = name
        self.save_figures = save_figures,
        if output_path is None:
            self.output_path = os.path.join(pathcalibration, name)
        else:
            self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
    def _filename_grid_point(self, index_grid, figure=False):
        if not figure:
            fn = os.path.join(self.output_path, 'output_' + str(index_grid) + '.p')
        else:
            fn = os.path.join(self.output_path, 'output_' + str(index_grid) + '.pdf')
        return fn
    
    def _filename_summary_default(self):
        return os.path.join(self.output_path, 'summary.csv')
    
    def _run_grid_point(self, index_grid):
        fnout = self._filename_grid_point(index_grid)
        fnoutfig = self._filename_grid_point(index_grid, figure=True)
        if not self.save_figures:
            fnoutfig = None
        # modify slump parameters; pay attention to same_roughness_length
        slump_parameters_grid_point = copy.deepcopy(self.baseline_slump_parameters)
        parameters_grid_point = self._calibration_grid[index_grid]
        for jparameter, parameter in enumerate(self.calibration_parameters):
            if parameter == 'debris_thickness':  # for convenience
                debrist = parameters_grid_point[jparameter]
                # only implemented for two-layer structure
                assert len(slump_parameters_grid_point['constituents']) == 2
                # top layer unchanged (i.e. debris thickness > 0)
                assert debrist > 0
                # also must be smaller than grid size
                assert debrist < slump_parameters_grid_point['mesh_length']
                # lower layer: non-melting massive ice; may be changed
                if (slump_parameters_grid_point.has_key('thermal_properties') and
                    slump_parameters_grid_point.has_key('thermal_properties') is not None):
                    tp_temp = slump_parameters_grid_point['thermal_properties']
                    slump_parameters_grid_point['constituents'] = [(), ()]
                    slump_parameters_grid_point['constituents'][0] = (0.0,
                                        tp_temp.output(constituents_only=True))
                    slump_parameters_grid_point['thermal_constants'] = (
                        tp_temp.output(parameters_only=True))
                    slump_parameters_grid_point['thermal_properties'] = None

                constituents_lower = (debrist,
                    {'theta_t':0.0, 'theta_m':0.0, 'theta_o':0.0, 'theta_n':1.0})

                slump_parameters_grid_point['constituents'][-1] = constituents_lower
            else:
                slump_parameters_grid_point[parameter] = parameters_grid_point[jparameter]
                if parameter == 'z0m' and self.same_roughness_length:
                    slump_parameters_grid_point['z0a'] = slump_parameters_grid_point['z0m']
        ts = setup_slump(slump_parameters=slump_parameters_grid_point,
                         integration_parameters=self.integration_parameters,
                         T_initial=self.T_initial, date_initial=self.date_initial,
                         date_final=self.date_final, forcing_MERRA=self.forcing_MERRA)
        ts = integrate_slump(ts, date_final=self.date_final,
                             viewer_variable=None, fnout=fnout, fnoutfig=fnoutfig)
        
    def gridSearch(self, n_jobs=32, overwrite=False, write_summary=False,
                   fnsummary=None):
        if overwrite or not os.path.exists(self._filename_grid_point(0)):
            from joblib import Parallel, delayed
            Parallel(n_jobs=n_jobs)(delayed(self._run_grid_point)(index_grid)
                                  for index_grid in range(len(self._calibration_grid)))
        misfits = []
        if write_summary:
            if fnsummary is None:
                fnsummary = self._filename_summary_default()
            summary = []
            header = 'index,' + string.join(list(self.calibration_parameters.keys()),
                                ',') + ',misfit'
            summary.append(header)

        for index_grid in range(len(self._calibration_grid)):
            misfit = self._misfit_grid_point(index_grid, overwrite=False)
            misfits.append(misfit)
            if write_summary:
                summary_line = str(index_grid) + ','
                summary_line = summary_line + string.join(
                    ['{:.3f}'.format(param) for param in self._calibration_grid[index_grid]],
                    sep=',')
                summary_line = summary_line + ',' + '{:.3f}'.format(misfit)
                summary.append(summary_line)
        if write_summary:
            with open(fnsummary, 'w') as fsummary:
                fsummary.writelines([line + '\n' for line in summary])
        return np.argmin(misfits)

    def _misfit_grid_point(self, index_grid, overwrite=False):
        fn = self._filename_grid_point(index_grid)
        if overwrite or not os.path.exists(fn):
            self._run_grid_point(index_grid)
        sr = SlumpResults.fromFile(fn)
        # to do: misfit
        misfit = 0.0
        try:
            for misfit_parameters_objective in self.misfit_parameters:
                values_predicted = sr.readVariable(
                    variable_name=misfit_parameters_objective['variable'],
                    interp_dates=misfit_parameters_objective['obs_dates'],
                    interp_depths=(misfit_parameters_objective['depth'],))
                if misfit_parameters_objective['type'] == 'least_squares':
                    misfit_objective = np.nanmean((
                        values_predicted - misfit_parameters_objective['obs_values']) ** 2)
                    weight_objective = misfit_parameters_objective['weight'] \
                        if 'weight' in misfit_parameters_objective else 1.0
                    misfit = misfit + weight_objective * misfit_objective
                else:
                    raise NotImplementedError
        except:
            misfit = np.nan
        return misfit
