'''
Created on 19 Nov 2017

@author: Simon
'''
from fipy import Variable, FaceVariable, CellVariable, TransientTerm, DiffusionTerm
import numpy as np
import datetime
import pickle
from scipy.interpolate import interp1d

from boundary import BoundaryConditionCollection1D
from diagnostic import DiagnosticModule

class ThawSlump(object):  # 1D

    # time_initial only works when forcing is provided
    def __init__(
            self, tsmesh, time_step_module=None, output_step_module=None,
            forcing_module=None, thermal_properties=None, time_initial=None):
        self.mesh = tsmesh
        self.variables = {}
        self.variables_store = []
        self.diagnostic_modules = {}
        self.diagnostic_update_order = []
        self.eq = None
        self.boundary_condition_collection = None
        self._time = Variable(value=0)
        self._time_step_module = time_step_module
        self._timeref = None  # will generally be set by forcing_module; otherwise manually
        if forcing_module is not None:
            self.initializeForcing(forcing_module)
            if time_initial is not None:
                self.time = time_initial
        if thermal_properties is not None:
            self.initializeThermalProperties(thermal_properties)

        self._output_step_module = output_step_module
        self._output_module = SlumpOutput()
        if output_step_module is None:
            self._output_step_module = OutputStep()

    @property
    def time(self):
        return float(self._time.value)

    @time.setter
    def time(self, t):  # can also handle date objects
        try:
            self.date = t
        except:
            self._time.setValue(t)

    @property
    def timeStep(self):
        return self._time_step_module.calculate(self)

    @property
    def date(self):
        return self._internal_time_to_date(self.time)

    def _internal_time_to_date(self, internal_time):
        return self._timeref + datetime.timedelta(seconds=internal_time)

    @date.setter
    def date(self, d):
        dtsec = self._date_to_internal_time(d)
        self._time.setValue(dtsec)

    def _date_to_internal_time(self, d):
        dt = d - self._timeref
        dtsec = dt.days * 24 * 3600 + dt.seconds + dt.microseconds * 1e-6
        return dtsec

    def initializeTimeReference(self, timeref):
        # timeref is a datetime object
        self._timeref = timeref

    def initializePDE(self, tseq=None):
        self.eq = tseq

    def initializeTimeStepModule(self, time_step_module):
        self._time_step_module = time_step_module

    def _initializeSourcesZero(self, source_name='S'):
        self.variables[source_name] = CellVariable(
            name=source_name, mesh=self.mesh.mesh, value=0.0)

    def initializeDiagnostic(
            self, variable, funpointer, default=0.0, face_variable=False,
            output_variable=True):
        if not face_variable:
            self.variables[variable] = CellVariable(
                name=variable, mesh=self.mesh.mesh, value=default)
        else:
            self.variables[variable] = FaceVariable(
                name=variable, mesh=self.mesh.mesh, value=default)
        self.diagnostic_modules[variable] = DiagnosticModule(funpointer, self)
        if output_variable:
            self.variables_store.append(variable)
        self.diagnostic_update_order.append(variable)

    def initializeOutputStepModule(self, output_step_module):
        self._output_step_module = output_step_module

    def initializeThermalProperties(self, thermal_properties):
        self.thermal_properties = thermal_properties
        self.thermal_properties.initializeVariables(self)
        self.initializeTright()

    def initializeForcing(self, forcing_module):
        self.forcing_module = forcing_module
        for varj in self.forcing_module.variables:
            assert varj not in self.variables
            self.variables[varj] = self.forcing_module.variables[varj]
        self.initializeTimeReference(self.forcing_module._timeref)

    def initializeEnthalpyTemperature(self, T_initial, proportion_frozen=None,
                                      time=None):
        # time can be internal time or also a datetime object
        pf = 0.0 if proportion_frozen is None else proportion_frozen
        assert pf >= 0.0 and pf <= 1.0
        self.variables['T'].setValue(T_initial)
        self.variables['h'].setValue(self.thermal_properties.enthalpyFromTemperature(
            self, T=T_initial, proportion_frozen=pf))
        self.updateDiagnostics()
        if time is not None:
            self.time = time

    def updateDiagnostic(self, variable):
        self.variables[variable].setValue(self.diagnostic_modules[variable].evaluate())

    def updateDiagnostics(self, variables=None):
        if variables is not None:
            variablesorder = variables
        else:
            variablesorder = self.diagnostic_update_order
        for variable in variablesorder:
            self.updateDiagnostic(variable)

    def specifyBoundaryConditions(self, boundary_condition_collection):
        self.boundary_condition_collection = boundary_condition_collection
        self.updateGeometryBoundaryConditions()
        self.invokeBoundaryConditions()
        self.initializePDE()

    def updateGeometryBoundaryConditions(self):
        self.boundary_condition_collection.updateGeometry(self)

    def updateBoundaryConditions(self, bc_data, invoke=True):
        self.boundary_condition_collection.update(bc_data)
        if invoke:
            self.invokeBoundaryConditions()

    def invokeBoundaryConditions(self):
        self.boundary_condition_collection.invoke(self)

    def updateGeometry(self):
        self.boundary_condition_collection.updateGeometry(self)

    def nextOutput(self):
        return self._output_step_module.next(self)

    def updateOutput(self, datanew={}):
        for v in self.variables_store:
            datanew[v] = np.copy(self.variables[v].value)
        # boundary condition outputs:
        # separate routine: total source, source components, or for basic b.c. just value)
        datanew.update(self.boundary_condition_collection.output())
        self._output_module.update(self.date, datanew)

    def exportOutput(self, fn):
        self._output_module.export(fn)

    def addStoredVariable(self, varname):
        # varname can also be list
        if isinstance(varname, str):
            if varname not in self.variables_store:
                self.variables_store.append(varname)
        else:  # tuple/list,etc.
            for varnamej in varname:
                self.addStoredVariable(varnamej)

class ThawSlumpEnthalpy(ThawSlump):

    # both boundary conditions bc_inside and bc_headwall have to be provided,
    # and they are only activated when forcing and thermal_properties are also given
    def __init__(
            self, tsmesh, time_step_module=None, output_step_module=None, h_initial=0.0,
            T_initial=None, time_initial=None, proportion_frozen_initial=None,
            forcing_module=None, thermal_properties=None, bc_inside=None, bc_headwall=None):
        # T_initial only works if thermal_properties are provided
        ThawSlump.__init__(
            self, tsmesh, time_step_module=time_step_module,
            output_step_module=output_step_module, time_initial=time_initial,
            forcing_module=forcing_module, thermal_properties=thermal_properties)
        self._initializeSourcesZero(source_name='S')
        self._initializeSourcesZero(source_name='S_inside')
        self._initializeSourcesZero(source_name='S_headwall')
        # specific volumetric enthalpy
        self.variables['h'] = CellVariable(
            name='h', mesh=self.mesh.mesh, value=h_initial, hasOld=True)
        self.addStoredVariable('h')
        if T_initial is not None:  # essentially overrides h_initial
            self.initializeEnthalpyTemperature(
                T_initial, proportion_frozen=proportion_frozen_initial)
        if (bc_inside is not None and bc_headwall is not None
            and self.thermal_properties is not None and self.forcing_module is not None):
            bcc = BoundaryConditionCollection1D(
                bc_headwall=bc_headwall, bc_inside=bc_inside)
            self.specifyBoundaryConditions(bcc)

        self._output_module.storeInitial(self)

    def initializePDE(self):
        self.eq = (TransientTerm(var=self.variables['h']) ==
                   DiffusionTerm(coeff=self.variables['k'], var=self.variables['T']) +
                   self.variables['S'] + self.variables['S_headwall'] +
                   self.variables['S_inside'])

    def initializeTright(self):
        extrapol_dist = (self.mesh.mesh.faceCenters[0, self.mesh.mesh.facesRight()][0]
                         -self.mesh.cell_mid_points)
        self.dxf = CellVariable(mesh=self.mesh.mesh, value=extrapol_dist)
        self.variables['T_right'] = (
            self.variables['T'] + self.variables['T'].grad[0] * self.dxf)

    def updateGeometry(self):
        ThawSlump.updateGeometry(self)
        self.initializeTright()

    def _integrate(
            self, time_step, max_time_step=None, residual_threshold=1e-3, max_steps=20):
        apply_max_time_step = False
        if time_step is None:
            time_step = self.timeStep
        if max_time_step is not None and time_step > max_time_step:
            time_step = max_time_step
            apply_max_time_step = True
        residual = residual_threshold + 1
        steps = 0
        assert self._timeref == self.forcing_module._timeref
        self.forcing_module.evaluateToVariable(t=self.time)
        while residual > residual_threshold:
            residual = self.eq.sweep(var=self.variables['h'], dt=time_step)
            steps = steps + 1
            if steps >= max_steps:
                raise RuntimeError('Sweep did not converge')
        self.time = self.time + time_step
        self.variables['h'].updateOld()
        self.updateDiagnostics()
        return time_step, apply_max_time_step

    def integrate(
            self, time_end, time_step=None, residual_threshold=1e-2, max_steps=10, 
            time_start=None, viewer=None):
        # time_end can also be date
        if time_start is not None:
            self.time = time_start
        self.variables['h'].updateOld()
        try:
            interval = time_end - self.time
            time_end_internal = time_end
        except:
            time_end_internal = self._date_to_internal_time(time_end)

        time_output = self.nextOutput()
        write_output = False
        write_output_limit = False
        time_steps = []

        while self.time < time_end_internal:
            max_time_step = time_end_internal - self.time
            if time_output is not None and time_output < time_end_internal:
                max_time_step = time_output - self.time
                write_output_limit = True
            time_step_actual, apply_max_time_step = self._integrate(
                time_step, max_time_step=max_time_step)
            time_steps.append(time_step_actual)
            if apply_max_time_step and write_output_limit:
                write_output = True

            if viewer is not None:
                viewer.plot()
                viewer.axes.set_title(self.date)
            if write_output:
                time_output = self.nextOutput()
                write_output = False
                write_output_limit = False
                # actually write output
                datanew = {'nsteps':len(time_steps), 'mean_time_step':np.mean(time_steps)}
                self.updateOutput(datanew=datanew)
                time_steps = []


class SlumpOutput(object):
    def __init__(self):
        self.dates = []
        self.data = {}
        self.initial = {}
        
    def update(self, date, datanew):
        records = set(self.data.keys() + datanew.keys())
        for record in records:
            if record in self.data and record in datanew:
                self.data[record].append(datanew[record])
            elif record in self.data:
                self.data[record].append(None)
            else:
                # new record; fill with Nones
                self.data[record] = [None] * len(self.dates)
                self.data[record].append(datanew[record])
        self.dates.append(date)
        
    def storeInitial(self, ts):
        self.initial['mesh_mid_points'] = ts.mesh.cell_mid_points
        self.initial['mesh_face_left'] = ts.mesh.face_left_position
        self.initial['mesh_face_right'] = ts.mesh.face_right_position
        self.initial['mesh_cell_volumes'] = ts.mesh.cell_volumes
        self.initial['T_initial'] = np.copy(ts.variables['T'].value)
        self.initial.update(ts.thermal_properties.output())
        
    def export(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump(self.read(), f)
            
    def read(self):
        return (self.dates, self.data, self.initial)

# nice way to read pickled SlumpOutput data (read method)
class SlumpResults(object):  
    def __init__(self, dates, data, initial, timeref=None):
        self.dates = dates
        self.data = data
        self.initial = initial
        if timeref is not None:
            self._timeref = timeref
        else:
            self._timeref = self.dates[0]
            
    @classmethod
    def fromFile(cls, fn):
        dates, data, initial = pickle.load(open(fn, 'rb'))
        return cls(dates, data, initial)
    
    def _date_to_internal_time(self, ds):
        # ds is list
        dts = [d - self._timeref for d in ds]
        dtsec = [dt.days * 24 * 3600 + dt.seconds + dt.microseconds * 1e-6 for dt in dts]
        return np.array(dtsec)
    
    @property
    def _depths(self):
        return self.initial['mesh_face_right'] - self.initial['mesh_mid_points']
    
    def readVariable(self, variable_name='T', interp_dates=None, interp_depths=None):
        vararr = np.array(self.data[variable_name])
        if interp_dates is not None:
            dates_int = self._date_to_internal_time(self.dates)
            interp_dates_int = self._date_to_internal_time(interp_dates)
            interpolator_dates = interp1d(dates_int, vararr, axis=0)
            vararr = interpolator_dates(interp_dates_int)
        if interp_depths is not None:
            # check dimensions
            assert len(vararr.shape) == 2
            assert vararr.shape[1] == self.initial['mesh_mid_points'].shape[0]
            # interpolate
            interpolator_depths = interp1d(self._depths, vararr, axis=1)
            vararr = interpolator_depths(interp_depths)
        return vararr


class TimeStep(object):
    def __init__(self):
        pass
    
    def calculate(self, ts):
        pass


class TimeStepConstant(TimeStep):
    def __init__(self, step=1.0):
        self.step = step
        
    def calculate(self, ts):
        return self.step


class TimeStepCFL(TimeStep):
    def __init__(self, safety=0.9):
        self.safety = safety
        
    def calculate(self, ts):
        K = np.array(ts.variables['K'])
        K = 0.5 * (K[1::] + K[:-1:])
        CFL = np.min(0.5 * (ts.mesh.cell_volumes) ** 2 / np.array((K / ts.variables['C'])))
        step = self.safety * CFL
        return step


class TimeStepCFLSources(TimeStep):
    def __init__(
            self, safety=0.9, relative_enthalpy_change=0.01, 
            slow_time_scale=3600 * 24 * 30):
        self.safety = safety
        self.relative_enthalpy_change = relative_enthalpy_change
        # internal time scale, should be >> process time scale; to avoid / zero
        self.slow_time_scale = slow_time_scale
        
    def calculate(self, ts):
        K = np.array(ts.variables['k'])
        # hack, only works in 1D and is insufficient for highly irregular grids
        K = 0.5 * (K[1::] + K[:-1:])  
        CFL = np.min(0.5 * (ts.mesh.cell_volumes) ** 2 / np.array((K / ts.variables['c'])))
        step = self.safety * CFL
        S_total = np.abs(
            ts.variables['S'] + ts.variables['S_headwall'] + ts.variables['S_inside'])
        denom = (np.abs(ts.variables['h']) / self.slow_time_scale + S_total)
        step_sources = (self.relative_enthalpy_change
                        * np.min(np.abs(np.array(ts.variables['h'] / denom))))
        if step_sources < step:
            step = step_sources
        return step


class OutputStep(object):
    
    def __init__(self):
        pass
    
    def next(self, ts):
        return None


class OutputStepHourly(OutputStep):
    
    def __init__(self):
        pass
    
    def next(self, ts):
        d0 = ts.date
        datenext = (datetime.datetime(d0.year, d0.month, d0.day, d0.hour)
                    + datetime.timedelta(seconds=3600))
        return ts._date_to_internal_time(datenext)


class Forcing(object):
    
    def __init__(self, values_inp, timeref=datetime.datetime(2012, 1, 1), variables=None):
        if variables is None:
            self.variables = [vj for vj in values_inp]
        else:
            self.variables = variables
        self._timeref = timeref
        self.variables = {vj:Variable(value=values_inp[vj]) for vj in self.variables}
        self.values = {vj: values_inp[vj] for vj in self.variables}
        
    def evaluate(self, t=None):
        return self.values
    
    def evaluateToVariable(self, t=None):
        for vj, ij in self.evaluate(t=t).iteritems():
            self.variables[vj].setValue(ij)


class ForcingInterpolation(Forcing):
    
    def __init__(self, values_inp, t_inp=None, variables=None, key_time='time'):
        if t_inp is None:
            t_inp_int = values_inp[key_time]
        else:
            t_inp_int = t_inp
        self.t_inp = t_inp_int
        self._timeref = t_inp_int[0]
        t_inp_rel = [tj - self._timeref for tj in self.t_inp]
        try:
            self.t_inp_rel = np.array([tj.total_seconds() for tj in t_inp_rel])
        except:
            self.t_inp_rel = np.array(t_inp_rel)
        if variables is None:
            self.variables = [vj for vj in values_inp if vj != key_time]
        else:
            self.variables = variables
        self.variables = {vj:Variable(value=values_inp[vj][0]) for vj in self.variables}
        self.values = {vj: values_inp[vj] for vj in self.variables}
        
    def evaluate(self, t=0):
        try:
            t_rel = (t - self._timeref).total_seconds()  # datetime object
        except:
            t_rel = t  # slump-internal time
        vals = {vj:np.interp(t_rel, self.t_inp_rel, self.values[vj])
                for vj in self.variables}
        return vals
