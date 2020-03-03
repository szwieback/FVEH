'''
Created on Mar 26, 2018

@author: zwieback
'''

from fipy import numerix

from constants import thermal_constants_default

class ThermalProperties(object):

    def __init__(self):
        pass

    def totalWaterContent(self, ts):
        return ts.variables['theta_t']

    def mineralContent(self, ts):
        return ts.variables['theta_m']

    def organicContent(self, ts):
        return ts.variables['theta_o']

    def nonmeltingIceContent(self, ts):
        return ts.variables['theta_n']

class ThermalPropertiesSoilPhaseChange(ThermalProperties):

    def __init__(
            self, parameters=thermal_constants_default, theta_t=0.0, theta_m=0.6,
            theta_o=0.0, theta_n=0.0):
        # theta_t: total water content (solid and liquid)
        self.T0 = parameters['T0']  # phase change temp
        self.Tref = parameters['Tref']  # reference T (h = 0)
        self.c_m = parameters['c_m']
        self.c_o = parameters['c_o']
        self.c_i = parameters['c_i']
        self.c_w = parameters['c_w']
        self.c_n = self.c_i  # non-melting ice; currently not to be combined with water/ice
        self.L_v = parameters['L_v']
        self.k_m = parameters['k_m']
        self.k_o = parameters['k_o']
        self.k_i = parameters['k_i']
        self.k_n = self.k_i
        self.k_w = parameters['k_w']
        self.k_a = parameters['k_a']
        self.theta_t = theta_t
        self.theta_m = theta_m
        self.theta_o = theta_o
        self.theta_n = theta_n

    def output(self):
        dict_out = {
            'T0': self.T0, 'Tref': self.Tref, 'c_m': self.c_m, 'c_o': self.c_o,
            'c_i': self.c_i, 'c_w': self.c_w, 'c_n': self.c_n, 'L_v': self.L_v,
            'k_m': self.k_m, 'k_o': self.k_o, 'k_a': self.k_a, 'k_i': self.k_i,
            'k_w': self.k_w, 'k_n': self.k_n, 'theta_t': self.theta_t,
            'theta_m': self.theta_m, 'theta_o': self.theta_o, 'theta_n': self.theta_n}
        return dict_out

    def initializeVariables(self, ts):
        ts.initializeDiagnostic('theta_t', self.totalWaterContent,
                                default=self.theta_t, output_variable=False)
        ts.initializeDiagnostic('T', self.temperatureFromEnthalpy,
                                default=self.Tref, output_variable=True)
        ts.initializeDiagnostic('theta_w', self.waterContent,
                                default=0.0 * self.theta_t, output_variable=False)
        ts.initializeDiagnostic('theta_i', self.iceContent,
                                default=self.theta_t, output_variable=False)
        ts.initializeDiagnostic('theta_m', self.mineralContent,
                                default=self.theta_m, output_variable=False)
        ts.initializeDiagnostic('theta_o', self.organicContent,
                                default=self.theta_o, output_variable=False)
        ts.initializeDiagnostic('theta_n', self.nonmeltingIceContent,
                                default=self.theta_n, output_variable=False)
        ts.initializeDiagnostic('k', self.thermalConductivity,
                                default=0.0, face_variable=True, output_variable=False)
        ts.initializeDiagnostic('c', self.heatCapacity, default=0.0, output_variable=False)

    def _cFrozen(self, ts):
        c = (self.c_m * ts.variables['theta_m'] +
             self.c_o * ts.variables['theta_o'] +
             self.c_i * ts.variables['theta_t'] +
             self.c_n * ts.variables['theta_n'])
        return c

    def _cThawed(self, ts):
        c = (self.c_m * ts.variables['theta_m'] +
             self.c_o * ts.variables['theta_o'] +
             self.c_w * ts.variables['theta_t'] +
             self.c_n * ts.variables['theta_n'])
        return c

    def heatCapacity(self, ts):
        # this is a lower bound; needed for time step calculations
        c = (self.c_m * ts.variables['theta_m'] +
             self.c_o * ts.variables['theta_o'] +
             self.c_i * ts.variables['theta_i'] +
             self.c_w * ts.variables['theta_w'] +
             self.c_n * ts.variables['theta_n']
             )
        return c

    def thermalConductivity(self, ts):
        # Cosenza
        theta_a = (1 - ts.variables['theta_m'] - ts.variables['theta_o']
                   -ts.variables['theta_t'] - ts.variables['theta_n'])
        k = ((self.k_m) ** 0.5 * ts.variables['theta_m'] +
             (self.k_o) ** 0.5 * ts.variables['theta_o'] +
             (self.k_i) ** 0.5 * ts.variables['theta_i'] +
             (self.k_w) ** 0.5 * ts.variables['theta_w'] +
             (self.k_n) ** 0.5 * ts.variables['theta_n'] +
             (self.k_a) ** 0.5 * theta_a
             ) ** 2
        return k.arithmeticFaceValue

    def enthalpyFromTemperature(self, ts, T=None, proportion_frozen=0.0):
        # only needed for convenient initial conditions
        # returns value instead of variable
        # proportion_frozen only needed if T == T0
        # very awkward for nonmelting ice; only meaningful for T<T0
        if T is None:
            T = ts.variables['T'].value
        h = 0 * T
        h[T < self.T0] = self._cFrozen(ts).value[T < self.T0] * (T[T < self.T0] - self.Tref)
        h[T == self.T0] = (
            self._cFrozen(ts).value[T == self.T0] * (self.T0 - self.Tref)
            +proportion_frozen * ts.variables['theta_t'].value[T == self.T0] * self.L_v)
        h[T > self.T0] = (
            self._cFrozen(ts).value[T > self.T0] * (self.T0 - self.Tref)
            +ts.variables['theta_t'].value[T > self.T0] * self.L_v
            +self._cThawed(ts).value[T > self.T0] * (T[T > self.T0] - self.T0))
        assert (numerix.sum((T > self.T0) * (ts.variables['theta_n'] > 0)) == 0)
        return h

    def _h0Frozen(self, ts):
        # h at phase change temp if frozen
        return self._cFrozen(ts) * (self.T0 - self.Tref)

    def _h0Thawed(self, ts):
        # h at phase change temp if all thawed
        # only applies to cells with 0 non-melting ice
        h_0_thawed = (
            self._cFrozen(ts) * (self.T0 - self.Tref) + ts.variables['theta_t'] * self.L_v)
        return h_0_thawed

    def temperatureFromEnthalpy(self, ts):
        # returns variable
        T = (self.Tref
            +(ts.variables['h'] < self._h0Frozen(ts))
              * (ts.variables['h']) / self._cFrozen(ts)
            +(ts.variables['theta_n'] == 0)
              * ((ts.variables['h'] >= self._h0Frozen(ts))
                  * (ts.variables['h'] <= self._h0Thawed(ts)) * (self.T0 - self.Tref)
                  +(ts.variables['h'] > self._h0Thawed(ts))
                    * ((self.T0 - self.Tref)
                       +(ts.variables['h'] - self._h0Thawed(ts)) / self._cThawed(ts)))
            +(ts.variables['theta_n'] > 0)
              * (ts.variables['h'] >= self._h0Frozen(ts)) * (self.T0 - self.Tref)
            )
        return T

    def iceContent(self, ts):
        # does not include non-melting ice
        # assumption: water-containing cells do not contain any non-melting ice
        theta_i = ((ts.variables['h'] < self._h0Frozen(ts)) * ts.variables['theta_t']
                   +(ts.variables['h'] >= self._h0Frozen(ts))
                     * (ts.variables['h'] <= self._h0Thawed(ts))
                     * ts.variables['theta_t'] * ((self._h0Thawed(ts) - ts.variables['h']))
                     / (self._h0Thawed(ts) - self._h0Frozen(ts)
                        +1.0 * (ts.variables['theta_t'] <= 0.0)))  # hack to avoid 0/0
        return theta_i

    def waterContent(self, ts):
        theta_w = ((ts.variables['h'] > self._h0Thawed(ts)) * ts.variables['theta_t']
                   +(ts.variables['h'] >= self._h0Frozen(ts))
                     * (ts.variables['h'] <= self._h0Thawed(ts))
                     * ts.variables['theta_t'] * (ts.variables['h'] - self._h0Frozen(ts))
                     / (self._h0Thawed(ts) - self._h0Frozen(ts)
                        +1.0 * (ts.variables['theta_t'] <= 0.0)))
        return theta_w

class ThermalPropertiesSoilPhaseChangeLinearCharacteristic(
        ThermalPropertiesSoilPhaseChange):

    def __init__(
            self, parameters=thermal_constants_default, theta_t=0.0, theta_m=0.6,
            theta_o=0.0, theta_n=0.0):
        # theta_t: total water content (solid and liquid)
        # theta_n: assumed 0
        self.T0 = parameters['T0']  # phase change temp
        self.Tref = parameters['Tref']  # reference T (h = 0)
        self.Tt = parameters['Tt']
        self.c_m = parameters['c_m']
        self.c_o = parameters['c_o']
        self.c_i = parameters['c_i']
        self.c_w = parameters['c_w']
        self.c_n = self.c_i  # non-melting ice; currently not to be combined with water/ice
        self.L_v = parameters['L_v']
        self.k_m = parameters['k_m']
        self.k_o = parameters['k_o']
        self.k_i = parameters['k_i']
        self.k_n = self.k_i
        self.k_w = parameters['k_w']
        self.k_a = parameters['k_a']
        self.theta_t = theta_t
        self.theta_m = theta_m
        self.theta_o = theta_o
        self.theta_n = theta_n

    def output(self, constituents_only=False, parameters_only=False):
        dict_parameters = {
            'T0': self.T0, 'Tref': self.Tref, 'Tt': self.Tt, 'c_m': self.c_m,
            'c_o': self.c_o, 'c_i': self.c_i, 'c_w': self.c_w, 'c_n': self.c_n,
            'L_v': self.L_v, 'k_m': self.k_m, 'k_o': self.k_o, 'k_a': self.k_a,
            'k_i': self.k_i, 'k_w': self.k_w, 'k_n': self.k_n}
        dict_constituents = {'theta_t': self.theta_t, 'theta_m': self.theta_m,
                             'theta_o': self.theta_o, 'theta_n': self.theta_n}
        if constituents_only:
            return dict_constituents
        elif parameters_only:
            return dict_parameters
        else:
            dict_parameters.update(dict_constituents)
            return dict_parameters

    def initializeVariables(self, ts):
        ts.initializeDiagnostic('theta_t', self.totalWaterContent,
                                default=self.theta_t, output_variable=False)
        ts.initializeDiagnostic('T', self.temperatureFromEnthalpy,
                                default=self.Tref, output_variable=True)
        ts.initializeDiagnostic('theta_w', self.waterContent,
                                default=0.0 * self.theta_t, output_variable=False)
        ts.initializeDiagnostic('theta_i', self.iceContent,
                                default=self.theta_t, output_variable=False)
        ts.initializeDiagnostic('theta_m', self.mineralContent,
                                default=self.theta_m, output_variable=False)
        ts.initializeDiagnostic('theta_o', self.organicContent,
                                default=self.theta_o, output_variable=False)
        ts.initializeDiagnostic('theta_n', self.nonmeltingIceContent,
                                default=self.theta_n, output_variable=False)
        ts.initializeDiagnostic('k', self.thermalConductivity,
                                default=0.0, face_variable=True, output_variable=False)
        ts.initializeDiagnostic('c', self.heatCapacity, default=0.0, output_variable=False)

    def enthalpyFromTemperature(self, ts, T=None):
        # only needed for convenient initial conditions
        # returns value instead of variable

        if T is None:
            T = ts.variables['T'].value
        h = 0 * T
        h[T <= self.Tt] = (self._cFrozen(ts).value[T <= self.Tt]
                           * (T[T <= self.Tt] - self.Tref))
        h[T > self.Tt] = (self._cFrozen(ts).value[T > self.Tt]
                          * (T[T > self.Tt] - self.Tref)
                          +ts.variables['theta_t'].value[T > self.Tt] * self.L_v
                           * (T[T > self.Tt] - self.Tt) / (self.T0 - self.Tt))
        # overwrite results where > T0
        h[T > self.T0] = (
            self._cFrozen(ts).value[T > self.T0] * (self.T0 - self.Tref)
            +ts.variables['theta_t'].value[T > self.T0] * self.L_v
            +self._cThawed(ts).value[T > self.T0] * (T[T > self.T0] - self.T0))
        # non-melting ice
        assert (numerix.sum((T > self.T0) * (ts.variables['theta_n'] > 0)) == 0)
        return h

        # return h
    def _h0Frozen(self, ts):
        # h at Tt if frozen
        return self._cFrozen(ts) * (self.Tt - self.Tref)

    def _h0Thawed(self, ts):
        # h at phase change temp
        # only applies to cells with 0 non-melting ice
        # (otherwise: minimum enthalpy at freezing point)
        # note that subfreezing water has same heat capacity as ice (for now)
        h_0_thawed = (self._cFrozen(ts) * (self.T0 - self.Tref) +
                      ts.variables['theta_t'] * self.L_v
                      )
        return h_0_thawed

    def temperatureFromEnthalpy(self, ts):
        # returns variable
        T = (self.Tref +
            (ts.variables['theta_n'] == 0)
             * ((ts.variables['h'] <= self._h0Frozen(ts))
                 * (ts.variables['h']) / self._cFrozen(ts)
                    +(ts.variables['h'] > self._h0Thawed(ts))
                    * ((self.T0 - self.Tref)
                        +(ts.variables['h'] - self._h0Thawed(ts)) / self._cThawed(ts))
                    +(ts.variables['h'] > self._h0Frozen(ts))
                    * (ts.variables['h'] <= self._h0Thawed(ts))
                    * ((self.Tt - self.Tref) + (self.T0 - self.Tt)
                       * (ts.variables['h'] - self._h0Frozen(ts))
                       / (self._h0Thawed(ts) - self._h0Frozen(ts))))
            +(ts.variables['theta_n'] > 0)
              * ((ts.variables['h'] >= self._h0Thawed(ts)) * (self.T0 - self.Tref)
                 +(ts.variables['h'] < self._h0Thawed(ts))
                   * (ts.variables['h'] / self._cFrozen(ts))))
        return T

    def iceContent(self, ts):
        # does not handle non-melting ice
        theta_i = ts.variables['theta_t'] - self.waterContent(ts)
        return theta_i

    def waterContent(self, ts):
        theta_w = ((ts.variables['h'] > self._h0Thawed(ts)) * ts.variables['theta_t']
                   +(ts.variables['h'] <= self._h0Thawed(ts))
                    * (ts.variables['h'] > self._h0Frozen(ts))
                    * (ts.variables['h'] - self._h0Frozen(ts))
                    / (self._h0Thawed(ts) - self._h0Frozen(ts)) * ts.variables['theta_t'])
        return theta_w
