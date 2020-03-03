'''
Created on 19 Nov 2017

@author: Simon
'''
from fipy import numerix
from scipy.constants import Stefan_Boltzmann
from collections import namedtuple

Fluxes = namedtuple('Fluxes', 'radiation latent sensible total')

von_Karman = 0.41
cp_air = 1.01e3  # use constant for now
rho_air = 1.25  # use constant for now
Le = 2.5e6  # use constant for now

class BoundaryCondition(object):

    def __init__(self):
        pass

class BoundaryConditionCollection(object):

    def __init__(self):
        pass

class BoundaryCondition1D(BoundaryCondition):

    def __init__(self, side):
        self.side = side
        self.meshfaces = None
        self.volume = None
        self.dxf = None

    def invoke(self, ts):
        ts._initializeSourcesZero(source_name=self._name_source())

    def updateGeometry(self, ts):
        self.meshfaces = self._face(ts)
        self.volume = self._volume(ts)

    def output(self):
        return {}

    def _volume(self, ts):
        if self.side == 'inside':
            volume = ts.mesh.cell_left_volume
        elif self.side == 'headwall':
            volume = ts.mesh.cell_right_volume
        else:
            raise NotImplementedError
        return volume

    def indexCell(self, ts):
        if self.side == 'inside':
            index = 0
        elif self.side == 'headwall':
            index = len(ts.mesh.mesh.cellCenters.value[0]) - 1
        else:
            raise NotImplementedError
        return index

    def indexFace(self, ts):
        if self.side == 'inside':
            index = 0
        elif self.side == 'headwall':
            index = len(ts.mesh.mesh.cellCenters.value[0])
        else:
            raise NotImplementedError
        return index

    def _face(self, ts):
        if self.side == 'inside':
            face = ts.mesh.mesh.facesLeft
        elif self.side == 'headwall':
            face = ts.mesh.mesh.facesRight
        else:
            raise NotImplementedError
        return face

    def _name_source(self):
        if self.side == 'inside':
            source = 'S_inside'
        elif self.side == 'headwall':
            source = 'S_headwall'
        else:
            raise NotImplementedError
        return source

class BoundaryCondition1DConstantTemperature(BoundaryCondition1D):
    # elements of bc_data can be numbers (e.g. numpy arrays) or FiPy variables
    # not functional at the moment

    def __init__(self, name_T='Tair', side='inside'):
        BoundaryCondition1D.__init__(self, side)
        self.name_T = name_T

    def invoke(self, ts):
        BoundaryCondition1D.invoke(self, ts)
        self.T = ts.variables[self.name_T]
        ts.variables['T'].constrain(self.T, where=self.meshfaces)
        self.flux = (ts.variables['T'].faceGrad()[0][self.indexFace(ts)] *
                     ts.variables['k'][self.indexFace(ts)])
    def update(self):
        pass

    def output(self):
        return {'Tsurf': self.T.value, 'flux': self.flux.value}

class BoundaryCondition1DNoFlux(BoundaryCondition1D):

    def __init__(self, side='inside'):
        BoundaryCondition1D.__init__(self, side)

    def invoke(self, ts):
        BoundaryCondition1D.invoke(self, ts)
        ts.variables['T'].faceGrad.constrain(0, where=self.meshfaces)

    def update(self):
        pass

    def output(self):
        return {'flux': 0.0}

class BoundaryCondition1DNewtonLawCooling(BoundaryCondition1D):

    def __init__(self, coeff_newton=1.0, name_Tair='Tair', side='headwall'):
        # coeff_newton: per unit area!
        # vertical headwall assumed!
        assert side == 'headwall'
        BoundaryCondition1D.__init__(self, side)
        self.name_source = self._name_source()
        self.name_Tair = name_Tair
        self.coeff_newton = coeff_newton

    def update(self, coeff_newton=None):
        if coeff_newton is not None:
            self.coeff_newton = coeff_newton

    def invoke(self, ts):
        # set flux to zero and add appropriate source
        self.Tair = ts.variables[self.name_Tair]
        ts.variables['T'].faceGrad.constrain(0, where=self.meshfaces)
        Af = ts.mesh.mesh._faceAreas[self.meshfaces.value][0]
        self.Tsurf = ts.variables['T_right'][self.indexCell(ts)]
        # compute mask scaled by 1/volume --> volumetric density
        masked_volume = (self.meshfaces * ts.mesh.mesh.faceNormals).divergence
        flux = (self.Tair - self.Tsurf) * self.coeff_newton
        source_boundary = masked_volume * (Af) * flux
        self.flux = flux
        ts.variables[self.name_source] = source_boundary

    def output(self):
        return {'flux': self.flux.value, 'Tair': self.Tair.value,
                'Tsurf': self.Tsurf.value}

class BoundaryCondition1DEnergyBalance(BoundaryCondition1D):
    def __init__(self, bc_parameters, celsius=True, meas_height=10.0,
                 name_Tair='Tair', name_Sin='Sin', name_Lin='Lin',
                 name_VPair='VPair', name_wind='wind', name_pressure='p',
                 side='headwall'):

        assert side == 'headwall'
        if not celsius:
            raise NotImplementedError
        BoundaryCondition1D.__init__(self, side)
        self.bc_parameters = bc_parameters
        self.meas_height = meas_height
        self.name_source = self._name_source()
        self.name_Tair = name_Tair
        self.name_Sin = name_Sin
        self.name_Lin = name_Lin
        self.name_VPair = name_VPair
        self.name_wind = name_wind
        self.name_pressure = name_pressure

    def update(self, bc_parameters=None):
        if bc_parameters is not None:
            raise NotImplementedError

    def fluxes(self, ts):
        # wind-independent roughness as in doi:10.3189/2014JoG13J045
        Tsurf = ts.variables['T_right'][self.indexCell(ts)]
        # Tsurf = ts.variables['T'][self.indexCell]
        # sat; T in Celsius; in Pa
        VPsurf = 100 * 6.1094 * numerix.exp((17.625 * Tsurf) / (Tsurf + 243.04))
        # T in Celsius
        flux_radiation = (
            self.Sin * (1 - self.bc_parameters['albedo'])
            + self.bc_parameters['emissivity'] * self.Lin
            - self.bc_parameters['emissivity'] * Stefan_Boltzmann * (Tsurf + 273.15) ** 4)  
        C = von_Karman ** 2 / (
            numerix.log(
                (self.meas_height - self.bc_parameters['d0']) / self.bc_parameters['z0m'])
            * numerix.log(
                (self.meas_height - self.bc_parameters['d0']) / self.bc_parameters['z0a']))
        flux_latent = (self.bc_parameters['beta'] * C * rho_air * Le * 0.622 *
                       self.wind * (self.VPair - VPsurf) / self.pressure)
        flux_sensible = C * rho_air * cp_air * self.wind * (self.Tair - Tsurf)
        return Fluxes(radiation=flux_radiation, latent=flux_latent,
                      sensible=flux_sensible,
                      total=flux_radiation + flux_latent + flux_sensible)

    def invoke(self, ts):
        # set flux to zero and add appropriate source
        self.Tair = ts.variables[self.name_Tair]
        self.Sin = ts.variables[self.name_Sin]
        self.Lin = ts.variables[self.name_Lin]
        self.VPair = ts.variables[self.name_VPair]
        self.wind = ts.variables[self.name_wind]
        self.pressure = ts.variables[self.name_pressure]
        ts.variables['T'].faceGrad.constrain(0, where=self.meshfaces)
        Af = ts.mesh.mesh._faceAreas[self.meshfaces.value][0]
        # compute mask scaled by 1/volume --> volumetric density
        masked_volume = (self.meshfaces * ts.mesh.mesh.faceNormals).divergence
        # flux in W/m2, always into the slump
        fluxes = self.fluxes(ts)
        source_boundary = masked_volume * (Af) * (fluxes.total)
        ts.variables[self.name_source] = source_boundary
        self.flux_latent = fluxes.latent
        self.flux_sensible = fluxes.sensible
        self.flux_radiation = fluxes.radiation
        self.flux_total = fluxes.total
        self.Tsurf = ts.variables['T_right'][self.indexCell(ts)]
        
    def output(self):
        return {'flux': self.flux_total.value, 'flux_latent': self.flux_latent.value,
                'flux_sensible': self.flux_sensible.value, 'Tsurf': self.Tsurf.value,
                'flux_radiation':self.flux_radiation.value}


class BoundaryConditionCollection1D(object):
    
    def __init__(self, bc_headwall=None, bc_inside=BoundaryCondition1DNoFlux()):
        assert bc_headwall is not None
        self.headwall = bc_headwall
        self.inside = bc_inside
        
    def updateGeometry(self, ts):
        self.headwall.updateGeometry(ts)
        self.inside.updateGeometry(ts)
        
    def invoke(self, ts):
        self.headwall.invoke(ts)
        self.inside.invoke(ts)
        
    def update(self):
        # useless (at the moment)
        self.headwall.update()
        self.inside.update()
        
    def output(self):
        outp = {}
        outp.update(
            {key + '_headwall': value for key, value in self.headwall.output().items()})
        outp.update(
            {key + '_inside': value for key, value in self.inside.output().items()})
        return outp
