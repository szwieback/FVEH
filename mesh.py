'''
Created on 19 Nov 2017

@author: Simon
'''
from fipy import Grid1D
import numpy as np

class OneDMesh(object):

    def __init__(self):
        pass

    @property
    def cell_volumes(self):
        return self.mesh.cellVolumes

    @property
    def cell_mid_points(self):
        return self.mesh.cellCenters.value[0]

    @property
    def face_right_position(self):
        return np.array(self.mesh.faceCenters[0, self.mesh.facesRight()])[0]

    @property
    def face_left_position(self):
        return np.array(self.mesh.faceCenters[0, self.mesh.facesLeft()])[0]

    @property
    def cell_left_volume(self):
        return self.mesh.cellVolumes[0]

    @property
    def cell_right_volume(self):
        return self.mesh.cellVolumes[-1]

class OneDMeshRegular(OneDMesh):

    def __init__(self, nx, dx):
        assert np.array(dx).shape == ()
        self.mesh = Grid1D(nx=nx, dx=dx)

class OneDMeshIrregular(OneDMesh):
    def __init__(self, dx):
        self.mesh = Grid1D(dx=dx)
