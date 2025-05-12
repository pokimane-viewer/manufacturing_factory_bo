# cad_optimization.py

import math

class CADModel:
    def __init__(self, geometry):
        self.geometry = geometry
    def compute_volume(self):
        return math.prod(self.geometry)
    def surface_area(self):
        return 2*(self.geometry[0]*self.geometry[1] + self.geometry[1]*self.geometry[2] + self.geometry[2]*self.geometry[0])

def optimize_cad_model(cad: CADModel):
    best = cad.compute_volume() / cad.surface_area()
    scale_factor = 1.01
    new_geo = tuple(g*scale_factor for g in cad.geometry)
    improved = CADModel(new_geo)
    improved_score = improved.compute_volume() / improved.surface_area()
    if improved_score > best:
        return improved
    return cad
