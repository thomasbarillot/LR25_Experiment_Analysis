"""Processing for the L3 E-Beam parameters"""
from psana import *

ebeam_det_name = 'EBeam' #TODO check this
gammaConvFact=13720. #TODO check this (from Alberto or Ago for eV calculation)
peConvFact=8330. #TODO check this (from Alberto or Ago for eV calculation)
#TODO funcitonality for returning None

class L3EnergyProcessor(object):
    
    def __init__(self):
        #The following all hard-coded in
        self.gammaConvFact=float(gammaConvFact)
        self.peConvFact=float(peConvFact)
        self.ebeam_det=Detector(ebeam_det_name)
    
    def CentPE(self, event): #'central' photon energy - harder to define 
                             # for two-colour expt
        ebeamparams=self.ebeam_det.get(event)
        if ebeamparams is None:
            return None
        energyL3=ebeamparams.ebeamL3Energy()
        gamma = energyL3/self.gammaConvFact

        return self.peConvFact * gamma**2
