from psana import *
import numpy as np
import scipy.io
from xtcav.ShotToShotCharacterization import *
import os
from mpi4py import MPI
import argparse

# Analysis modules
sys.path.append(
    '/reg/neh/home4/tbarillo/amolr2516/LR25_Analysis/PythonRoutines/')

import ITOFDataPreProcessing
import UXSDataPreProcessing
import SHESPreProcessing
import XTCAV_Processing

# Parallel processing


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###############################################################################


class XTCExporter(object):
    def __init__(self, args):
        self.args = args

###############################################################################

    def run(self):

        self.filenum = 0
        self.DetInit()
        self.endrun = False
        self.loop_idx = 0
        self.ArrInit(1024)
        for nevent, evt in enumerate(self.ds.events()):
            if nevent == self.args.noe:
                break
            if nevent % (size) != rank:
                continue  # different ranks look at different events
            self.loop_idx += 1

            self.getevtdata(evt)

            # print self.nsave
            if hasattr(self, 'nsave') and self.nsave == self.args.nsave:
                self.save()

        self.save()

        print 'Client', rank, 'done'

###############################################################################

    def DetInit(self):
        self.ds = DataSource(self.args.exprun +
                             ':smd:dir=/reg/d/psdm/amo/amolr2516/xtc:live')

        self.SHES = SHESPreProcessing.SHESPreProcessor()
        self.UXS = Detector(self.args.UXS)
        self.UXS_Pre = UXSDataPreProcessing.UXSDataPreProcessing()
        self.ITOF = Detector(self.args.ITOF)
        self.XTCAV = XTCAV_Processing.XTCavProcessor()
        self.XTCAV.set_data_source(self.ds)
        self.EBeam = Detector(self.args.EBeam)
        self.GMD = Detector(self.args.GMD)
        self.ENV = self.ds.env().configStore()
        self.PRESS = Detector('AMO:LMP:VG:21:PRESS')
        self.CHIC = Detector('SIOC:SYS0:ML01:AO901')

###############################################################################

    def ArrInit(self, uxslength):
        # SHES
        # Scienta Hemispherical array 2D: Energy projection; index of event
        shlength = len(self.SHES.calib_array)
        self.shEnergy = np.zeros(shlength)
        self.shProjArr = np.zeros((self.args.nsave, shlength))
        self.ehitsX = []
        self.ehitsY = []
        self.ehitsTS = []

        # UXS
        # XRay spectro array 2D principal components:
        # Ampl1,pos1,FWHM1,Ampl2,pos2,FWHM2
        self.uxsPCArr = np.zeros((self.args.nsave, 6))
        # Xray spectro array 2D: Energy projection
        self.uxsProjArr = np.zeros((self.args.nsave, uxslength))
        self.uxsProjArr2 = np.zeros((self.args.nsave, uxslength))

        # ITOF
        # micro tof: ion yield
        self.itofArr = np.zeros((self.args.nsave, 40000))

        # GMD
        # Gas detector array (6 ebeam EL3 values)
        self.gmdArr = np.zeros((self.args.nsave, 6))

        # ENV
        # Acq parameters:
        # offset and fullscale for both channels low gain: 2, high gain 1
        self.envArr = np.zeros((1, 5))

        # EBeam
        self.ebeamArr = np.zeros((self.args.nsave, 21))

        # EPICS
        # Sample pressure array
        self.sPressArr = np.zeros((self.args.nsave))

        # XTCAV
        # "principal components" - centre, width and sum of pulses
        self.xtcavPCArr = np.zeros((self.args.nsave, 5))

        # Time stamp array
        self.TimeSt = np.zeros((self.args.nsave,))

        # Chicane delay
        self.chicane_fs = np.nan

        # initialize arrays to nan
        self.shProjArr[:, :] = np.nan
        self.uxsProjArr[:, :] = np.nan
        self.uxsProjArr2[:, :] = np.nan
        self.uxsPCArr[:, :] = np.nan
        self.itofArr[:, :] = np.nan

        self.gmdArr[:, :] = np.nan
        self.ebeamArr[:, :] = np.nan
        self.envArr[:, :] = np.nan
        self.TimeSt[:] = np.nan
        self.sPressArr[:] = np.nan
        self.xtcavPCArr[:, :] = np.nan
        self.nsave = 0


###############################################################################

    def getevtdata(self, evt):
        # Initialize the arrays
        if not hasattr(self, 'shEnergy'):
            shEnergy = len(self.SHES.calib_array)
            if shEnergy is None:
                print 'shEnergy is None. Returning'
                return
        self.shEnergy = self.SHES.calib_array

        # Get the environnement data
        envdata = self.ENV.get(psana.Acqiris.Config)
        if envdata is not None and self.nsave == 0:
            self.envArr[self.nsave, 0] = envdata.vert()[0].offset()
            self.envArr[self.nsave, 1] = envdata.vert()[0].fullScale()
            self.envArr[self.nsave, 2] = envdata.vert()[2].offset()
            self.envArr[self.nsave, 3] = envdata.vert()[2].fullScale()
            self.envArr[self.nsave, 4] = envdata.horiz().sampInterval()
        if self.nsave == 0:
            self.chicane_fs = self.CHIC(evt)

        # Get the event ID and timestamp
        evtid = evt.get(EventId)
        evttime = evtid.idxtime()
        try:
            self.TimeSt[self.nsave] = evttime.time()
        except Exception as e:
            print 'getting event time failed with exception {0}'.format(e)
            return

        # Get the hemisperical analyser signal data
        lx, ly, proj, proj_raw = self.SHES.PreProcess(evt)
        self.ehitsX += lx  # Xpos
        self.ehitsY += ly  # Ypos
        # electron hits timestamps
        self.ehitsTS += list(np.ones((len(lx))) * evttime.time())
        self.shProjArr[self.nsave, :] = proj
        if any((any(np.isnan(lx)),
                any(np.isnan(ly)),
                any(np.isnan(proj)),
                any(np.isnan(proj_raw)))):
            print 'Pre-processing of SHES data failed'

        # Get the Xray spectrometer analyser data
        XrayImg = self.UXS.raw(evt)
        if XrayImg is not None:
            PCs, proj, proj2 = self.UXS_Pre.StandardAnalysis(XrayImg)
            self.uxsPCArr[self.nsave, :] = PCs
            self.uxsProjArr[self.nsave, :] = proj
            self.uxsProjArr2[self.nsave, :] = proj2
        else:
            print 'No UXS data ({0} events saved)'.format(self.nsave)

        # Get the ITOF data
        waveforms = self.ITOF.waveform(evt)
        if waveforms is not None:
            waveform = waveforms[1]
            tmp = ITOFDataPreProcessing.ITOFDataPreProcessing(waveform)
            tmp.StandardAnalysis()
            self.itofArr[self.nsave, :] = tmp.wf
        else:
            print 'No ITOF data ({0} events saved)'.format(self.nsave)

        # Get the XTCAV data
        # self.XTCAV.set_event(evt)
        # ok = self.XTCAV.process()
        ok = False
        if ok:
            self.xtcavPCArr[self.nsave, :] = [
                self.XTCAV.results['delay'],
                self.XTCAV.results['moment_fwhms'][0],  # pump
                self.XTCAV.results['pulse_sums'][0],  # probe
                self.XTCAV.results['moment_fwhms'][1],  # pump
                self.XTCAV.results['pulse_sums'][1]  # probe
            ]

        gmddata = self.GMD.get(evt)
        samplepressuredata = self.PRESS(evt)

        # Get the Ebeam data
        EBeamdata = self.EBeam.get(evt)
        if EBeamdata is not None:
            # record ebeam parameters
            ebeamlistpar = ['damageMask', 'ebeamCharge', 'ebeamDumpCharge', 'ebeamEnergyBC1',
                            'ebeamEnergyBC2', 'ebeamL3Energy', 'ebeamLTU250', 'ebeamLTU450',
                            'ebeamLTUAngX', 'ebeamLTUAngY', 'ebeamLTUPosX', 'ebeamLTUPosY',
                            'ebeamPhotonEnergy', 'ebeamPkCurrBC1', 'ebeamPkCurrBC2', 'ebeamUndAngX',
                            'ebeamUndAngY', 'ebeamUndPosX', 'ebeamUndPosY', 'ebeamXTCAVAmpl', 'ebeamXTCAVPhase']

            for i, par in enumerate(ebeamlistpar):
                self.ebeamArr[self.nsave, i] = getattr(EBeamdata, par)()
        else:
            print 'No EBeam data ({0} events saved)'.format(self.nsave)

        if gmddata is not None:
            # upstream
            self.gmdArr[self.nsave, 0] = gmddata.f_11_ENRC()
            self.gmdArr[self.nsave, 1] = gmddata.f_12_ENRC()
            # downstream
            self.gmdArr[self.nsave, 2] = gmddata.f_21_ENRC()
            self.gmdArr[self.nsave, 3] = gmddata.f_22_ENRC()
            # set values
            self.gmdArr[self.nsave, 4] = gmddata.f_63_ENRC()
            self.gmdArr[self.nsave, 5] = gmddata.f_64_ENRC()

            gmd = self.gmdArr[self.nsave, self.args.prefgmd]
        else:
            print 'No Gas Detector data ({0} events saved)'.format(self.nsave)

        if samplepressuredata is not None:
            self.sPressArr[self.nsave] = samplepressuredata
        else:
            print 'No pressure data ({0} events saved)'.format(self.nsave)

        if self.no_useful_data(self.nsave):
            print 'No errors but no useful data at shot ', self.loop_idx
        else:
            self.nsave += 1
        if self.nsave % 100 == 0:
            print 'Loop iteration {0} in rank {1}, {2} events saved'.format(
                self.loop_idx, rank, self.nsave)
###############################################################################

    def no_useful_data(self, nsave, verbose=False):
        """Checks whether for the shot saved at nsave, there is any data at all
        Does not check the timestamp since that's never NaN
        Does not check the pressure since it is not important

        Args:
            nsave (int): Index into the saved data array

        Returns:
            Bool: True if there is no useful data at all, False if there is
                  some useful data in the shot
        """
        if not np.isnan(self.ebeamArr[nsave, :]).all():
            if verbose:
                print 'Found Ebeam data'
            return False
        if not np.isnan(self.shProjArr[nsave, :]).all():
            if verbose:
                print 'Found SHES proj data'
            return False
        if not np.isnan(self.uxsPCArr[nsave, :]).all():
            if verbose:
                print 'Found UXS PCs data'
            return False
        if not np.isnan(self.uxsProjArr[nsave, :]).all():
            if verbose:
                print 'Found UXS proj'
            return False
        if not np.isnan(self.uxsProjArr2[nsave, :]).all():
            if verbose:
                print 'Found UXS proj2'
            return False
        if not np.isnan(self.itofArr[nsave, :]).all():
            if verbose:
                print 'Found ITOF data'
            return False
        if not np.isnan(self.gmdArr[nsave, :]).all():
            if verbose:
                print 'Found gas detector data'
            return False
        return True

###############################################################################

    def save(self):
        if self.args.save is True:
            runnumber = int(self.args.exprun[18:])
            filename = 'amolr2516_r' + str(runnumber).zfill(4) + '_' + \
                str(rank).zfill(3) + '_' + str(self.filenum).zfill(3)

            directory_n = ('/reg/d/psdm/AMO/amolr2516/results/npzfiles/'
                           + 'run'
                           + str(runnumber).zfill(4)
                           + '/')
            directory_m = ('/reg/d/psdm/AMO/amolr2516/results/matfiles/'
                           + 'run'
                           + str(runnumber).zfill(4)
                           + '/')
            if not os.path.exists(directory_n):
                os.makedirs(directory_n)
            if not os.path.exists(directory_m):
                os.makedirs(directory_m)
            directory_n = ('/reg/d/psdm/AMO/amolr2516/results/npzfiles/'
                           + 'run'
                           + str(runnumber).zfill(4)
                           + '/')
            directory_m = ('/reg/d/psdm/AMO/amolr2516/results/matfiles/'
                           + 'run'
                           + str(runnumber).zfill(4)
                           + '/')
            if not os.path.exists(directory_n):
                os.makedirs(directory_n)
            if not os.path.exists(directory_m):
                os.makedirs(directory_m)

            if rank == 1:
                print 'rank 1 writing file...'

        # Save electrons hits in one array:
        SHESHits = np.zeros((len(self.ehitsX), 3))
        SHESHits[:, 0] = self.ehitsX
        SHESHits[:, 1] = self.ehitsY
        SHESHits[:, 2] = self.ehitsTS

        data = {'EBeamParameters': self.ebeamArr[0:self.nsave, :],
                'SHESHits': SHESHits.astype(np.float64),
                'SHESwf': self.shProjArr[0:self.nsave, :].astype(np.float32),
                'UXSpc': self.uxsPCArr[0:self.nsave, :].astype(np.float32),
                'UXSwf': self.uxsProjArr[0:self.nsave, :].astype(np.float32),
                'UXSwf2': self.uxsProjArr2[0:self.nsave, :].astype(np.float32),
                'ITOF': self.itofArr[0:self.nsave, :].astype(np.float32),
                'XTCAV': self.xtcavPCArr[0:self.nsave, :],
                'Pressure': self.sPressArr[0:self.nsave],
                'GasDetector': self.gmdArr[0:self.nsave, :],
                'EnvVar': self.envArr,
                'chicane_fs': self.chicane_fs,
                'TimeStamp': self.TimeSt[0:self.nsave]}
        scipy.io.savemat(directory_m + filename + '.mat', data)

        np.savez(directory_n + filename, **data)
        if rank == 1:
            print 'rank1 done with writing file.'
        self.filenum += 1
        self.ArrInit(1024)

###############################################################################


# def runclient(args):
#     client = XTCExporter(args)
#     client.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exprun",
        help="psana experiment/run string (e.g. exp=amoj1516:run=43)")
    parser.add_argument(
        "-n",
        "--noe",
        help="number of events, all events=0",
        default=-1,
        type=int)

    args = parser.parse_args()
    args.SHES = 'OPAL3'
    args.UXS = 'OPAL1'
    args.ITOF = 'ACQ1'  # 'Acq01'#'ACQ1'
    args.GMD = 'FEEGasDetEnergy'
    args.EBeam = 'EBeam'
    args.PRESS = 'AMO:LMP:VG:21:PRESS'
    args.XTCAV = 'xtcav'

    # GMD intensity threshold for shots to be considered for online plots:
    # args.gmdThr = 0.1
    # Prefered gmd value for online plots:
    args.prefgmd = 0

    # Prefered  channel for online plots and raw traces of itof:
    # args.prefch = 1

    # Save files:
    args.save = True
    # Number of shots saved per file
    args.nsave = 5000

    exporter = XTCExporter(args)
    exporter.run()
