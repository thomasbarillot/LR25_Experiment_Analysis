import re
import sys
import pickle
import numpy as np
from mpi4py import MPI
from scipy import optimize
import psana
from xtcav import ShotToShotCharacterization

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def gaussian(x, mean, fwhm, height):
    w =fwhm/(2*np.sqrt(np.log(2)))
    return height*np.exp(- ((x-mean)/w)**2)


def two_gauss(x, mean_1, fwhm_1, height_1, mean_2, fwhm_2, height_2):     
    return gaussian(x, mean_1, fwhm_1, height_1), gaussian(x, mean_2, fwhm_2, height_2) 


def double_gauss(x, mean_1, fwhm_1, height_1, mean_2, fwhm_2, height_2):     
    tgauss = two_gauss(x, mean_1, fwhm_1, height_1, mean_2, fwhm_2, height_2)     
    return tgauss[0]+tgauss[1]


def moment(x, y, axis=None):
    norm = y.sum(axis)
    return (x * y).sum(axis)/norm


def var(x, y, axis=None):
    mu = moment(x,y,axis)
    return (y/y.sum(axis) * (x-mu)**2).sum(axis)


def fwhm_from_var(x, y, axis=None):
    return 2*np.sqrt(2*np.log(2))*np.sqrt(var(x, y, axis))


class XTCavProcessor:
    def __init__(self):
        self.s2s = ShotToShotCharacterization.ShotToShotCharacterization()
        self.ds = None
        self.event_ok = False
        self.results = None

    def set_data_source(self, ds):
        self.ds = ds
        self.s2s.SetEnv(ds.env())
        self.results = None

    def set_run(self, run):
        jobname = self.ds.env().jobName()
        new_jobname = re.sub('run=[0-9]{2,4}', 'run={0}'.format(run), jobname)
        ds = psana.DataSource(new_jobname)
        self.set_data_source(ds)
        self.results = None
        
    def set_event(self, evt):
        self.event_ok = self.s2s.SetCurrentEvent(evt)
        self.results = None
        return self.event_ok

    def process(self, force_split=True, agr_thresh=0.5, verbose=False):
        if not self.event_ok:
            return False
        t, power, ok = self.s2s.XRayPower()
        if not ok:
            if verbose:
                print('X-Ray power retrieval unsuccessful')
            return False
        agr, ok = self.s2s.ReconstructionAgreement()
        if not (ok and agr>agr_thresh):
            if verbose:
                print('Reconstruction agreement insufficient')
            return False
        
        t = t.squeeze()
        power = power.squeeze()
        t_range = t.max() - t.min()
        
        #Find where the trace actually has support
        thresh = power.max()*0.05
        power_nonz = (power*(power > thresh)).nonzero()
        t_lims = (t[power_nonz[0][0]], t[power_nonz[0][-1]])
        trace_centre = np.mean(t_lims)
        if verbose:
            print('Trace extends from {0} fs to {1} fs, centre at {2:.1f} fs'.format(
                t_lims[0], t_lims[1], trace_centre))

        ## Moments method
        (t_left, t_right) = (t[t <= trace_centre], t[t >= trace_centre])
        (power_left, power_right) = (power[t <= trace_centre], power[t >= trace_centre])

        moment_l = moment(t_left, power_left)
        moment_r = moment(t_right, power_right)

        fwhm_l = fwhm_from_var(t_left, power_left)
        fwhm_r = fwhm_from_var(t_right, power_right)
        
        if any(np.isnan([moment_l, moment_r, fwhm_l, fwhm_r])):
            return False

        mom_delay = moment_r - moment_l

        ## Gaussian fit method
        if force_split:
            mean_ubound_left = trace_centre
            mean_lbound_right = trace_centre
        else: 
            mean_ubound_left = t.max()
            mean_lbound_right = t.min()
        
        fwhm_lbound = 0.5    
        fwhm_ubound = t_range/2
        height_lbound = 0
        height_ubound = 4*t.max()
        #mean, fwhm, height
        lbounds_left = [t.min(), fwhm_lbound, height_lbound]
        ubounds_left = [mean_ubound_left, fwhm_ubound, height_ubound]
        lbounds_right = [mean_lbound_right, fwhm_lbound, height_lbound]
        ubounds_right = [t.max(), fwhm_ubound, height_ubound]

        #Parameter constraints
        bounds = (lbounds_left+lbounds_right, ubounds_left+ubounds_right)
        #Initial parameter values
        p0 = [t.min(), fwhm_ubound, height_ubound, t.max(), fwhm_ubound, height_ubound]

        opt_p, cov = optimize.curve_fit(double_gauss, t, power, bounds=bounds)
        residue = ((double_gauss(t, *opt_p) - power)**2).sum()
        errors = np.sqrt(np.diag(cov))
        if verbose:
            print('Residual sum of squares: {0:.2f}'.format(residue))

        delay = opt_p[3] - opt_p[0]
        pulse_fwhms = (opt_p[1], opt_p[4])

        self.results = dict(moment_delay=mom_delay,
                            moment_fwhms=(fwhm_l, fwhm_r),
                            fit_delay=delay,
                            fit_fwhms=pulse_fwhms,
                            fit_params=opt_p,
                            fit_sum_of_squares=residue,
                            fit_cov_mat=cov,
                            fit_errors=errors,
                            retr_agreement=agr,
                            t=t,
                            power=power)
        return True

def process_run_mpi(ds_string, calib_dir, out_filename):
    ds = psana.DataSource(ds_string)
    proc = XTCavProcessor()
    proc.set_data_source(ds)
    psana.setOption('psana.calib-dir', calib_dir)

    fit_delays = []
    fit_fwhms = []
    fit_errors = []
    moment_delays = []
    moment_fwhms = []
    agreement = []

    n_good = 0
    for idx, evt in enumerate(ds.events()):
        if idx%size != rank:
            continue
        ok = proc.set_event(evt)
        if not ok:
            continue
        ok = proc.process(agr_thresh=0., verbose=False, force_split=True)
        if not ok:
            continue
        fit_delays.append(proc.results['fit_delay'])
        fit_fwhms.append(proc.results['fit_fwhms'])
        fit_errors.append(proc.results['fit_errors'])
        moment_delays.append(proc.results['moment_delay'])
        moment_fwhms.append(proc.results['moment_fwhms'])
        agreement.append(proc.results['retr_agreement'])

        n_good += 1
        if n_good%100 == 0:
            print('Processed {0} events with {1} successes in rank {2}'.format(idx, n_good, rank))

    fit_delays = np.array(fit_delays)
    fit_fwhms = np.array(fit_fwhms)
    fit_errors = np.array(fit_errors)
    moment_delays = np.array(moment_delays)
    moment_fwhms = np.array(moment_fwhms)
    agreement = np.array(agreement)


    fit_delays = comm.gather(fit_delays, root=0)
    fit_fwhms = comm.gather(fit_fwhms, root=0)
    fit_errors = comm.gather(fit_errors, root=0)
    moment_delays = comm.gather(moment_delays, root=0)
    moment_fwhms = comm.gather(moment_fwhms, root=0)
    agreement = comm.gather(agreement, root=0)

    if rank == 0:
        fit_delays = np.concatenate(fit_delays)
        fit_fwhms = np.concatenate(fit_fwhms)
        fit_errors = np.concatenate(fit_errors)
        moment_delays = np.concatenate(moment_delays)
        moment_fwhms = np.concatenate(moment_fwhms)
        agreement = np.concatenate(agreement)

        d = dict(fit_delays=fit_delays, fit_fwhms=fit_fwhms, fit_erors=fit_errors,
                 moment_delays=moment_delays, moment_fwhms=moment_fwhms, agreement=agreement)
        with open(out_filename, 'wb') as file:
            pickle.dump(d, file)
    MPI.Finalize()

 
def main():
     process_run_mpi('exp=AMO/amon0816:run=221:smd', '/reg/d/psdm/amo/amon0816/calib', 'XTCav_test_out.p')


if __name__ == '__main__':
    main()
