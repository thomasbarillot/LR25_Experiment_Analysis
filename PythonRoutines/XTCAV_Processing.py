import re
import sys
import pickle
import argparse
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
    # w = fwhm/1.76
    return height*np.exp(- ((x-mean)/w)**2)
    # return height/np.cosh((x-mean)/w)**2


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
    variance = var(x, y, axis)
    if variance < 0:
        return -1.
    return 2*np.sqrt(2*np.log(2))*np.sqrt(variance)


def smooth(signal, box_width):
    box = np.ones(box_width)/box_width
    return np.convolve(signal, box, mode='same')


def full_width(x, y, frac=0.5):
    try:
        y_thr = (y >= frac*y.max())
    except ValueError:
        return 0
    idcs_above_thr = y_thr.nonzero()[0]
    idcs_lims = np.array([idcs_above_thr[0], idcs_above_thr[-1]])
    x_lims = x[idcs_lims]
    return np.abs(x_lims[1] - x_lims[0])


class XTCavProcessor:
    def __init__(self):
        self.s2s = ShotToShotCharacterization.ShotToShotCharacterization()
        self.ds = None
        self.event_ok = False
        self.results = None
        self.n_processed = 0
        self.n_success = 0

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

    def process(self, force_split=True, smoothing=5, use_abs=True, agr_thresh=0.5,
                fwhm_low_lim=3, fwhm_up_lim=20, verbose=False):
        self.n_processed += 1
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
        
        t = t.squeeze() # remove extra dimensions (i.e. bunch index)
        power = power.squeeze()
        t_range = t.max() - t.min()
        if use_abs:
            power = np.abs(power)
        if smoothing:
            power = smooth(power, smoothing)
        
        # Find where the trace actually has support
        thresh = power.max()*0.05
        power_nonz = (power > thresh).nonzero()
        # First and last points in time where power > thresh
        t_lims = (t[power_nonz[0][0]], t[power_nonz[0][-1]])
        trace_centre = np.mean(t_lims)
        if verbose:
            print('Trace extends from {0} fs to {1} fs, centre at {2:.1f} fs'.format(
                t_lims[0], t_lims[1], trace_centre))

        ## Moments method
        # Time and power arrays on both sides of trace:
        (t_left, t_right) = (t[t <= trace_centre], t[t >= trace_centre])
        (power_left, power_right) = (power[t <= trace_centre], power[t >= trace_centre])
	    # Centre of gravity for pulse centre
        moment_l = moment(t_left, power_left)
        moment_r = moment(t_right, power_right)

        # Second moment (variance) for fwhm width:
        fwhm_l = fwhm_from_var(t_left, power_left)
        fwhm_r = fwhm_from_var(t_right, power_right)
        
        # Check that we haven't got nonsense for the moments or widths:
        if any(np.isnan([moment_l, moment_r, fwhm_l, fwhm_r])):
            return False
        if moment_l < t.min() or moment_r > t.max():
            return False

        mom_delay = moment_r - moment_l

        ## Threshold method for pulse duration
        # Better estimate of centre of trace:
        trace_centre = np.mean([moment_l, moment_r])
        # Sometimes goes wrong if the trace is messy, checking for that
        if trace_centre < t.min() or trace_centre > t.max():
            return False
        (t_left, t_right) = (t[t <= trace_centre], t[t >= trace_centre])
        (power_left, power_right) = (power[t <= trace_centre], power[t >= trace_centre])
        w_left = full_width(t_left, power_left, 1/2.71)
        w_right = full_width(t_right, power_right, 1/2.71)

        if verbose:
            print('Updated trace centre at {0}, moments at ({1}, {2})'.format(
                trace_centre, moment_l, moment_r))


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
        p0 = [moment_l, fwhm_l, power_left.max(), moment_r, fwhm_r, power_right.max()]


        try:
            opt_p, cov = optimize.curve_fit(double_gauss, t, power, bounds=bounds)
        except Exception as e:
            print('Fit failed with exception'+e)
            return False

        residue = ((double_gauss(t, *opt_p) - power)**2).sum()
        errors = np.sqrt(np.diag(cov))
        if verbose:
            print('Residual sum of squares: {0:.2f}'.format(residue))

        fit_delay = opt_p[3] - opt_p[0]
        fit_fwhms = (opt_p[1], opt_p[4])

        all_fwhms = np.array(fit_fwhms+(fwhm_l, fwhm_r))
        if  any(all_fwhms < fwhm_low_lim) or any(all_fwhms > fwhm_up_lim):
            return False
        if fit_delay < 1:
            return False
        self.results = dict(moment_pulse_t0=(moment_l, moment_r),
                            moment_delay=mom_delay,
                            moment_fwhms=(fwhm_l, fwhm_r),
                            threshold_widths=(w_left, w_right),
                            fit_pulse_t0=(opt_p[0], opt_p[3]),
                            fit_delay=fit_delay,
                            fit_fwhms=fit_fwhms,
                            fit_params=opt_p,
                            fit_sum_of_squares=residue,
                            fit_cov_mat=cov,
                            fit_errors=errors,
                            retr_agreement=agr,
                            t=t,
                            power=power)
        self.n_success += 1
        return True

def process_run_mpi(ds_string, calib_dir, out_filename):
    ds = psana.DataSource(ds_string)
    proc = XTCavProcessor()
    proc.set_data_source(ds)
    psana.setOption('psana.calib-dir', calib_dir)

    # Keys in this dict must be the same as in XTCAV_Processor.results
    arrays = dict(fit_delay=[],
                  fit_fwhms=[],
                  fit_errors=[],
                  moment_delay=[],
                  moment_fwhms=[],
                  threshold_widths=[],
                  retr_agreement=[])
    evt_idx = []

    n_good = 0
    n = 0
    for idx, evt in enumerate(ds.events()):
        if idx%size != rank:
            continue
        n += 1
        proc.set_event(evt)
        ok = proc.process(agr_thresh=0.5, verbose=False, force_split=True)
        if not ok:
            continue
        for key in arrays.keys():
            arrays[key].append(proc.results[key])
        evt_idx.append(idx)
        n_good += 1
        if n_good%100 == 0:
            print('Processed {0} events with {1} successes in rank {2}'.format(n, n_good, rank))

    for key, array in arrays.items():
        arrays[key] = np.array(array)
    evt_idx = np.array(evt_idx)

    gathered_arrays = {key: comm.gather(array, root=0) for key, array in arrays.items()}
    gathered_evt_index = comm.gather(evt_idx, root=0)

    if rank == 0:
        concatenated_arrays = {key: np.concatenate(array) for key, array in gathered_arrays.items()}
        concatenated_arrays['evt_idx'] = np.concatenate(gathered_evt_index)
        with open(out_filename, 'wb') as file:
            pickle.dump(concatenated_arrays, file)
    MPI.Finalize()
    return arrays

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process XTCav traces from a single run (MPI enabled)')
    parser.add_argument('datasource', help='Data source string (e.g. exp=AMO/amolr2516:run=55)')
    parser.add_argument('-o', dest='output', help='Output file name')
    args = parser.parse_args()
    if args.output is None:
        exp_matches = re.findall('exp=AMO/[0-9a-z]+', args.datasource)
        run_matches = re.findall('run=[0-9]+', args.datasource)
        if not run_matches or not exp_matches:
            raise ValueError('Could not generate automatic file name from datasource string {0}'.format(datasource))
        exp = exp_matches[0].split('/')[1]
        run = run_matches[0].split('=')[1]
        out_file = 'xtcav_processed_{0}_run{1}'.format(exp, run)
    else:
        out_file = args.output
    process_run_mpi(args.datasource, '/reg/d/psdm/amo/amon0816/calib', out_file)

