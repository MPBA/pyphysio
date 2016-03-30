import numpy as _np
from ..BaseTool import _Tool
from ..PhUI import PhUI
from ..Signal import EvenlySignal, UnevenlySignal
from ..Filters import IIRFilter, ConvolutionalFilter, DeConvolutionalFilter, Diff

"""
Tools are generic operations that take as input a SIGNAL (or numpy array?) and gives as output one or more np.array.
"""

# TODO: len(signal) : can I use signal.duration?

class PeakDetection(_Tool):
	'''
    Estimate the maxima and the minima in the signal (in particular for periodic signals).
    
    Parameters
    ----------
    deltas : nparray or float
        The threshold for the detection of the peaks. If array it should have the same length of the signal.
    refractory : int
        Number of samples to skip after detection of a peak
    startMax : boolean
        Whether to start looking for a max.
    
    Returns
    -------
    maxs : nparray
        Array containing indexes (first column) and values (second column) of the maxima
    mins : nparray
        Array containing indexes (first column) and values (second column) of the minima
    '''
	
	@classmethod
	def algorithm(cls, signal, params):
		delta = params['delta']
		refractory = params['refractory']
		startMax = params['startMax']
		if _np.shape(delta) == ():
			deltas = _np.repeat(delta, len(signal))
		# initialization
		mn_candidate, mx_candidate = np.Inf, -np.Inf
		mnpos_candidate, mxpos_candidate = np.nan, np.nan
		
		lookformax = startmax

		i_activation_max = 0
		i_activation_min = 0

		i = 0
		while i<=len(signal)-1:
			this = signal[i]
			delta = deltas[i]
			
			if this > mx_candidate:
				mx_candidate = this
				mxpos_candidate = i
			if this < mn_candidate:
				mn_candidate = this
				mnpos_candidate = i
			
			if lookformax:
				if i >= i_activation_max and this < mx_candidate - delta: #new max
					maxtab.append((mxpos_candidate, mx_candidate))
					i_activation_max = i + refractory
					
					mn_candidate = this
					mnpos_candidate = i
					
					lookformax = False
			else:
				if i >= i_activation_min and this > mn_candidate + delta: # new min
					mintab.append((mnpos_candidate, mn_candidate))
					i_activation_min = i + refractory
					
					mx_candidate = this
					mxpos_candidate = i
					
					lookformax = True
			i += 1
		return(_np.array(maxtab), _np.array(mintab))
	
	@classmethod
    def check_params(cls, params):
		if not 'delta' in params:
			#default = 0 # GRAVE
			pass
		if not 'refractory' in params or params['refractory']<0:
			#default = 1 # OK
			pass
		if not 'startMax' in params:
			#default = True # OK
		return(params)

class Range(_Tool):
	'''
    Estimate the local range of the signal
    
    Parameters
    ----------
    winstep : int
        The increment indexes to start the next window
    winlen : int
        The dimension of the window
    smooth : boolean
        Whether to convolve the result with a gaussian window
    
    Returns
    -------
    deltas : nparray
        Result of estimation of local range
    '''
	
	@classmethod
	def algorithm(cls, signal, params):
		winlen = params['win_len']
		winstep = params['win_step']
		smooth = params['smooth']
		
		windows = _np.arange(0, len(signal)-winlen, winstep)
		deltas = _np.zeros(len(signal))
		for start in windows:
			portion_curr = signal[start: start+winlen]
			curr_delta = (_np.max(portion_curr) - _np.min(portion_curr))
			deltas[start:start+winlen] = curr_delta
		deltas[start+winlen:] = curr_delta
		
		if smooth:
			deltas = ConvolutionalFilter(irftype='gauss', N = winlen*2, normalize=True)(deltas) # TODO: check sintax
		return(deltas)
	
	@classmethod
    def check_params(cls, params):
		if not 'win_len' in params:
			#default = None # GRAVE
			pass
		if not 'win_step' in params:
			#default = None # GRAVE
			pass
		if not 'smooth' in params:
			#default = True # OK
			pass
		return(params)

class PSD(_Tool):
	"""
	"""
	
	@classmethod
	def algorithm(cls, signal, params):
		pass
	
	@classmethod
    def check_params(cls, params):
		return(params)

class Energy(_Tool):
	'''
    Estimate the local energy of the signal
    
    Parameters
    ----------
    winstep : int
        The increment indexes to start the next window
    winlen : int
        The dimension of the window
    smooth : boolean
        Whether to convolve the result with a gaussian window
    
    Returns
    -------
    energy : nparray
        Result of estimation of local energy
    '''
	
	@classmethod
	def algorithm(cls, signal, params):
		winlen = params['win_len']
		winstep = params['win_step']
		smooth = params['smooth']
		
		windows = _np.arange(0, len(signal)-winlen, winstep)
		
		energy = []    
		for start in windows:
			portion_curr = signal[start: start+winlen]
			curr_energy = _np.sum(portion_curr**2)/len(portion_curr)
			energy.append(curr_energy)
		energy.append(curr_energy)
		energy.insert(0, energy[0])
		
		idx_interp = _np.r_[0, windows+round(winlen/2), len(signal)]
		energy = _np.array(energy)
		energy_out = flt.interpolate_unevenly(energy, idx_interp, 1, kind='linear') # TODO: FIX THIS
    
		if smooth:
			energy_out = ConvolutionalFilter(irftype='gauss', N = winlen*2, normalize=True)(energy_out) # TODO: check sintax
		
		return(energy_out)
	
		@classmethod
    def check_params(cls, params):
		if not 'win_len' in params:
			#default = None # GRAVE
			pass
		if not 'win_step' in params:
			#default = None # GRAVE
			pass
		if not 'smooth' in params:
			#default = True # OK
			pass
		return(params)

class Maxima(_Tool):
	"""
	"""
	
	@classmethod
	def algorithm(cls, signal, params):
		pass
	
	@classmethod
    def check_params(cls, params):
		return(params)

class Minima(_Tool):
	"""
	"""
	
	@classmethod
	def algorithm(cls, signal, params):
		pass
	
	@classmethod
    def check_params(cls, params):
		return(params)

class CreateTemplate(_Tool):
	"""
	"""
	
	@classmethod
	def algorithm(cls, signal, params):
		pass
	
	@classmethod
    def check_params(cls, params):
		return(params)

class OptimizeBateman(_Tool):
	"""
	"""
	
	@classmethod
	def algorithm(cls, signal, params):
		pass
	
	@classmethod
    def check_params(cls, params):
		return(params)
	
	def _loss_function(par_bat, signal, fsamp, delta = 20, verbose=False):
		"""
		Computes the loss for optimization of Bateman parameters.
		
		Parameters
		----------
		par_bat : list
			Bateman parameters to be optimized
		signal : nparray
			The EDA signal
		fsamp : float
			The sampling frequency
		delta : float
			Minimum amplitude of the peaks in the driver
		verbose : boolean
			Print Bateman parameters and loss at each evaluation
		
		Returns
		-------
		loss : float
			The computed loss
		"""
		T1_MIN = 0.04
		T2_MAX = 15
		if par_bat[0] < T1_MIN or par_bat[1]>T2_MAX or par_bat[0]>=par_bat[1]:
			return(10000)

		driver = estimate_driver(signal, fsamp, par_bat)
		loss = _response_energy(driver, fsamp, delta)
		if verbose:
			print(par_bat, loss)
		return(loss)

	def _response_energy(driver, fsamp, delta):
		"""
		Estimates the phasic and tonic components of a driver.
		
		It uses a detection algorithm based on the derivative of the driver.
		
		Parameters
		----------
		driver : nparray
			The driver signal
		fsamp : float
			The sampling frequency
		delta : float
			Minimum amplitude of the peaks in the driver
			
		Returns
		-------
		response : float
			Response energy computed on the driving signal
		
		Raises
		------
		RuntimeWarning: if no eligible peaks are found to compute the response energy
		
		Notes
		-----
		See paper ...
		"""
		maxs, mins = tll.peakdet(driver, delta)
		
		if len(maxs)!=0:
			idx_maxs = maxs[:,0]
		else: 
			warnings.warn('Unable to find peaks in driver signal for computation of Energy. Returning Inf', RuntimeWarning)
			return(10000)
		
		#STAGE 1: select maxs distant from the others
		diff_maxs = np.diff(np.r_[idx_maxs, len(driver)-1])
		th_diff = 15*fsamp
		
		# TODO: select th such as to have enough maxs, e.g. diff_maxs_tentative = np.median(diff_maxs)
		idx_selected_maxs = np.where(diff_maxs>th_diff)[0]
		selected_maxs = idx_maxs[idx_selected_maxs]
		
		if len(selected_maxs)!=0:
			energy = 0
			for idx_max in selected_maxs:
				driver_portion = driver[idx_max:idx_max + 15*fsamp]
				
				half = len(driver_portion) - 5 * fsamp
				
				y = driver_portion[half:]
				diff_y = np.diff(y)
				th_75 = np.percentile(diff_y, 75)
				th_25 = np.percentile(diff_y, 25)
				
				idx_sel_diff_y = np.where((diff_y > th_25) & (diff_y < th_75))[0]
				diff_y_sel = diff_y[idx_sel_diff_y]
				
	#            A2, B = _find_slope(y, half)
				mean_s = tll.bootstrap_estimation(diff_y_sel, np.mean, N=100, k=0.5)
				
				mean_y =  tll.bootstrap_estimation(y, np.median)
				B_mean_s = mean_y - mean_s * (half + (len(driver_portion)-half)/2)
				
	#            line_A2 = A2 * np.arange(len(driver_portion)) + B
				line_mean_s = mean_s * np.arange(len(driver_portion)) + B_mean_s

	#            plt.figure()
	#            plt.plot(diff_y)
	#            plt.plot(idx_sel_diff_y, diff_y_sel, 'o')
	#            plt.hlines(mean_s, 0, len(diff_y))
	#            plt.plot(driver_portion)
	#            plt.plot(np.arange(half, half+len(y)), y, 'r', linewidth=1.2)
	#            plt.plot(line_mean_s, 'k')
	#            plt.show()

				driver_detrended  = driver_portion - line_mean_s
	#            if driver_detrended[0] < 0:
	#                print('driver0')
				driver_detrended = driver_detrended/np.max(driver_detrended)
				energy_curr = (1/fsamp) * np.sum(driver_detrended[fsamp:]**2) / (len(driver_detrended)-fsamp)
				
	#                driver_portion = driver_detrended / driver_portion[0]
	#                idx_negative = np.where(driver_portion<0)[0]
	#                
	#                if len(idx_negative)!=0:
	#                    idx_t1 = idx_negative[0]
	#                    driver_t1_on = driver_portion[idx_t1:]
	#                    energy_curr = (1/fsamp) * np.sum(driver_t1_on**2) / len(driver_t1_on)
	#                else: 
	#                    energy_curr = (1/fsamp) * np.sum(driver_portion**2) / len(driver_portion)
	#            else:
				energy += energy_curr
			return(energy)
			
		else: 
			warnings.warn('Peaks found but too near. Returning Inf', RuntimeWarning)
			return(10000)
	#            else:
	#                driver_portion = driver_detrended / abs(driver_detrended[0])
	#                energy_curr = (1/fsamp)*np.sum(driver_detrended**2)/len(driver_detrended)

class BootstrapEstimation(_Tool):
	"""
    Perform a bootstrapped estimation of given statistical indicator
    
    Parameters
    ----------
    func : numpy function
        The estimator. Must accept data as input
    N : int
        Number of iterations
    k : float (0-1)
        Portion of data to be used at each iteration
    
    Returns
    -------
    estim : float
        The bootstrapped estimate
    """
	
	@classmethod
	def algorithm(cls, signal, params):
		L = len(signal) # TODO: can I use signal.duration?
		niter = params['niter']
		func = params['func']
		
		estim = []
		for i in range(niter):
			idxs = _np.arange(L)
			idxs_p = _np.random.permutation(idxs)
			sampled_data = data[ idxs_p[:round(k*L)] ]
			curr_est = func(sampled_data)
			estim.append(curr_est)
		return(_np.mean(estim))
	
	@classmethod
    def check_params(cls, params):
		if not 'func' in params:
			#default = _np.mean # GRAVE
			pass
		if not 'niter' in params:
			#default = 100 # OK
			pass
		if not 'k' in params:
			#default = 0.5 # OK
		return(params)
