import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

import pandas as pd

def response_detection_core(pdf,samplingrate = 30000,baseline_window = 1,thr = 3.5):

	baseline = np.mean(pdf[:samplingrate*baseline_window])
	baseline_std = np.std(pdf[:samplingrate*baseline_window])
	max_response = np.max(pdf)
	thr = baseline+thr*baseline_std

	width = np.argmax(pdf)+(0.25*samplingrate)

	if width > pdf.size:
		width = pdf.size-np.argmax(pdf)
	   
	if (max_response > thr) & (np.mean(pdf[np.argmax(pdf):int(width)]) > thr):
		return 'responsive'
		
	else:
		return 'non_responsive'

def calculate_base_peak_core(raster,selection,window = 0.3,baselineend = None,
						stimulus_start = None,samplingrate = 30000): # input needs to be row of df
	
	baselinestart = 0
	stimulus_length = int(selection.duration+1)
	
	start_base = np.random.choice(np.arange(int(baselinestart*samplingrate),
										int((baselineend-window)*samplingrate)),size = 1000)
	baselines = np.zeros([start_base.size])

	for i in range(start_base.size):
		baselines[i] = np.sum(raster[:,int(start_base[i]):int(start_base[i]+(window*samplingrate))])/raster.shape[0]/window

	baseline_fr = np.mean(baselines)
	baseline_fr_std = np.std(baselines)

	start_stim = np.random.choice(int(samplingrate*(stimulus_length-window)),size = 5000)
	start_stim = stimulus_start*samplingrate + start_stim
	stimuli = np.zeros([start_stim.size])

	for i in range(start_stim.size):
		stimuli[i] = np.sum(raster[:,int(start_stim[i]):int(start_stim[i]+(window*samplingrate))])/raster.shape[0]/window

	stimulus_fr = np.max(stimuli)

	return  (baseline_fr, stimulus_fr,baseline_fr_std)


def response_time_core(pdf,stimulus_start = 1,samplingrate = 30000):
	
	responsetime = np.argmax(pdf)-stimulus_start*samplingrate
	
	return responsetime


def response_duration(pdf,peak_time,stimulus_start=1, samplingrate=30000):
	count_neg = 0
	count_pos = 0

	baseline = np.mean(pdf[:int(stimulus_start*samplingrate)])
	peak = pdf[int(peak_time+stimulus_start*samplingrate)]
	half_Peak = baseline+(peak- baseline)/2


	while pdf[int((peak_time+stimulus_start*samplingrate)-count_neg)] > half_Peak:
		count_neg +=1
		if int((peak_time+stimulus_start*samplingrate)-count_neg) < 0:
			break

	while pdf[int((peak_time+stimulus_start*samplingrate)+count_pos)] > half_Peak:
		count_pos +=1
		if int((peak_time+stimulus_start*samplingrate)+count_pos) > pdf.size-1:
			break

	start = (peak_time+stimulus_start*samplingrate)-count_neg
	end = (peak_time+stimulus_start*samplingrate)+count_pos
	duration = end-start
	
	return start,end,duration


def detect_onset(x, threshold=0, n_above=1, n_below=0,
				 threshold2=None, n_above2=1, show=False, ax=None):
	"""Detects onset in data based on amplitude threshold.

	Parameters
	----------
	x : 1D array_like
		data.
	threshold : number, optional (default = 0)
		minimum amplitude of `x` to detect.
	n_above : number, optional (default = 1)
		minimum number of continuous samples >= `threshold`
		to detect (but see the parameter `n_below`).
	n_below : number, optional (default = 0)
		minimum number of continuous samples below `threshold` that
		will be ignored in the detection of `x` >= `threshold`.
	threshold2 : number or None, optional (default = None)
		minimum amplitude of `n_above2` values in `x` to detect.
	n_above2 : number, optional (default = 1)
		minimum number of samples >= `threshold2` to detect.
	show  : bool, optional (default = False)
		True (1) plots data in matplotlib figure, False (0) don't plot.
	ax : a matplotlib.axes.Axes instance, optional (default = None).

	Returns
	-------
	inds : 1D array_like [indi, indf]
		initial and final indeces of the onset events.

	Notes
	-----
	You might have to tune the parameters according to the signal-to-noise
	characteristic of the data.

	See this IPython Notebook [1]_.

	References
	----------
	.. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectOnset.ipynb

	Examples
	--------
	>>> from detect_onset import detect_onset
	>>> x = np.random.randn(200)/10
	>>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
	>>> detect_onset(x, np.std(x[:50]), n_above=10, n_below=0, show=True)

	>>> x = np.random.randn(200)/10
	>>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
	>>> x[80:140:20] = 0
	>>> detect_onset(x, np.std(x[:50]), n_above=10, n_below=0, show=True)

	>>> x = np.random.randn(200)/10
	>>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
	>>> x[80:140:20] = 0
	>>> detect_onset(x, np.std(x[:50]), n_above=10, n_below=2, show=True)

	>>> x = [0, 0, 2, 0, np.nan, 0, 2, 3, 3, 0, 1, 1, 0]
	>>> detect_onset(x, threshold=1, n_above=1, n_below=0, show=True)

	>>> x = np.random.randn(200)/10
	>>> x[11:41] = np.ones(30)*.3
	>>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
	>>> x[80:140:20] = 0
	>>> detect_onset(x, .2, n_above=10, n_below=1, show=True)

	>>> x = np.random.randn(200)/10
	>>> x[11:41] = np.ones(30)*.3
	>>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
	>>> x[80:140:20] = 0
	>>> detect_onset(x, .4, n_above=10, n_below=1, show=True)

	>>> x = np.random.randn(200)/10
	>>> x[11:41] = np.ones(30)*.3
	>>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
	>>> x[80:140:20] = 0
	>>> detect_onset(x, .2, n_above=10, n_below=1,
					 threshold2=.4, n_above2=5, show=True)

	Version history
	---------------
	'1.0.6':
		- Deleted 'from future import'
		+ Added parameters `threshold2` and `n_above2`
	"""

	x = np.atleast_1d(x).astype('float64')
	# deal with NaN's (by definition, NaN's are not greater than threshold)
	x[np.isnan(x)] = -np.inf
	# indices of data greater than or equal to threshold
	inds = np.nonzero(x >= threshold)[0]
	if inds.size:
		# initial and final indexes of almost continuous data
		inds = np.vstack((inds[np.diff(np.hstack((-np.inf, inds))) > n_below+1], \
						  inds[np.diff(np.hstack((inds, np.inf))) > n_below+1])).T
		# indexes of almost continuous data longer than or equal to n_above
		inds = inds[inds[:, 1]-inds[:, 0] >= n_above-1, :]
		# minimum amplitude of n_above2 values in x to detect
		if threshold2 is not None and inds.size:
			idel = np.ones(inds.shape[0], dtype=bool)
			for i in range(inds.shape[0]):
				if np.count_nonzero(x[inds[i, 0]: inds[i, 1]+1] >= threshold2) < n_above2:
					idel[i] = False
			inds = inds[idel, :]
	if not inds.size:
		inds = np.array([])  # standardize inds shape for output
	if show and x.size > 1:  # don't waste my time ploting one datum
		_plot(x, threshold, n_above, n_below, threshold2, n_above2, inds, ax)

	return inds


def _plot(x, threshold, n_above, n_below, threshold2, n_above2, inds, ax):
	"""Plot results of the detect_onset function, see its help."""
	try:
		import matplotlib.pyplot as plt
	except ImportError:
		print('matplotlib is not available.')
	else:
		if ax is None:
			_, ax = plt.subplots(1, 1, figsize=(8, 4))

		if inds.size:
			for (indi, indf) in inds:
				if indi == indf:
					ax.plot(indf, x[indf], 'ro', mec='r', ms=6)
				else:
					ax.plot(range(indi, indf+1), x[indi:indf+1], 'r', lw=1)
					ax.axvline(x=indi, color='b', lw=1, ls='--')
				ax.axvline(x=indf, color='b', lw=1, ls='--')
			inds = np.vstack((np.hstack((0, inds[:, 1])),
							  np.hstack((inds[:, 0], x.size-1)))).T
			for (indi, indf) in inds:
				ax.plot(range(indi, indf+1), x[indi:indf+1], 'k', lw=1)
		else:
			ax.plot(x, 'k', lw=1)
			ax.axhline(y=threshold, color='r', lw=1, ls='-')

		ax.set_xlim(-.02*x.size, x.size*1.02-1)
		ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
		yrange = ymax - ymin if ymax > ymin else 1
		ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
		ax.set_xlabel('Data #', fontsize=14)
		ax.set_ylabel('Amplitude', fontsize=14)
		if threshold2 is not None:
			text = 'threshold=%.3g, n_above=%d, n_below=%d, threshold2=%.3g, n_above2=%d'
		else:
			text = 'threshold=%.3g, n_above=%d, n_below=%d, threshold2=%r, n_above2=%d'            
		#ax.set_title(text % (threshold, n_above, n_below, threshold2, n_above2))
		ax.set_title(inds[0,:])
		# plt.grid()
		plt.show()

def plot_PSTH_gaussian_cv(selection, times_cut,trials=None,samplingrate=30000,cv = None,start_rel=None,end_rel=None):
	
	# selection needs to be row of dataframe
	
	from get_response_prop import calculate_bandwidth,response_detection_core

		
	times_cut = np.sort(times_cut)
	pdf,bw,x_grid,bins = calculate_bandwidth(times_cut,cv=cv) # start=start_rel,end=end_rel

	response = response_detection_core(pdf,samplingrate=samplingrate)

	fig, ax1 = plt.subplots()
	ax1.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.1f // binsize=%.1f' % ((bw/30),(bins[1]-bins[0])/30))
	ax2 = ax1.twinx()
	ax2.hist(times_cut, bins, fc='gray', histtype='stepfilled', alpha=0.3)
	#ax1.legend(loc='upper left')

	if response == 'responsive':   
		ax1.set_title(str(selection.expID)+'//'+str(selection.cluster)+'responsive')
	else:
		ax1.set_title(str(selection.expID)+'//'+str(selection.cluster)+'non_responsive')

	plt.show()
	plt.close()

	return pdf,response,fig

def plot_PSTH_gaussian_manual(ax1,selection, times_cut,samplingrate=30000,binsize = None,bandwidth= None,start=None,end=None,stepsize = None,pdf = True):

    # selection needs to be row of dataframe

    #from get_response_prop import calculate_bandwidth,response_detection_core,build_pdf


    times_cut = np.sort(times_cut)
    pdf = build_pdf(times_cut,start,end,samplingrate=30000,kernel = 'gaussian',bw = 150,stepsize=stepsize)


    bins = np.arange(start*samplingrate,end*samplingrate,binsize*samplingrate/1000)
    x_grid = np.arange(start*samplingrate,end*samplingrate,stepsize*samplingrate/1000)

    n, bins, patches= ax1.hist(times_cut, bins, fc='black', histtype='stepfilled', alpha=0.8,)
    
    #ax2 = ax1.twinx()
    #test = ax2.plot(x_grid, pdf, linewidth=2, alpha=0.5, color='black')#,label='bw=%.1f ms // binsize=%.1f ms' % (bandwidth,binsize))
    #ax2.legend(loc='lower left')

   


    

    #ax2.set_yticks([])
    ax1.set_xticks(np.array([4,9,14,19])*samplingrate)
    ax1.set_xticklabels(['-5','0','5','10'])
    #plt.show()
    #plt.close()

    return n
# %load ./../functions/detect_cusum.py
"""Cumulative sum algorithm (CUSUM) to detect abrupt changes in data."""

'''from __future__ import division, print_function
import numpy as np

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = "1.0.4"
__license__ = "MIT"
'''

def detect_cusum(x, threshold=1, drift=0, ending=False, show=True, ax=None):
    """Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.

    Parameters
    ----------
    x : 1D array_like
        data.
    threshold : positive number, optional (default = 1)
        amplitude threshold for the change in the data.
    drift : positive number, optional (default = 0)
        drift term that prevents any change in the absence of change.
    ending : bool, optional (default = False)
        True (1) to estimate when the change ends; False (0) otherwise.
    show : bool, optional (default = True)
        True (1) plots data in matplotlib figure, False (0) don't plot.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ta : 1D array_like [indi, indf], int
        alarm time (index of when the change was detected).
    tai : 1D array_like, int
        index of when the change started.
    taf : 1D array_like, int
        index of when the change ended (if `ending` is True).
    amp : 1D array_like, float
        amplitude of changes (if `ending` is True).

    Notes
    -----
    Tuning of the CUSUM algorithm according to Gustafsson (2000)[1]_:
    Start with a very large `threshold`.
    Choose `drift` to one half of the expected change, or adjust `drift` such
    that `g` = 0 more than 50% of the time.
    Then set the `threshold` so the required number of false alarms (this can
    be done automatically) or delay for detection is obtained.
    If faster detection is sought, try to decrease `drift`.
    If fewer false alarms are wanted, try to increase `drift`.
    If there is a subset of the change times that does not make sense,
    try to increase `drift`.

    Note that by default repeated sequential changes, i.e., changes that have
    the same beginning (`tai`) are not deleted because the changes were
    detected by the alarm (`ta`) at different instants. This is how the
    classical CUSUM algorithm operates.

    If you want to delete the repeated sequential changes and keep only the
    beginning of the first sequential change, set the parameter `ending` to
    True. In this case, the index of the ending of the change (`taf`) and the
    amplitude of the change (or of the total amplitude for a repeated
    sequential change) are calculated and only the first change of the repeated
    sequential changes is kept. In this case, it is likely that `ta`, `tai`,
    and `taf` will have less values than when `ending` was set to False.

    See this IPython Notebook [2]_.

    References
    ----------
    .. [1] Gustafsson (2000) Adaptive Filtering and Change Detection.
    .. [2] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectCUSUM.ipynb

    Examples
    --------
    >>> from detect_cusum import detect_cusum
    >>> x = np.random.randn(300)/5
    >>> x[100:200] += np.arange(0, 4, 4/100)
    >>> ta, tai, taf, amp = detect_cusum(x, 2, .02, True, True)

    >>> x = np.random.randn(300)
    >>> x[100:200] += 6
    >>> detect_cusum(x, 4, 1.5, True, True)

    >>> x = 2*np.sin(2*np.pi*np.arange(0, 3, .01))
    >>> ta, tai, taf, amp = detect_cusum(x, 1, .05, True, True)
    """

    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta, tai, taf = np.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    amp = np.array([])
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < 0:
            gp[i], tap = 0, i
        if gn[i] < 0:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = np.append(ta, i)    # alarm index
            tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm
    # THE CLASSICAL CUSUM ALGORITHM ENDS HERE

    # Estimation of when the change ends (offline form)
    if tai.size and ending:
        _, tai2, _, _ = detect_cusum(x[::-1], threshold, drift, show=False)
        taf = x.size - tai2[::-1] - 1
        # Eliminate repeated changes, changes that have the same beginning
        tai, ind = np.unique(tai, return_index=True)
        ta = ta[ind]
        # taf = np.unique(taf, return_index=False)  # corect later
        if tai.size != taf.size:
            if tai.size < taf.size:
                taf = taf[[np.argmax(taf >= i) for i in ta]]
            else:
                ind = [np.argmax(i >= ta[::-1])-1 for i in taf]
                ta = ta[ind]
                tai = tai[ind]
        # Delete intercalated changes (the ending of the change is after
        # the beginning of the next change)
        ind = taf[:-1] - tai[1:] > 0
        if ind.any():
            ta = ta[~np.append(False, ind)]
            tai = tai[~np.append(False, ind)]
            taf = taf[~np.append(ind, False)]
        # Amplitude of changes
        amp = x[taf] - x[tai]

    if show:
        _plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn)

    return ta, tai, taf, amp


def _plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn):
    """Plot results of the detect_cusum function, see its help."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        t = range(x.size)
        ax1.plot(t, x, 'b-', lw=2)
        if len(ta):
            ax1.plot(tai, x[tai], '>', mfc='g', mec='g', ms=10,
                     label='Start')
            if ending:
                ax1.plot(taf, x[taf], '<', mfc='g', mec='g', ms=10,
                         label='Ending')
            ax1.plot(ta, x[ta], 'o', mfc='r', mec='r', mew=1, ms=5,
                     label='Alarm')
            ax1.legend(loc='best', framealpha=.5, numpoints=1)
        ax1.set_xlim(-.01*x.size, x.size*1.01-1)
        ax1.set_xlabel('Data #', fontsize=14)
        ax1.set_ylabel('Amplitude', fontsize=14)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax1.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax1.set_title('Time series and detected changes ' +
                      '(threshold= %.3g, drift= %.3g): N changes = %d'
                      % (threshold, drift, len(tai)))
        ax2.plot(t, gp, 'y-', label='+')
        ax2.plot(t, gn, 'm-', label='-')
        ax2.set_xlim(-.01*x.size, x.size*1.01-1)
        ax2.set_xlabel('Data #', fontsize=14)
        ax2.set_ylim(-0.01*threshold, 1.1*threshold)
        ax2.axhline(threshold, color='r')
        ax1.set_ylabel('Amplitude', fontsize=14)
        ax2.set_title('Time series of the cumulative sums of ' +
                      'positive and negative changes')
        ax2.legend(loc='best', framealpha=.5, numpoints=1)
        plt.tight_layout()
        plt.show()

def cut_data(selection,data,start_rel = None,end_rel = None,trials = None,samplingrate = None):
    
    '''function needs df row as input. also data needs to be 1d
        start_rel & end_rel means relative to stimulus start'''
    
    start = selection.pre -start_rel
    end =  selection.pre +end_rel
    
    
    raster = np.zeros([trials,int((end-start)*samplingrate)])
    times = (data[:,0]/samplingrate)
    times_inx = data[:,1]
    times_cut = ((times[(times> start)&(times< end)]*samplingrate)-start*samplingrate).astype(int)
    times_inx_cut = times_inx[(times> start)&(times< end)].astype(int)
    raster[times_inx_cut,times_cut] = 1
    #cumsum_trials = np.cumsum(raster,axis=1)
    cumsum = np.sum((np.cumsum(raster,axis=1)),axis = 0)
    
    return raster,np.sort(times_cut),cumsum

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_raster(data_tmp,ax1,trials):
    for i in range(trials):
        
        dots_x = (data_tmp[:,0][data_tmp[:,1]==i]).astype(int)
        dots = np.ones(dots_x.size)+i
        ax1.scatter(dots_x,dots,s=0.4,color='black')
    #cleanAxes(ax1,total=True)
    #plt.show()
    return

def placeAxesOnGrid(fig,dim=[1,1],xspan=[0,1],yspan=[0,1],wspace=None,hspace=None,):
    '''
    Takes a figure with a gridspec defined and places an array of sub-axes on a portion of the gridspec
    
    Takes as arguments:
        fig: figure handle - required
        dim: number of rows and columns in the subaxes - defaults to 1x1
        xspan: fraction of figure that the subaxes subtends in the x-direction (0 = left edge, 1 = right edge)
        yspan: fraction of figure that the subaxes subtends in the y-direction (0 = top edge, 1 = bottom edge)
        wspace and hspace: white space between subaxes in vertical and horizontal directions, respectively
        
    returns:
        subaxes handles
        
    written by doug ollerenshaw
    '''
    import matplotlib.gridspec as gridspec

    outer_grid = gridspec.GridSpec(100,100)
    inner_grid = gridspec.GridSpecFromSubplotSpec(dim[0],dim[1],
                                                  subplot_spec=outer_grid[int(100*yspan[0]):int(100*yspan[1]),int(100*xspan[0]):int(100*xspan[1])],
                                                  wspace=wspace, hspace=hspace)
    

    #NOTE: A cleaner way to do this is with list comprehension:
    # inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
    inner_ax = dim[0]*[dim[1]*[fig]] #filling the list with figure objects prevents an error when it they are later replaced by axis handles
    inner_ax = np.array(inner_ax)
    idx = 0
    for row in range(dim[0]):
        for col in range(dim[1]):
            inner_ax[row][col] = plt.Subplot(fig, inner_grid[idx])
            fig.add_subplot(inner_ax[row,col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist() #remove redundant dimension
    return inner_ax


def cleanAxes(ax,bottomLabels=False,leftLabels=False,rightLabels=False,topLabels=False,total=False):
    ax.tick_params(axis='both',labelsize=10)
    ax.spines['top'].set_visible(False);
    ax.yaxis.set_ticks_position('left');
    ax.spines['right'].set_visible(False);
    ax.xaxis.set_ticks_position('bottom')
    if not bottomLabels or topLabels:
        ax.set_xticklabels([])
    if not leftLabels or rightLabels:
        ax.set_yticklabels([])
    if rightLabels:
        ax.spines['right'].set_visible(True);
        ax.spines['left'].set_visible(False);
        ax.yaxis.set_ticks_position('right');
    if total:
        ax.set_frame_on(False);
        ax.set_xticklabels('',visible=False);
        ax.set_xticks([]);
        ax.set_yticklabels('',visible=False);
        ax.set_yticks([])

def plot_onset_cumsum(cumsum,raster,trials = None,time_limit_peak = None,samplingrate = 30000,
						ax = None,ax_zoom = None,start = None,end=None):
    
    cumsum = cumsum/cumsum[-1]
    ref = np.linspace(cumsum[0],cumsum[-1],cumsum.size)
    diff = cumsum-ref
    onset = np.argmin(diff[:int(time_limit_peak*samplingrate)])


        
    ax.plot([0,cumsum.size],[ref[0],ref[-1]],'--',color = 'black')
    ax.plot(diff,color='green')
    ax.plot(cumsum,color='blue',label=str((onset/samplingrate)-start)[:5]+' [s]')
    raster_plot=np.where(raster==1)
    ax.scatter(raster_plot[1],raster_plot[0]/trials/2,s=.5,color = 'grey')
    #ax.set_xlim(onset-.5*samplingrate,onset+.5*samplingrate)
    #plt.show()
    
    ax_zoom.plot([0,cumsum.size],[ref[0],ref[-1]],'--',color = 'black')
    ax_zoom.plot(diff,color='green')
    raster_plot=np.where(raster==1)
    ax_zoom.scatter(raster_plot[1],raster_plot[0]/trials/2,s=.5,color = 'grey',label = str(round((onset/samplingrate),3))+' [s]')
    ax_zoom.set_xticks(np.arange(0,int((start+end)*samplingrate),samplingrate))
    ax_zoom.set_xticklabels(np.arange(-start,end))
    
    ax.set_xticks(np.arange(int(start*samplingrate),int((start+2)*samplingrate),samplingrate/10))
    ax.set_xticklabels(np.arange(0,start+2,(1/10)))
    ax.set_xlim(onset-.5*samplingrate,onset+.5*samplingrate)
    ax.set_xlabel(r'time [s]')
    
    #ax_zoom.axvline(start*samplingrate,ls='--',color = 'black',lw=0.7)
    #ax.axvline(start*samplingrate,ls='--',color = 'black',lw=0.7)
    ax_zoom.set_xlabel(r'time [s]')
    
    #ax_zoom.axvline(onset,ls='--',color = 'red',lw=0.7,label = str((onset/samplingrate))[:5]+' [s]')
    #ax.axvline(onset,ls='--',color = 'red',lw=0.7,label=str(onset/samplingrate)[:5]+' [s]')
    return ax,ax_zoom

def calculate_bandwidth(times_cut,samplingrate=30000,cv = 40):   
	bandwidths = np.array([10,15,25,50,75,100,150,200,250])*samplingrate/1000
	#bandwidths = np.arange(1,1500,5)*30

	grid = GridSearchCV(KernelDensity(kernel='gaussian'),{'bandwidth': bandwidths},cv=cv,n_jobs=5) # 20-fold cross-validation
	grid.fit(times_cut[:,np.newaxis])

	return grid

def detect_onset_cumsum(cumsum,trials = None,end = None,samplingrate = 30000):
    
    cumsum = cumsum/cumsum[-1]*trials
    ref = np.linspace(cumsum[0],cumsum[-1],cumsum.size)
    diff = cumsum-ref
    onset = np.argmin(diff[:int(end*samplingrate)])

    if onset > int(end*samplingrate*0.95):
    	onset = np.argmin(diff)
                      
    return onset


def build_DF_selection(selection_tmp,baseline_fr,peak_fr,response_time,onset,response_duration,baseline_std):
    
    selection_responsive_tmp=pd.DataFrame([])

    selection_responsive_tmp.at['0','cluster'] = selection_tmp['cluster']
    selection_responsive_tmp.at['0','expID'] = selection_tmp['expID']
    selection_responsive_tmp.at['0','sweeplength'] = selection_tmp['sweeplength']
    selection_responsive_tmp.at['0','basetemp'] = selection_tmp['basetemp']
    selection_responsive_tmp.at['0','sweep_Id'] = selection_tmp['sweep_Id']
    selection_responsive_tmp.at['0','stimtemp'] = selection_tmp['stimtemp']
    selection_responsive_tmp.at['0','duration'] = selection_tmp['duration']
    selection_responsive_tmp.at['0','pre'] = selection_tmp['pre']
    selection_responsive_tmp.at['0','post'] = selection_tmp['post']
    selection_responsive_tmp.at['0','trials'] = selection_tmp['trials']
    selection_responsive_tmp.at['0','date_rec'] = selection_tmp['date_rec']
    selection_responsive_tmp.at['0','animal_id'] = selection_tmp['animal_id']
    selection_responsive_tmp.at['0','structure'] = selection_tmp['structure']

    selection_responsive_tmp.at['0','baselinefr'] = baseline_fr
    selection_responsive_tmp.at['0','baselinestd'] = baseline_std
    selection_responsive_tmp.at['0','peakfr'] = peak_fr
    selection_responsive_tmp.at['0','response_time'] = response_time
    selection_responsive_tmp.at['0','response_onset'] = onset
    selection_responsive_tmp.at['0','response_duration'] = response_duration[2]
    
    return selection_responsive_tmp


def plot_pdf_hist(selection_tmp,times_cut,pdf,bins,start = None,end = None,
                  samplingrate = 30000,trials = None,response_time = None,
                  response_duration=None,ax = None,onset= None):
    
    from get_response_prop import calculate_bandwidth,response_detection_core

    x,y = np.histogram(times_cut,bins)

    rate = x/trials/((bins[1]-bins[0])/samplingrate)
    ax.bar(y[:-1],rate,width=bins[1]-bins[0],color='grey',alpha=0.7,align = 'edge')

    ax1 = ax.twinx()
    ax1.plot(pdf)

    ax.set_xticks(np.arange(0*samplingrate,int(start+end)*samplingrate,samplingrate))
    ax.set_xticklabels(np.arange(-start,end+2))
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax.set_xlabel(r'time s ')
    ax.set_ylabel(r'firingrate hz')
    
    if response_time:
        ax1.scatter((start*samplingrate)+response_time,pdf[(start*samplingrate)+response_time],color = 'red',
                    label = str(round(((response_time/samplingrate)),3))+' [s]')
    
    if response_duration:
        ax1.scatter(response_duration[0],pdf[response_duration[0]],color = 'green',
                    label = str(round(((response_duration[2]/samplingrate)),3))+' [s]')
    
        ax1.scatter(response_duration[1],pdf[response_duration[0]],color = 'green')
    #ax1.scatter[onset,1,label=]	
    response = response_detection_core(pdf,samplingrate=samplingrate)
    if response == 'responsive':   
        title = (str(selection_tmp.expID)+'//'+str(selection_tmp.cluster)+' responsive // base:' + str(selection_tmp.basetemp)+' // stim:' + str(selection_tmp.stimtemp))
    else:
        title = (str(selection_tmp.expID)+'//'+str(selection_tmp.cluster)+' non_responsive // base:' + str(selection_tmp.basetemp)+' // stim:' + str(selection_tmp.stimtemp))

    
    
    
    return ax, ax1, title

def build_pdf(times_cut,start,end,samplingrate=30000,kernel = 'gaussian',bw = None,stepsize=None):
   
    x = np.arange(start*samplingrate,end*samplingrate,stepsize*samplingrate/1000)[:, np.newaxis]
    times_cut = np.sort(times_cut)
    # Gaussian KDE
    kde = KernelDensity(kernel=kernel, bandwidth=bw*30).fit(times_cut[:, np.newaxis])
    log_dens = kde.score_samples(x)
    pdf = np.exp(log_dens)

    return pdf
