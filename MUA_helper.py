import time
import numpy as np
from scipy.signal import butter, filtfilt
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import os
from joblib import Parallel, delayed
import multiprocessing
from time import time
import gc
from scipy import stats


def make_NP_array_from_linear_data(data):
	data_tmp = data.copy()
	data_tmp = data_tmp.reshape(96,4)


	data_np = data_tmp.copy()
	#  1. we put the data in the correct columns
	data_np[:,0] = data_tmp[:,2]
	data_np[:,1] = data_tmp[:,0]
	data_np[:,2] = data_tmp[:,3]
	data_np[:,3] = data_tmp[:,1]

	# % we then repeat the array
	data_np = np.repeat(data_np,2,axis=0)

	#  and add the shift in y
	data_np_with_shift = np.ones([data_np.shape[0]+1,data_np.shape[1]])*-10
	data_np_with_shift[1::,0] = data_np[:,0]
	data_np_with_shift[0:-1,1] = data_np[:,1]
	data_np_with_shift[1::,2] = data_np[:,2]
	data_np_with_shift[0:-1,3] = data_np[:,3]

	# we add a mask
	data_np_with_shift = np.ma.masked_where(data_np_with_shift==-10, data_np_with_shift)

	return data_np_with_shift

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


	
#%%    
def filter_detect(channel, start, sampling_rate, thresh):
	data_filt = butter_bandpass_filter(channel, 300, 3000, sampling_rate, order=2)
	threshold = thresh * np.std(data_filt)
	spiketimes_tmp = basic_peak_detector(data_filt, thresh=threshold, orientation='negative', verbose=False)
	spiketimes_tmp += start
	return spiketimes_tmp  

#%% Butterworth bandpass filter from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = filtfilt(b, a, data)
	return y

#%%
def basic_peak_detector(sig, thresh=-3.5, orientation='both', verbose=False):
	"""
	Spike detection from Tobi
	"""
	if orientation == 'both':
		sig0_neg = sig[0:-2]
		sig1_neg = sig[1:-1]
		sig2_neg = sig[2:]
		peak_ind_neg, = np.nonzero((sig1_neg<=sig0_neg)&(sig1_neg<sig2_neg)&(sig1_neg<thresh))
		peak_ind_neg = np.array([peak_ind_neg])
		
		sig_inv = sig *-1
		sig0_pos = sig_inv[0:-2]
		sig1_pos = sig_inv[1:-1]
		sig2_pos = sig_inv[2:]
		peak_ind_pos, = np.nonzero((sig1_pos<=sig0_pos)&(sig1_pos<sig2_pos)&(sig1_pos<thresh))
		peak_ind_pos = np.array([peak_ind_pos])
		
		peak_ind = np.concatenate((peak_ind_neg.ravel(), peak_ind_pos.ravel()), axis = 0)
		peak_ind = np.sort(peak_ind)
		
		if verbose:
			size = len(sig)
			n = len(peak_ind)
			print('nb peak={}// {}% of datapints over thr'.format(n, n/size*100))
		
		return peak_ind+1
	
	if orientation == 'negative':
		sig0_neg = sig[0:-2]
		sig1_neg = sig[1:-1]
		sig2_neg = sig[2:]
		peak_ind_neg, = np.nonzero((sig1_neg<=sig0_neg)&(sig1_neg<sig2_neg)&(sig1_neg<thresh))
		
		if verbose:
			size = len(sig)
			n = len(peak_ind_neg)
			print('nb peak={}// {}% of datapints over thr'.format(n, n/size*100))
		
		return peak_ind_neg+1
	
	if orientation == 'positive':
		sig_inv = sig *-1
		sig0_pos = sig_inv[0:-2]
		sig1_pos = sig_inv[1:-1]
		sig2_pos = sig_inv[2:]
		peak_ind_pos, = np.nonzero((sig1_pos<=sig0_pos)&(sig1_pos<sig2_pos)&(sig1_pos<thresh))
		
		return peak_ind_pos + 1


def make_NP_array_from_linear_data(data):
	data_tmp = data.copy()
	data_tmp = data_tmp.reshape(96,4)


	data_np = data_tmp.copy()
	#  1. we put the data in the correct columns
	data_np[:,0] = data_tmp[:,2]
	data_np[:,1] = data_tmp[:,0]
	data_np[:,2] = data_tmp[:,3]
	data_np[:,3] = data_tmp[:,1]

	# % we then repeat the array
	data_np = np.repeat(data_np,2,axis=0)

	#  and add the shift in y
	data_np_with_shift = np.ones([data_np.shape[0]+1,data_np.shape[1]])*-10
	data_np_with_shift[1::,0] = data_np[:,0]
	data_np_with_shift[0:-1,1] = data_np[:,1]
	data_np_with_shift[1::,2] = data_np[:,2]
	data_np_with_shift[0:-1,3] = data_np[:,3]

	# we add a mask
	data_np_with_shift = np.ma.masked_where(data_np_with_shift==-10, data_np_with_shift)

	return data_np_with_shift

def psth_perframe(data,fstarts,n_ch,window): 
	psths_tmp = np.zeros(n_ch)
	for ch_i in range(n_ch):

		spiketimes = data[ch_i]
		psths_tmp[ch_i] = spiketimes[(spiketimes>fstarts)&(spiketimes<int(fstarts+window))].size
	
	
	return psths_tmp

def get_prepare_PSTHs(folder_path,stim_data_path,stimulus,n_ch=384,start=4,end=16,resolution = 0.01,windowsize=0.35,
			   baselinestart = 2,baselineend= 4,sampling_rate=30000):
	
	files = os.listdir(folder_path)
	files = files[2:]
	#with open(stim_data_path, 'rb') as f:
	#   x = pickle.load(f)
	
	#feedback = x["Feedback"][stimulus]
	
	all_trials = {}
	for ch in range(n_ch):
		all_trials[str(ch)]= np.array([])
	print('dict created....')

	for trial in range(len(files)):
		test0 = np.load(os.path.join(folder_path,files[trial]))
		for ch in range(test0.shape[0]):
			all_trials[str(ch)] = np.sort(np.hstack((all_trials[str(ch)],test0[ch])))
			
	print('dict filled....')
	
	all_spikes = []
	for ch2 in range(n_ch):
		all_spikes.append(np.array(all_trials[str(ch2)]))
		
	all_spikes = np.array(all_spikes)
	
	print('array created....')
	
	start = start *sampling_rate
	end = end * sampling_rate
	framestarts = np.arange(start,end,resolution*sampling_rate)
	window = windowsize*sampling_rate
	number_frames = int((end-start)/(resolution*sampling_rate))
	psths=np.zeros([n_ch,number_frames])
	
	
	for frame in range(number_frames):
		#print(frame)
		psths[:,frame] = psth_perframe(all_spikes,framestarts[frame],n_ch,window)
	print('psths created....')
	
	
	baselinestart = int(baselinestart/resolution)
	baselineend = int(baselineend/resolution)
	
	psths_corrected = psths*0
	for i in range(psths.shape[0]):
		try:
			baseline = psths[i,baselinestart:baselineend]
			psths_corrected[i,:] = (psths[i,:] - np.mean(baseline))/np.std(baseline)
		except: 
			psths_corrected[i,:] = 0
	
	
	norm=np.max(psths_corrected)
	print('psths normalized....')

	return psths_corrected, norm, all_trials,framestarts,psths,all_spikes



def plot_save(psths_corrected_cold,psths_corrected_warm,norm_cold,count,framestarts,feedback_cold,feedback_warm,savepath,start=4,end=16,window=0.35):
	
	# takes sliced array as input due to multiprocessing
	
	psths_norm_cold = psths_corrected_cold/norm_cold
	psths_shift_cold = make_NP_array_from_linear_data(psths_norm_cold)
		
	psths_norm_warm = psths_corrected_warm/norm_cold
	psths_shift_warm = make_NP_array_from_linear_data(psths_norm_warm)
		
		
		
	fig = plt.figure(figsize=(12,35))

	ax1 = placeAxesOnGrid(fig,xspan=[0.16,0.33],yspan=[0.08,1])
	ax2 = placeAxesOnGrid(fig,xspan=[0,0.5],yspan=[0,0.05])
		
	ax3 = placeAxesOnGrid(fig,xspan=[0.66,0.82],yspan=[0.08,1])
	ax4 = placeAxesOnGrid(fig,xspan=[0.5,1],yspan=[0,0.05])
	
		
	cleanAxes(ax2, total=True)
	cleanAxes(ax4, total=True)
		
	ax1.pcolormesh(psths_shift_cold,vmin=-0.2,vmax=0.7)
	ax1.set_ylim([0,96])
		#ax1.set_clim([0,1])
			#plt.colorbar()
	ax1.set_xticks([])
	ax1.set_yticks([])

	ax2.plot(np.mean(np.mean(feedback_cold,axis=1),axis=1)[int(start*1000):int(end*1000)],lw=8,color='blue')
	ax2.axvspan(framestarts/30-int(start*1000),framestarts/30+int(window*1000)-int(start*1000),alpha=0.5,color='blue')
	ax2.set_ylim(2.1,4.3)

	ax3.pcolormesh(psths_shift_warm,vmin=-0.2,vmax=0.7)
	ax3.set_ylim([0,96])
		#ax1.set_clim([0,1])
			#plt.colorbar()
	ax3.set_xticks([])
	ax3.set_yticks([])

	ax4.plot(np.mean(np.mean(feedback_warm,axis=1),axis=1)[int(start*1000):int(end*1000)],lw=8,color='red')
	ax4.axvspan(framestarts/30-int(start*1000),framestarts/30+int(window*1000)-int(start*1000),alpha=0.5,color='red')
	ax4.set_ylim(2.1,4.3)
		
	plt.savefig(savepath.format(count))
	plt.close()
		
	return

def plot_save_single(psths_corrected,vmax,count,framestarts,feedback,savepath,border,timepoint,start=0,end=22,window=0.35):
	
	# takes sliced array as input due to multiprocessing
	
	psths_shift = make_NP_array_from_linear_data(psths_corrected)
		

		
		
		
	fig = plt.figure(figsize=(12,35))

	ax1 = placeAxesOnGrid(fig,xspan=[0.16,0.33],yspan=[0.08,1])
	ax2 = placeAxesOnGrid(fig,xspan=[0,0.5],yspan=[0,0.05])
		
	
	
		
	cleanAxes(ax2, total=True)
		
	ax1.pcolormesh(psths_shift,vmax = vmax)
	ax1.axhline(border, ls='--', color = 'red', lw= 9)
	#ax1.set_ylim([0,96])
		#ax1.set_clim([0,1])
			#plt.colorbar()
	ax1.set_xlabel('{:.3f} s'.format(timepoint),fontsize=50)
	ax1.set_yticks([border-10])
	ax1.set_xticks([])
	ax1.set_xticklabels([])
	ax1.set_yticklabels(['THALAMUS'],rotation=90,fontsize=70)

	print(feedback.shape)
	feedbacktrace = (np.mean(np.mean(feedback[:],axis=1),axis=1))
	feedback_shifted = feedbacktrace
	#feedback_shifted = feedbacktrace*0
	#feedback_shifted[:9000] = feedbacktrace[-9000:]
	#feedback_shifted[9000:] = feedbacktrace[:11000]
	feedback_shifted = (feedback_shifted - 0.2926) * 17.0898


	ax2.plot(feedback_shifted,lw=8,color='grey')
	ax2.axvspan(framestarts/30,framestarts/30+int(window*1000),alpha=0.5,color='grey')
	ax2.set_ylim(21,43)


		
	plt.savefig(savepath.format(count))
	plt.close()
		
	return