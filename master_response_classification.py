import matplotlib.pyplot as plt
from neo.io import PickleIO
import pandas as pd
import os
wd = '/Users/tobiasleva/Desktop'

from neo_access import *
from get_response_prop import *
import numpy as np

os.chdir(wd)
io = PickleIO(filename="vpl_block.pkl")
blk2 = io.read()[0]
dataframe = pd.read_pickle(os.path.join(wd,'vpl_block_df.pkl'))


clusters = ['1562']

selection = dataframe[(dataframe['basetemp'] == 32.0) &
                      (dataframe['stimtemp'] == 22.0)&
                      (dataframe['cluster'].isin(clusters)) ]

data = get_all_data(blk2,selection,raster=True,timestamps=True,spikes=True,aisignal=True)

trials = 50
samplingrate = 30000
baseline_window = 1
cv = 40
stimulus_start = 1
window = 7500
selection_responsive = pd.DataFrame([])
for unit in range(selection.shape[0])[:1]:

    selection_tmp = selection.iloc[0]
    data_tmp = data[:][0][:]

    start = selection_tmp.pre - 1
    end = selection_tmp.pre + 5

    signal = data_tmp[0]
    raster = np.zeros([trials, int((end - start) * samplingrate)])
    times = (data_tmp[0][:, 0] / samplingrate)
    times_inx = data_tmp[0][:, 1]
    times_cut = ((times[(times > start) & (times < end)] * samplingrate) - start * samplingrate).astype(int)
    times_inx_cut = times_inx[(times > start) & (times < end)].astype(int)
    raster[times_inx_cut, times_cut] = 1
    cumsum = np.sum((np.cumsum(raster, axis=1)), axis=0)

    if times_cut.size < cv:
        continue

    selection_responsive_tmp = pd.DataFrame([])

    pdf, response, fig = plot_PSTH_gaussian(selection_tmp, data_tmp, trials=50, samplingrate=30000,cv=40)

    baseline = np.mean(pdf[:stimulus_start * samplingrate])

    if response == 'responsive':
        selection_responsive_tmp.at['0', 'cluster'] = selection_tmp['cluster']
        selection_responsive_tmp.at['0', 'expID'] = selection_tmp['expID']
        selection_responsive_tmp.at['0', 'sweeplength'] = selection_tmp['sweeplength']
        selection_responsive_tmp.at['0', 'basetemp'] = selection_tmp['basetemp']
        selection_responsive_tmp.at['0', 'sweep_Id'] = selection_tmp['sweep_Id']
        selection_responsive_tmp.at['0', 'stimtemp'] = selection_tmp['stimtemp']
        selection_responsive_tmp.at['0', 'duration'] = selection_tmp['duration']
        selection_responsive_tmp.at['0', 'pre'] = selection_tmp['pre']
        selection_responsive_tmp.at['0', 'post'] = selection_tmp['post']
        selection_responsive_tmp.at['0', 'trials'] = selection_tmp['trials']
        selection_responsive_tmp.at['0', 'date_rec'] = selection_tmp['date_rec']
        selection_responsive_tmp.at['0', 'animal_id'] = selection_tmp['animal_id']
        selection_responsive_tmp.at['0', 'structure'] = selection_tmp['structure']

        selection_responsive_tmp.at['0', 'response'] = response
        selection_responsive_tmp.at['0', 'baselinefr'] = calculate_base_peak_core(raster, selection_tmp,
                                                                                  window=0.3,
                                                                                  baselineend=1,
                                                                                  stimulus_start=1,
                                                                                  samplingrate=30000)[0]
        selection_responsive_tmp.at['0', 'peakfr'] = calculate_base_peak_core(raster, selection_tmp,
                                                                              window=0.3,
                                                                              baselineend=1,
                                                                              stimulus_start=1,
                                                                              samplingrate=30000)[1]
        response_time = response_time_core(pdf, stimulus_start=1, samplingrate=30000)
        selection_responsive_tmp.at['0', 'response_time'] = response_time
        selection_responsive_tmp.at['0', 'response_onset'] = detect_onset(pdf,
                                                                          threshold=baseline + 2 * np.std(pdf[0:int(
                                                                              stimulus_start * samplingrate)]),
                                                                          n_above=window, n_below=30,
                                                                          show=False)[0][0] - (stimulus_start * samplingrate)
        selection_responsive_tmp.at['0', 'response_duration'] = response_duration(pdf, response_time)[2]

        selection_responsive = selection_responsive.append(selection_responsive_tmp)

