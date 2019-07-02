from helper import sorting_quality as sq

import seaborn as sns
sns.set_style("white")
import numpy as np
import seaborn as sns
import pandas as pd
import os, csv, time

import matplotlib.pyplot as plt

# ================================================ #
date_of_rec = '20190628'
animal_id = 'JPO-001823'
recording_location = 'VPL'
working_directory = 'F:/NeuropixelData/2019-06-28_09-27-20/'
stimdata_folder = os.path.join(working_directory,'NationalInstruments/')
path_to_stimttls = os.path.join(stimdata_folder,'001_stimData.pkl')
depth_of_probe = 4671


os.chdir(working_directory)

ks_output_folder = os.path.join(working_directory, 'experiment1/recording1/continuous/Neuropix-PXI-100.0')
time_limits = None
rawdata_filename = 'continuous.dat'

# ================================================ #
cluster_summary_tmp = pd.DataFrame([])


quality = sq.masked_cluster_quality(ks_output_folder, time_limits=None, n_fet=3, minimum_number_of_spikes=10)

cluster_summary_tmp['cluster_id'] = quality[0]
cluster_summary_tmp['isolation_distance'] = quality[1]
cluster_summary_tmp['mahalanobis_contamination'] = np.ones(len(quality[2]))-quality[2]
cluster_summary_tmp['flda'] = quality[3]*-1

cluster_groups = sq.read_cluster_groups_CSV(ks_output_folder)
cluster_group = []
color = []
for clu_id in quality[0]:
    if clu_id in cluster_groups[0]:
        cluster_group.append('good')
        color.append(sns.color_palette()[1])
    else:
        if clu_id in cluster_groups[1]:
            cluster_group.append('mua')
            color.append(sns.color_palette()[0])
        else:
            if clu_id in cluster_groups[2]:
                cluster_group.append('unsorted')
                color.append(sns.color_palette()[0])
            else:
                cluster_group.append('noise')
                color.append(sns.color_palette()[0])

cluster_summary_tmp['cluster_groups'] = cluster_group

isiV, Nums = sq.isiViolations(ks_output_folder, time_limits)
cluster_summary_tmp['isi_violation rate'] = isiV
cluster_summary_tmp['isi_violations'] = Nums

sn_peak, sn_mean, data, mean_waveforms = sq.cluster_signalToNoise(ks_output_folder, time_limits=None, filename=rawdata_filename)

cluster_summary_tmp['sn_peak'] = sn_peak
cluster_summary_tmp['sn_mean'] = sn_mean

# ================================================ #

