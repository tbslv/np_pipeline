from helper import sorting_quality as sq

import seaborn as sns
sns.set_style("white")
import numpy as np
import seaborn as sns
import pandas as pd
import os, csv, time

import matplotlib.pyplot as plt

# ================================================ #
date_of_rec = '010101'
animal_id = 'JPO-040493'
recording_location = 'VPL'
path_to_stimttls = '/Volumes/Untitled/Neuropixel/180816/Exp1/A001_stimData.pkl'
depth_of_probe = 4782

working_directory = '/Users/tobiasleva/test'
os.chdir(working_directory)

ks_output_folder = '/Volumes/Untitled/Neuropixel/180816/Exp1/2018-08-16_13-19-34/Sorting_4/'
time_limits = None
rawdata_filename = 'experiment1_100-0_0.dat'

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

