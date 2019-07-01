
import os
import pandas as pd
import datetime
from neo.io import PickleIO
import generate_neo_helper as nh
import ReadSaveKSOutput_v2 as rKS

date_of_rec = '20190628'
animal_id = 'JPO-001823'
recording_location = 'VPL'
rawdata_filename = 'continuous.dat'
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

depth_of_probe = 4671
time_limits = None

working_directory = 'F:/NeuropixelData/2019-06-28_09-27-20/'
#spikes_tot = pd.read_pickle('/Volumes/Untitled/20180807/Exp3/Analysis/spikeData/TimesSamplesAmps')

stimdata_folder = os.path.join(working_directory,'NationalInstruments/')
print(stimdata_folder)
metadatas, stimdatas = nh.list_files(stimdata_folder)


ks_output_folder = os.path.join(working_directory, 'experiment1/recording1/continuous/Neuropix-PXI-100.0')
savefolder = os.path.join(working_directory, 'SpikeSortingResults')
os.chdir(working_directory)

spikes_tot = rKS.readKS_SaveDF(ks_output_folder,savefolder,30000,automated=True)

block = nh.generate_block_core(date=date, working_directory=working_directory, date_of_rec=date_of_rec,
                               recording_location=recording_location, animal_id=animal_id,
                               number_exp=len(metadatas))

for exp in range(len(metadatas)):

    segment = nh.generate_segment_core(date=date, date_of_rec=date_of_rec, working_directory=working_directory,
                                       exp=stimdatas[exp][:4])

    data = pd.read_pickle(os.path.join(stimdata_folder, stimdatas[exp]))
    metadata = pd.read_pickle(os.path.join(stimdata_folder, metadatas[exp]))

    segment = nh.generate_segment(data, segment, spikes_tot, metadata, quality='Good')
    block.segments.append(segment)


io = PickleIO(filename="vpl_block_{}.pkl".format(date_of_rec))
io.write(block)

df = nh.generate_DF(block)
df.to_pickle(os.path.join(savefolder, "vpl_block_df_{}.pkl".format(date_of_rec)))
