
import os
import pandas as pd
import datetime
from neo.io import PickleIO
import generate_neo_helper as nh
import ReadSaveKSOutput_v2 as rKS


working_directory = r"Z:\Neuropixel\Thalamus\SNA-035184\2019-07-18_10-52-01"
date_of_rec = '20190718'
animal_id = 'SNA-035184'
recording_location = 'VPL'
depth_of_probe = 4780




 
if recording_location == 'VPL':
	flag = 0
	probe = 'probe1'
else:
	flag = 2
	probe = 'probe2'
rawdata_filename = 'continuous.dat'
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

time_limits = None




#spikes_tot = pd.read_pickle('/Volumes/Untitled/20180807/Exp3/Analysis/spikeData/TimesSamplesAmps')

stimdata_folder = os.path.join(working_directory,'NationalInstruments/')
print(stimdata_folder)
metadatas, stimdatas = nh.list_files(stimdata_folder)

stimdata_select = []
[stimdata_select.append(i) for i in stimdatas if probe in i]

stimdatas = stimdata_select

print("@@@@@@@@@@@@@@    {}".format(len(stimdatas)))

ks_output_folder = os.path.join(working_directory, 'experiment1/recording1/continuous/Neuropix-PXI-100.{}/Sorting/1'.format(flag))
savefolder = os.path.join(working_directory, 'SpikeSortingResults_Probe_' +recording_location)                      

if not os.path.isdir(savefolder):
		os.makedirs(savefolder)


os.chdir(working_directory)

spikes_tot = rKS.readKS_SaveDF(ks_output_folder,savefolder,30000,automated=True)

block = nh.generate_block_core(date=date, working_directory=working_directory, date_of_rec=date_of_rec,
							   recording_location=recording_location, animal_id=animal_id,
							   number_exp=len(metadatas))

for exp in range(len(metadatas))[:]:

	print(stimdatas[exp][:3])

	segment = nh.generate_segment_core(date=date, date_of_rec=date_of_rec, working_directory=working_directory,
									   exp=stimdatas[exp][:3])

	data = pd.read_pickle(os.path.join(stimdata_folder, stimdatas[exp]))
	metadata = pd.read_pickle(os.path.join(stimdata_folder, metadatas[exp]))

	segment = nh.generate_segment(data, segment, spikes_tot, metadata, quality='Good')
	block.segments.append(segment)


io = PickleIO(filename="{}_block_data_{}.pkl".format(recording_location,date_of_rec))
io.write(block)

df = nh.generate_DF(block)
df.to_pickle(os.path.join(savefolder, "{}_block_df_{}.pkl".format(recording_location,date_of_rec)))
