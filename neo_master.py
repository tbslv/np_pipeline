
import os
import pandas as pd
import datetime
from neo.io import PickleIO
import generate_neo_helper as nh

date_of_rec = '010101'
animal_id = 'JPO-040493'
recording_location = 'VPL'
rawdata_filename = 'experiment1_100-0_0.dat'
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

depth_of_probe = 4782
time_limits = None

working_directory = '/Volumes/Untitled/20180807/Exp3/2018-08-07_17-22-40/'
spikes_tot = pd.read_pickle('/Volumes/Untitled/20180807/Exp3/Analysis/spikeData/TimesSamplesAmps')

metadatas, stimdatas = nh.list_files(working_directory)


ks_output_folder = os.path.join(working_directory, 'FinalSorting')
os.chdir(working_directory)

block = nh.generate_block_core(date=date, working_directory=working_directory, date_of_rec=date_of_rec,
                               recording_location=recording_location, animal_id=animal_id,
                               number_exp=len(metadatas))

for exp in range(len(metadatas)):

    segment = nh.generate_segment_core(date=date, date_of_rec=date_of_rec, working_directory=working_directory,
                                       exp=stimdatas[exp][:4])

    data = pd.read_pickle(os.path.join(working_directory, stimdatas[exp]))
    metadata = pd.read_pickle(os.path.join(working_directory, metadatas[exp]))

    segment = nh.generate_segment(data, segment, spikes_tot, metadata, quality='Good')
    block.segments.append(segment)


io = PickleIO(filename="vpl_block.pkl")
io.write(block)

df = nh.generate_DF(block)
df.to_pickle(os.path.join(working_directory, "vpl_block_df.pkl"))
