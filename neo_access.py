
import numpy as np

def access_raster_core(blk,string):
    string = string.split('_')
    data_trials = blk.segments[int(string[0])].irregularlysampledsignals[int(string[1])].as_array().ravel()
    data_spikes = blk.segments[int(string[0])].irregularlysampledsignals[int(string[1])].times
    
    data = np.zeros([data_spikes.size,2])
    data[:,0] = data_spikes
    data[:,1] = data_trials
    return data

def access_events_core(blk,string):
    string = string.split('_')
    timestamps = blk.segments[int(string[0])].events[int(string[1])].as_array().ravel()
    return timestamps

def access_aisignal_core(blk,string):
    string = string.split('_')
    aisignal = blk.segments[int(string[0])].analogsignals[int(string[1])].as_array()    
    return aisignal

def access_spiketrain_core(blk,string):
    string = string.split('_')
    spikes = blk.segments[int(string[0])].spiketrains[int(string[1])].times    
    return spikes

def get_all_data(blk2,selection,raster=None,timestamps=None,spikes = None, aisignal=None):

    iterations = selection.shape[0]
    
    raster_list = []
    timestamps_list = []
    spikes_list = []
    aisignal_list = []

    for it in range(iterations):   
        strings = list(selection[['raster_loc','event_loc','spiketrain_loc','ai_loc']].iloc[it,:].values)
        
        print(strings)

        if raster:
            raster_tmp = access_raster_core(blk2,strings[0])
            raster_list.append(raster_tmp)
            
        if timestamps:
            timestamps_tmp = access_events_core(blk2,strings[1])
            timestamps_list.append(timestamps_tmp)
            
        if spikes:
            spikes_tmp = access_spiketrain_core(blk2,strings[2])
            spikes_list.append(spikes_tmp)
            
        if aisignal:
            aisignal_tmp = access_aisignal_core(blk2,strings[3])
            aisignal_list.append(aisignal_tmp)
            
    return (raster_list,timestamps_list,spikes_list,aisignal_list)