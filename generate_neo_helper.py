import neo as neo
import numpy as np
import quantities as pq
import pandas as pd


def generate_analogsignal_core(data_tmp, units='C', samplingRate=1000, inx=None):

    ai_signal = neo.AnalogSignal(data_tmp, units=units, sampling_rate=samplingRate*pq.Hz,
                                 name=inx+1)

    return ai_signal


def generate_segment_core(date=None, date_of_rec=None,
                          working_directory=None, exp=None):
    seg = neo.Segment()
    seg.file_datetime = date
    seg.rec_datetime = date_of_rec
    seg.file_origin = working_directory
    seg.name = exp

    return seg


def generate_spiketrain_core(spikes, start=None, end=None, quality='Good', cluster_id=None):

    if start is not None and end is not None:

        spikes_segment = spikes[quality][cluster_id]['Spike Times'][
            ((spikes[quality][cluster_id]['Spike Times']) > start) & (
                        (spikes[quality][cluster_id]['Spike Times']) < end)]
        train = neo.SpikeTrain(spikes_segment, units='s', t_start=start, t_stop=end, name=cluster_id)
        train.annotate(quality=quality)
        return train

    spikes_segment = spikes[quality][cluster_id]['Spike Times']
    train = neo.SpikeTrain(spikes_segment, units='s', t_start=start, t_stop=end, name=cluster_id)
    train.annotate(quality=quality)

    return train


def generate_event_core(data_timestamps, title):
    ttl = neo.Event(data_timestamps * pq.s, labels=[title] * data_timestamps.size)
    ttl.name = title


    return ttl


def generate_block_core(date=None, working_directory=None, date_of_rec=None,
                        recording_location=None, animal_id=None,
                        number_exp=None):
    block_tmp = neo.Block()
    block_tmp.file_datetime = date
    block_tmp.file_origin = working_directory
    block_tmp.rec_datetime = date_of_rec
    block_tmp.name = recording_location
    block_tmp.annotate(animalID=animal_id, exps_tot=str(number_exp))

    return block_tmp


def generate_alignedspikes_core(raster, name=None): #, sweeplength=None, pre=None, post=None,index=None):

    spikes = neo.IrregularlySampledSignal(np.where(raster == 1)[1], np.where(raster == 1)[0],
                                          units='s', time_units='s', name=name,
                                          description='sweep aligned spike times // container_name.signal = sweep_id')

    #spikes.annotate(sweeplength=sweeplength, pre=pre, post=post, index=index)

    return spikes


def raster_build(spikes_tmp, trials, samplingrate=30000, sweeplength=None):
    raster = np.zeros([len(trials), sweeplength * samplingrate])

    for ii in range(len(trials)):
        raster_ind = spikes_tmp[np.where(np.logical_and(spikes_tmp > trials[ii],
                                                        spikes_tmp < (trials[ii] +
                                                                      sweeplength)))] - trials[ii]

        raster_ind = raster_ind * samplingrate
        raster[ii, raster_ind.astype(int)] = 1

    return raster


def generate_segment(data, segment, spike_data, metadata, quality='Good'):

    samplingrate, sweeplength, pre, post, rep, exp_Id, stim_infos = read_sweepparamter(metadata)

    count = 0
    for inx in range(data.shape[0]):
        data_feedback = data.iloc[inx]['Feedback']
        if str(data_feedback) == 'nan':
            continue
        ai_signal = generate_analogsignal_core(data_feedback, inx=inx)
        add_metaData(ai_signal, samplingrate, sweeplength, pre, post, rep, stim_infos[count])
        segment.analogsignals.append(ai_signal)
        count+=1

    count = [0, 0, 1, 1, 2, 2]
    count_inx = 0
    for inx in range(data.shape[0]):
        data_timestamps_array = data.iloc[inx]['Timestamps']
        data_timestamps = data['Timestamps']
        title = list(data_timestamps.index)[inx]

        if str(data_timestamps_array) == 'nan':
            continue
        ttl = generate_event_core(data_timestamps_array, title)
        add_metaData(ttl, samplingrate, sweeplength, pre, post, rep, stim_infos[count[count_inx]])
        segment.events.append(ttl)
        count_inx += 1

    start = segment.events[1].times[0]
    end = segment.events[-1].times[-1]

    cluster = spike_data[quality].keys()

    for clu in range(len(cluster)):
        cluster_id = cluster[clu]
        train = generate_spiketrain_core(spike_data, start=start, end=end, cluster_id=cluster_id)

        segment.spiketrains.append(train)

    for spk in segment.spiketrains:
        spikes = spk.as_array()
        count = 0
        for ii in range(len(segment.events)):

            if 'sweepstart' in segment.events[ii].name:
                ttls = segment.events[ii].as_array()
                raster = raster_build(spikes, ttls, sweeplength=sweeplength)

                alignedSpikes = generate_alignedspikes_core(raster, name=spk.name)
                add_metaData(alignedSpikes, samplingrate, sweeplength, pre, post, rep, stim_infos[count])
                segment.irregularlysampledsignals.append(alignedSpikes)
                count+=1

    return segment


def read_sweepparamter(metadata):
    combi = np.unique(metadata['sweepID'])
    samplingrate = metadata['Samplingrate'][0]
    sweeplength = metadata['Sweeplength'][0]
    pre = metadata['Pre Stimulus Time'][0]
    post = sweeplength - pre
    rep = metadata['Repititions'][0]
    exp_ID = metadata['ID'][0]

    stim_infos = []
    for i in range(len(combi)):
        selection = metadata[metadata['sweepID'] == combi[i]]
        try:
            stim_infos.append(
            {'sweepID': selection['sweepID'].iloc[0], 'stimtemp': selection['Stimulus Temp'].iloc[0],
             'basetemp': selection['Baseline Temp'].iloc[0], 'duration': selection['Stimulus Duration'].iloc[0]})
        except:
            stim_infos.append(
            {'sweepID': selection['sweepID'].iloc[0], 'stimfreq': selection['Stimulus Frequency'].iloc[0],
             'stimamp': selection['Stimulus Amplitude'].iloc[0], 'duration': selection['Stimulus Duration'].iloc[0]})
    return samplingrate, sweeplength, pre, post, rep, exp_ID, stim_infos


def add_metaData(ai_signal, samplingrate, sweeplength, pre, post, rep, stim_infos):
    sweep_id = list(stim_infos.values())[0]
    basetemp = list(stim_infos.values())[2] * 10
    duration = list(stim_infos.values())[3]
    stimtemp = list(stim_infos.values())[1] * 10

    ai_signal.annotate(sweeplength=sweeplength, samplinrate=samplingrate, basetemp=basetemp, sweep_Id=sweep_id,
                       stimtemp=stimtemp, duration=duration, pre=pre, post=post, trials=rep)


def list_files(working_directory):
    import os
    content = os.listdir(working_directory)

    metadatas = []
    stimdatas = []

    for i in range(len(content)):
        # print(content[i])
        if 'sweepParameter' in content[i]:
            metadatas.append(content[i])

        elif 'stimData' in content[i]:
            stimdatas.append(content[i])

    return metadatas, stimdatas


def generate_DF(blk2):
    df = pd.DataFrame([])
    nr_segmente = len(blk2.segments)

    for seg in range(nr_segmente):
        nr_raster = len(blk2.segments[seg].irregularlysampledsignals)
        nr_events = len(blk2.segments[seg].events)
        nr_aisignals = len(blk2.segments[seg].analogsignals)
        nr_spiketrains = len(blk2.segments[seg].spiketrains)

        df_tmp = pd.DataFrame([])

        for r in range(nr_raster)[:]:

            test = blk2.segments[seg].irregularlysampledsignals[r]
            df_tmp.at[str(r), 'cluster'] = test.name
            df_tmp.at[str(r), 'expID'] = blk2.segments[seg].name

            for an in (test.annotations.keys()):
                df_tmp.at[str(r), str(an)] = test.annotations[str(an)]

            df_tmp.at[str(r), 'raster_loc'] = str(seg) + '_' + str(r)

            for e in range(nr_events):
                if (test.annotations == blk2.segments[seg].events[e].annotations) & (
                        'sweepstart' in blk2.segments[seg].events[e].name):
                    df_tmp.at[str(r), 'event_loc'] = str(seg) + '_' + str(e)

            for ai in range(nr_aisignals):
                if test.annotations == blk2.segments[seg].analogsignals[ai].annotations:
                    df_tmp.at[str(r), 'ai_loc'] = str(seg) + '_' + str(ai)

            for sp in range(nr_spiketrains):
                if (test.name == blk2.segments[seg].spiketrains[sp].name):
                    df_tmp.at[str(r), 'spiketrain_loc'] = str(seg) + '_' + str(sp)

            df_tmp.at[str(r), 'date_rec'] = blk2.rec_datetime
            df_tmp.at[str(r), 'animal_id'] = blk2.annotations['animalID']
            df_tmp.at[str(r), 'rawdata_path'] = blk2.file_origin
            df_tmp.at[str(r), 'structure'] = blk2.name

        df = df.append(df_tmp)

    return df