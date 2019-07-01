
import numpy as np
import pandas as pd


def readKS_SaveDF (file_location, save_path, samplerate,automated = False):

    """Function to read KS output in Pandas DataFRame

    ---------------------
    input:
    file_location : string
            Main result folder, defined in KS_config.m
    save_path : string
            Location where Pandas DataFrame is saved
    samplerate = int (optional)
            default = 30000

    ---------------
    output:
    DataFrame for good, mua and noise cluster
    """

    def read_KS_output(file_location,automated = False):
        
        spike_samples = np.load(file_location +'/spike_times.npy')
        spike_times = spike_samples/samplerate # This gives you time of Spikes in seconds. THINK ABOUT TO SUBTRACT THE FIRST VALUE!
        spike_cluster = np.load(file_location+'/spike_clusters.npy')
        tempScalingAmps = np.load(file_location+'/amplitudes.npy') # The cluster identity of every spike. same length as spike_times. result of manual sorting
        '''spike_templates = np.load(file_location+'/spike_templates.npy') # template number rather than cluster number. was utilized for spike sorting// template matching. result of automated sorting
                                tempScalingAmps = np.load(file_location+'/amplitudes.npy') # amplitude of template with which spike was extracted using template matching
                                temps = np.load(file_location+'/templates.npy')
                                temps_ind = np.load(file_location+'/templates_ind.npy')
                                winv = np.load(file_location+'/whitening_mat_inv.npy')
                                channel_map = np.load(file_location+'/channel_map.npy')
                                channel_position = np.load(file_location+'/channel_positions.npy')'''
        if automated == False:
            df=pd.read_csv(file_location + '/cluster_KSLabel.tsv',header=0, delim_whitespace=True)
        else:
            df=pd.read_csv(file_location + '/cluster_KSLabel.tsv',header=0, delim_whitespace=True)

        replace = {'mua':1, 'good':2, 'noise':3}
        df_number = df.replace({'mua':1,'good':2,'noise':3})
        mua_cluster = df_number[df_number['KSLabel'] == 1]
        good_cluster = df_number[df_number['KSLabel'] == 2]
        noise_cluster = df_number[df_number['KSLabel'] == 3]

        mua_cluster_array = np.array(mua_cluster['cluster_id'])
        good_cluster_array = np.array(good_cluster['cluster_id'])
        noise_cluster_array = np.array(noise_cluster['cluster_id'])
        print(good_cluster_array.ravel().shape)
        
        def ismember(a, b):
            bind = {}
            for i, elt in enumerate(b):
                if elt not in bind:
                    bind[elt] = i
            #print(bind)
            return [bind.get(int(itm), None) for itm in a]

        #print(good_cluster_array)

        good_spikes_times_bool = ismember(spike_cluster, good_cluster_array)
        good_spikes_times_bool = np.array(good_spikes_times_bool)
        good_check = (good_spikes_times_bool == 1)
        
        good_spike_times = spike_times[good_check].flatten()
        good_spike_samples = spike_samples[good_check].flatten()
        good_spike_cluster = spike_cluster[good_check].flatten()
        good_spike_amps = tempScalingAmps[good_check].flatten()
        
        mua_spikes_times_bool = ismember(spike_cluster, mua_cluster_array)
        mua_spikes_times_bool = np.array(mua_spikes_times_bool)
        mua_check = (mua_spikes_times_bool == 1)
        
        mua_spike_times = spike_times[mua_check].flatten()
        mua_spike_samples = spike_samples[mua_check].flatten()
        mua_spike_cluster = spike_cluster[mua_check].flatten() 
        mua_spike_amps = tempScalingAmps[mua_check].flatten()
        
        noise_spikes_times_bool = ismember(spike_cluster, noise_cluster_array)
        noise_spikes_times_bool = np.array(noise_spikes_times_bool)
        noise_check = (noise_spikes_times_bool == 1)
        
        noise_spike_times = spike_times[noise_check].flatten()
        noise_spike_samples = spike_samples[noise_check].flatten()
        noise_spike_cluster = spike_cluster[noise_check].flatten() 
        noise_spike_amps = tempScalingAmps[noise_check].flatten()
        
        def spearte_spikes(good_cluster_array, spikes,samples,amps, spike_cluster_good):
            cluster_dict = {}
            
            for i in good_cluster_array:
                tmp1 = spikes[spike_cluster_good == i]
                tmp2 = samples[spike_cluster_good == i]
                tmp3 = amps[spike_cluster_good == i]

                tmp_tot = np.zeros([len(tmp1),3])
                tmp_tot[:,0] = tmp1
                tmp_tot[:,1] = tmp2
                tmp_tot[:,2] = tmp3
                
                parameter_dict = {}
                parameter_dict['Spike Times'] = tmp1
                parameter_dict['Spike Samples'] = tmp2
                parameter_dict['Spike Amps'] = tmp3
             
                cluster_dict[str(i)] = parameter_dict

            return cluster_dict



        good_cluster_dict = spearte_spikes(good_cluster_array, good_spike_times,good_spike_samples,good_spike_amps, good_spike_cluster)
        mua_cluster_dict = spearte_spikes(mua_cluster_array, mua_spike_times,mua_spike_samples,mua_spike_amps, mua_spike_cluster)
        noise_cluster_dict = spearte_spikes(noise_cluster_array, noise_spike_times,noise_spike_samples,noise_spike_amps, noise_spike_cluster)

        
        return good_cluster_dict, mua_cluster_dict, noise_cluster_dict




    samplerate = samplerate
    file_location = file_location
    Good_dict, Mua_dict, Noise_dict= read_KS_output(file_location)



    result_dict = {}
    result_dict['Good'] = Good_dict
    result_dict['Mua'] = Mua_dict
    result_dict['Noise'] = Noise_dict



    result_DataFrame  = pd.DataFrame.from_dict({(i,j): result_dict[i][j] 
                               for i in  result_dict.keys()
                                for j in result_dict[i].keys()
                               }, orient = 'columns')



    result_DataFrame.to_pickle('{}/TimesSamplesAmps'.format(save_path))


    return result_DataFrame



