import numpy as np
import scipy.signal as signal
from scipy import stats
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import zscore


def detect_UP(train_cut, BIN=5, ratioThreshold=0.4,
              sampling_rate=1., len_state=50.,
              gauss_width_ratio=10., min_for_up=0.2):

    # convolve with Gaussian
    time = range(len(train_cut))  # indexes
    gauss_width = gauss_width_ratio * sampling_rate

    # We obtain a gauss filter
    gauss_filter = np.exp(-0.5 * ((np.subtract(time, len(train_cut) / 2.0) / gauss_width) ** 2))
    gauss_norm = np.sqrt(2 * np.pi * gauss_width ** 2)
    gauss_filter = gauss_filter / gauss_norm

    # We filter the signal by convolving the gauss_filter
    train_filtered = signal.fftconvolve(train_cut, gauss_filter)
    train_filt = train_filtered[int(len(train_cut) / 2.0): \
                                int(3 * len(train_cut) / 2.0)]
    thresh = ratioThreshold * np.max(train_filt)

    # times at which filtered signal crosses threshold
    train_shift = np.subtract(train_filt, thresh) - min_for_up
    idx = np.where(np.multiply(train_shift[1:], train_shift[:-1]) < 0)[0]

    # train of 0 in DOWN state and 1 in UP state
    train_bool = np.zeros(len(train_shift))
    train_bool[train_shift > 0] = 1  # assign to 1 in UP states

    # cut states shorter than min length
    idx = np.concatenate(([0], idx, [len(train_filt)]))
    diff_remove = np.where(np.diff(idx*BIN) < len_state * sampling_rate)[0]
    idx_start_remove = idx[diff_remove]
    idx_end_remove = idx[np.add(diff_remove, 1)] + 1

    for ii_start, ii_end in zip(idx_start_remove, idx_end_remove):
        train_bool[ii_start:ii_end] = np.ones_like(train_bool[ii_start:ii_end]) * train_bool[ii_start - 1]
        # assign to same state as previous long

    idx = np.where(np.diff(train_bool) != 0)[0]
    idx = np.concatenate(([0], idx, [len(train_filt)])) / sampling_rate
    return idx, train_shift, train_bool, thresh


def ordered_train_bools(FR, dt, ratio_threshold=0.3,
                  len_state=20,
                  gauss_width_ratio=10,
                  units='s'):
    """
    return duration UP and DOWN for single node/MF (first state always is DOWN and last states always is UP)
    """
    # print("ratiothresh=", ratio_threshold)
    sampling_rate = 1 / dt
    stats_total = []
    stats_mean_m = []
    mean_down_all = []
    mean_up_all = []
    dur_UP_all = []
    dur_DOWN_all = []

    idx, train_shift, train_bool, _ = detect_UP(FR, ratioThreshold=ratio_threshold,
                                                sampling_rate=sampling_rate,
                                                len_state=len_state,
                                                gauss_width_ratio=gauss_width_ratio)

    durs = np.diff(idx)  # durations of all states

    if train_bool[0] == 0:  # starts with DOWN state
        dur_DOWN = durs[np.arange(0, len(durs), 2)]  # even states are DOWN
        dur_UP = durs[np.arange(1, len(durs), 2)]  # odd states are UP
        if len(durs) % 2 == 1:  # ends with DOWN
            dur_DOWN_l_m = dur_DOWN[:-1]  # discard last DOWN
        else:  # ends with UP
            dur_DOWN_l_m = dur_DOWN
        dur_UP_l_m = dur_UP  # each element has the duration of all the UP states for a region


    else:  # starts with UP state
        dur_UP = durs[np.arange(0, len(durs), 2)]  # even states are UP
        dur_DOWN = durs[np.arange(1, len(durs), 2)]  # odd states are DOWN
        if len(durs) % 2 == 0:  # ends with DOWN
            dur_DOWN_l_m = dur_DOWN[:-1]  # discard last DOWN
        else:
            dur_DOWN_l_m = dur_DOWN
        dur_UP_l_m = dur_UP[1:]  # each element has the duration of all the DOWN states for a region

    if len(np.unique(train_bool)) == 1:  # if there is only one state
        if train_bool[0] == 0:  # if it is a down state
            dur_UP_l_m = [0]
            dur_DOWN_l_m = [5000.]
        elif train_bool[0] == 1:
            dur_UP_l_m = [5000]
            dur_DOWN_l_m = [0.]

    return dur_UP_l_m,dur_DOWN_l_m

def clean_UD_Pear(FR, dt, ratio_threshold=0.3, BIN=5,
                  len_state=20, sampling_rate = 1 / 0.1,
                  gauss_width_ratio=10,
                  units='s'):

    # print("ratiothresh=", ratio_threshold)
#     sampling_rate = 1 / dt
    stats_total = []
    stats_mean_m = []
    mean_down_all = []
    mean_up_all = []
    dur_UP_all = []
    dur_DOWN_all = []

    idx, train_shift, train_bool, _ = detect_UP(FR, ratioThreshold=ratio_threshold, BIN=BIN,
                                                sampling_rate=sampling_rate,
                                                len_state=len_state,
                                                gauss_width_ratio=gauss_width_ratio)

    durs = np.diff(idx)  # durations of all states

    if train_bool[0] == 0:  # starts with DOWN state - all start with down and ends with UP
        dur_DOWN = durs[np.arange(0, len(durs), 2)]  # even states are DOWN
        dur_UP = durs[np.arange(1, len(durs), 2)]  # odd states are UP
        if len(durs) % 2 == 1:  # ends with DOWN
            dur_DOWN_l_m = dur_DOWN[:-1]  # discard last DOWN
        else:  # ends with UP
            dur_DOWN_l_m = dur_DOWN
        dur_UP_l_m = dur_UP  # each element has the duration of all the UP states for a region


    else:  # starts with UP state
        dur_UP = durs[np.arange(0, len(durs), 2)]  # even states are UP
        dur_DOWN = durs[np.arange(1, len(durs), 2)]  # odd states are DOWN
        if len(durs) % 2 == 0:  # ends with DOWN
            dur_DOWN_l_m = dur_DOWN[:-1]  # discard last DOWN
        else:
            dur_DOWN_l_m = dur_DOWN
        dur_UP_l_m = dur_UP[1:]  # each element has the duration of all the DOWN states for a region

    if len(np.unique(train_bool)) == 1:  # if there is only one state
        if train_bool[0] == 0:  # if it is a down state
            dur_UP_l_m = [0]
            dur_DOWN_l_m = [5000.]
        elif train_bool[0] == 1:
            dur_UP_l_m = [5000]
            dur_DOWN_l_m = [0.]

        # print(m)
        # print(dur_UP_l_m)

    mean_down = np.mean(dur_DOWN_l_m)  # mean down duration for each node
    mean_up = np.mean(dur_UP_l_m)

    dur_UP_all.append(dur_UP_l_m)
    dur_DOWN_all.append(dur_DOWN_l_m)
    # try:
    pears_stat_m = stats.pearsonr(dur_DOWN_l_m,
                                  dur_UP_l_m)  # pearson correlation for region : [0]-> P_cof, [1] -> p-value
    print(pears_stat_m)
    stats_total.append(pears_stat_m)  # list of the statistics
    if pears_stat_m[1] < 0.05:
        stats_mean_m.append(pears_stat_m[0])  # all the p_coef (without the p value in order to take the mean)
    else:
        stats_mean_m.append(0)
    # except ValueError:
    #     print("ratio_thresh=", ratio_threshold)
    #     print("dur_DOWN_l_m = ", dur_DOWN_l_m)
    #     print("dur_UP_l_m = ", dur_UP_l_m)

    mean_down_all.append(mean_down)
    mean_up_all.append(mean_up)
    # stats_total.append(pears_stat_m) #list of the statistics
    # stats_mean_m.append(pears_stat_m[0]) # all the p_coef (without the p value in order to take the mean)




    # dur_DOWN/UP_all (list): each element is an array with the durations of UP/DOWN for EACH NODE
    # stats_mean_m (list) : each element is the pcor for each node
    # stats_mean : the mean of pcor across the nodes
    # stats_total : the pcor and the p_val for each node

    return stats_mean_m, dur_UP_all, dur_DOWN_all

def rolling_average(signal, window, mode='valid'):
    kernel= np.ones(window)/window

    smoothed = np.convolve(signal, kernel, mode=mode)
    return smoothed

def savgol(singal, window, poly_order=2):
    window = 5000
    smoothed = savgol_filter(singal, window, poly_order)
    return smoothed

def lowpass(signal, cutoff_frequency, fs, order=2):

    b, a = butter(N=order, Wn=cutoff_frequency/(0.5*fs), btype='low')

    smoothed = filtfilt(b, a,signal)
    return smoothed

def segments_UP_DOWN(vM, idx_up_cor, min_durs, WIN=50, DT=0.1, tit=None):
    # WIN in 0.1*ms -> 50 = 5 ms
    period = int(min(min_durs)/DT)
    print(f"period= {period*DT} ms")
    fig, ax=plt.subplots(2,1)
    vM_all=[]
    for i in range(len(idx_up_cor)):
        ind = idx_up_cor[i]
         # Take the smallest UP or DOWN state
        # ax[0].plot(time[(ind-period):(ind+period)], muvi[(ind-period):(ind+period)])
        ax[0].plot(vM[(ind-period):(ind+period)])
        # ax[0].axvline(x=ind/10000,  color = 'black')
        # ax[0].axvline(x=len(muvi[(ind-period):(ind+period)])/2,  color = 'black')

        window = WIN

        kernel= np.ones(window)/window

        smoothed = np.convolve(vM[(ind-period):(ind+period)], kernel, mode='valid')
        # time_sm = np.linspace(time[(ind-period)],time[(ind+period)],smoothed.shape[0])
        # ax[1].plot(time_sm[1000:7000],smoothed[1000:7000])
        ax[1].plot(smoothed)
        vM_all.append(list(smoothed))
        

    # make all the arrays the same length but first make sure that the different is not too big
    maxlen = max(len(lst) for lst in vM_all)
    minlen = min(len(lst) for lst in vM_all)
    diflen = maxlen - minlen
    print(f"Difference between lengts = {diflen*DT} ms")
    if diflen/period < 0.15: #smaller than 15% of the period
        pass
    else:
        raise Exception(f"Difference of lengths = {diflen*DT} ms")

    #Trim the arrays:
    for i in range(len(vM_all)):
        vM_all[i] = vM_all[i][:minlen]

    # plt.axvline(x=len(smoothed)/2,  color = 'black')
    ax[0].axvline(x=len(vM[(ind-period):(ind+period)])/2,  color = 'black')
    ax[0].set_ylabel('mV')
    ax[0].set_title(f'mV (MF) centered at UP beginning, period= {period*DT} ms \n- {tit}')
    
    smoothed_mean = np.array(vM_all).mean(axis=0)
    ax[1].plot(smoothed_mean,'k', linewidth=2.5)
    ax[1].axvline(x=len(smoothed)/2,  color = 'black')
    ax[1].set_ylabel('mV')
    ax[1].set_title(f"Rolling averaged (window= {WIN*DT} ms)")

    plt.tight_layout()
    del smoothed
    return smoothed_mean

def find_transitions(train_bool):
    transitions = []
    for i in range(len(train_bool) - 1):
        if train_bool[i] == 0 and train_bool[i + 1] == 1:
            transitions.append(i + 1)  # Add 1 to get the index where the transition occurs
    return transitions

def filter_slow_range(FR, time):
    # Define the sampling rate (assuming 999 time points correspond to 5 seconds)
    sampling_rate = FR.shape[0] / (time[-1] - time[0])  # time points per second

    # Compute the frequency resolution
    freq_resolution = sampling_rate / FR.shape[0]

    # Compute the Fourier Transform
    fft_values = np.fft.fft(FR, axis=0)

    # Find the indices corresponding to the frequency range you want to keep
    min_freq_idx = int(0.1 / freq_resolution)
    max_freq_idx = int(4 / freq_resolution)

    # Zero out frequencies outside the desired range
    fft_values[:min_freq_idx, :] = 0
    fft_values[max_freq_idx:, :] = 0

    # Compute the inverse Fourier Transform to get back the filtered signal
    filtered_signal = np.fft.ifft(fft_values, axis=0)

    # The filtered_signal now contains your filtered array

    # You may also want to take the absolute value if you are only interested in magnitude
    filtered_signal_magnitude = np.abs(filtered_signal)

    return filtered_signal_magnitude

def find_delays(FR, time, percent=0.3, filt = True):
    if filt:
        FR = zscore(filter_slow_range(FR, time))
    nodes = FR.shape[1]
    peak_height = []
    for n in range(nodes):
        # find the mean peak amplitude per node 
        peak_h_mean= find_peaks(FR[:, n], height=0,distance = 2500)[1]['peak_heights'].mean()
        peak_height.append(peak_h_mean)

    #threshold height is a percentage of mean peak height, default = 0.3
    thresh = np.mean(peak_height) * percent
    print('thresh = ', thresh)


    
    #Find peak times per node and number of waves per node  
    peak_times = []
    n_peaks = []
    for n in range(nodes):
        peak_t = find_peaks(FR[:, n], height=thresh, distance = 100)[0]
        peak_times.append(peak_t)
        n_peaks.append(len(peak_t)) 

    #Pad the arrays so everyone has the same nodes and you can use then the np.where
    max_length = max(n_peaks)
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=0) for arr in peak_times]
    pad_arr = np.array([padded_arrays])[0]

    #Calculate the mean distance between the peaks and take the half for the window 
    diff = np.diff(pad_arr)
    means = []
    for i in range(nodes):
        diff_filt =np.nanmean(diff[i,:][diff[i,:] > 0]) #exclude the values that are lower than 0 
        means.append(diff_filt)
    mean_diff = np.nanmean(means)
    window = mean_diff/2
    print("window = ", window)
    #Find how many nodes have certain number of oscillations
    for i in sorted(np.unique(n_peaks))[::-1]: #descending order
        n_nodes=np.count_nonzero(n_peaks == i)
        print(i, ':', n_nodes)
        if n_nodes > int(nodes/4): # more than 25% of the nodes
            desired_length = i
            break

    # find the index of a row that has the desired length
    for j in range(len(n_peaks)):
        if n_peaks[j] == desired_length:
            print(j)
            break

    # Store the order of indices of nodes entering the oscillations
    order_of_indices = []
    for tpoint in pad_arr[j, :]:
        if tpoint != 0:  # ignore the zeros that we had padded
            order_of_indices.append(np.where(pad_arr == tpoint)[0][0])

    # center the oscillations in the arrays so that the columns correspond to the same osc time
    # you use the window from before
    row = j
    all_osc = []
    for tpoint in pad_arr[row,:]:
        if tpoint==0: #ignore the zeros that we had padded
            continue
        other_tpoints = pad_arr[np.where((pad_arr<tpoint+window) & (pad_arr>tpoint-window))] #find across nodes
        all_osc.append(other_tpoints)


    # pad also this array with zeros
    max_length = max(len(arr) for arr in all_osc)

    all_osc_pad = [np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=0) for arr in all_osc]
    all_osc_pad = np.array(all_osc_pad).T #has shape nodes,max n of oscillations

    #find the max delay across the nodes 
    delays = []
    for i in range(all_osc_pad.shape[1]):
        del_osc = max(all_osc_pad[:,i])-min(all_osc_pad[:,i][all_osc_pad[:,i] != 0]) #exclude zeros
        delays.append(del_osc)


    return all_osc_pad, delays, np.mean(delays), order_of_indices