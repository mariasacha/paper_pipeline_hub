import numpy as np
import scipy.signal as signal
from scipy import stats
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import zscore
import pandas as pd
import seaborn as sns
import scikit_posthocs as sp
import ptitprince as pt
import matplotlib.colors as mplcol

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


def find_transitions(train_bool):
    transitions = []
    for i in range(len(train_bool) - 1):
        if train_bool[i] == 0 and train_bool[i + 1] == 1:
            transitions.append(i + 1)  # Add 1 to get the index where the transition occurs
    return transitions

def box_and_whisker(data, palette,medianprops,meanpprops, axes, COLOR= None, widths = 0.6, ANNOT=False):
    """
    This is for the normal boxplot:
    data: list of PCI values - usually for one stimulus and multiple conditions
    meadian and mean props for the boxplots : dictionaries
    COLOR: if given (string) all the boxes will have this color
    ANNOT: if True the sample size will be displayed
    """

    bp = axes.boxplot(data, widths=widths, medianprops=medianprops,
                      meanprops=meanpprops,showmeans=True, meanline=False,patch_artist=True)
    
    # Change the colour of the boxes to Seaborn's 'pastel' palette
    if COLOR:
        
        for patch in bp['boxes']:
            patch.set_facecolor(COLOR)
    else:
        colors = sns.color_palette(palette)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    # Colour of the median lines
#     plt.setp(bp['medians'], color='k')
    
    
    Post_hoc = sp.posthoc_conover(data, p_adjust='holm')
    
#     # Get the shape of the Post_hoc results (assuming it's a square matrix)
    num_groups = len(Post_hoc)

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = c[0] - 1
        data2 = c[1] - 1
        p_value = Post_hoc.iloc[data1, data2]
        # Significance
#         U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p_value < 0.05:
            significant_combinations.append([c, p_value])

    # Get info about y-axis
    bottom, top = axes.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        axes.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        axes.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = axes.get_ylim()
    yrange = top - bottom
    axes.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    if ANNOT:
        for i, dataset in enumerate(data):
            sample_size = len(dataset)
            axes.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='small')

def custom_sort(item):
    """
    Sort according to absolute difference
    Used in stats_rain so that the longer stat bars will be higher 
    """
    return abs(item[0][0] - item[0][1])

def stats_rain(df, ax, val_col='PCI', group_col='cond'):
    """
    Adds statistical annotation in the boxplot of the raincloud plot
    It runs post hoc conover tests

    df : the dataset, make sure that it refers to one stim 
    val_col : the column with the values to be analysed (PCI)
    group_col : for which conditions (cond)
    """

    Post_hoc = sp.posthoc_conover(df, val_col=val_col, group_col=group_col, p_adjust='holm')
    num_groups = len(Post_hoc)

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(pd.unique(df[group_col])) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = c[0] - 1
        data2 = c[1] - 1
        p_value = Post_hoc.iloc[data1, data2]
        # Significance
    #         U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p_value < 0.05:
            significant_combinations.append([c, p_value])

    # The df is not ordered in the same way as the boxplots, so we create a dictionary
    # Maybe this needs adjustments 
    dict_post_corr = {'wake':0, 'nmda':1, 'gaba':2, 'sleep':3} # the way boxplots are plotted
    dict_wrong = {index+1:elem  for index, elem in enumerate(list(Post_hoc.columns))} #how the df is ordered
    
    #Replace df ordering in the combinations with the desired ones
    new_combs = []
    for comb in significant_combinations:
        tup = comb[0]
        new_tup = [dict_post_corr[dict_wrong[i]] for i in tup]
        new_tup = sorted(new_tup)
        new_combs.append([new_tup, comb[1]])
    
    # Sort according to distance between the boxes
    new_combs = sorted(new_combs, key=custom_sort, reverse=True)

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    boxes = ax.get_lines() 
    wanted_lines = []
    # Get the coordinates of the boxes
    for box in boxes:
        x1 = box.get_xdata()
        if x1[0] == x1[1]: # Only the ones that the x coincides - they are the whiskers in the boxplots
            if not any(np.array_equal(x1, arr) for arr in wanted_lines): #do not save doubles
                wanted_lines.append(x1)

    # That was to check in what order they are plotted the boxes    
    # colors = ['red', 'blue', 'green', 'yellow']
    # i= 0
    # for x2 in wanted_lines:
    #     print(x2)
    #     ax.axvline(x2[0], color=colors[i])
    #     i +=1


    # Significance bars
    for i, new_comb in enumerate(new_combs): #new_comb = [[x1, x2], p]
        # Columns corresponding to the datasets of interest
        x1 = float(wanted_lines[new_comb[0][0]][0])
        x2 = float(wanted_lines[new_comb[0][1]][0])
        
        # What level is this bar among the bars above the plot?
        level = len(new_combs) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        ax.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = new_comb[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        ax.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')
        
    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    return ax

def box_and_whisker_2(data, PCI_all,color_box, axes, zorder=100, widths = 0.08, ANNOT=False):
    """
    This is for the plot with the seeds with the lines:
    data: the df columns of the conditions with the pci values
    PCI_all : the list with the pci values (for the calculation of the statistics
            it is better to use for one stimulus)
    color_box: color of the box plots
    z_order : order that the boxes will be plotted, put a high value to be on top of the lines
    ANNOT: if True the sample size will be displayed
    """

    snsFig = sns.boxplot(data, showfliers=False, whis=0, \
        width=widths, ax=ax, medianprops={"color": color_box})
    for i,box in enumerate([p for p in snsFig.patches if not p.get_label()]): 
        # color = box.get_facecolor()
        box.set_edgecolor(color_box)
        box.set_facecolor((0, 0, 0, 0))
        box.set_zorder(zorder)

    
    Post_hoc = sp.posthoc_conover(PCI_all, p_adjust='holm')
    
#     # Get the shape of the Post_hoc results (assuming it's a square matrix)
    num_groups = len(Post_hoc)

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(PCI_all) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = c[0] - 1
        data2 = c[1] - 1
        p_value = Post_hoc.iloc[data1, data2]
        # Significance
#         U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p_value < 0.05:
            significant_combinations.append([c, p_value])

    # Get info about y-axis
    bottom, top = axes.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]-1
        x2 = significant_combination[0][1]-1
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        axes.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        axes.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = axes.get_ylim()
    yrange = top - bottom
    axes.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    if ANNOT:
        for i, dataset in enumerate(data):
            sample_size = len(dataset)
            axes.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='small')

def violin_and_whisker(data, palette,medianprops,meanpprops, axes, COLOR= None, widths = 0.6, ANNOT=False):

#     ax = plt.axes()
#     bp = axes.violinplot(data, widths=widths,showmeans=True)
    
    bp= sns.violinplot(data = data, width = widths, saturation = 0.5, 
                           color = COLOR, alpha = 0.06, ax=axes)
    
    
    Post_hoc = sp.posthoc_conover(data, p_adjust='holm')
    
#     # Get the shape of the Post_hoc results (assuming it's a square matrix)
    num_groups = len(Post_hoc)

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = c[0] - 1
        data2 = c[1] - 1
        p_value = Post_hoc.iloc[data1, data2]
        # Significance
#         U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p_value < 0.05:
            significant_combinations.append([c, p_value])

    # Get info about y-axis
    bottom, top = axes.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]-1
        x2 = significant_combination[0][1]-1
        
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        axes.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        axes.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    # Adjust y-axis
    bottom, top = axes.get_ylim()
    yrange = top - bottom
    axes.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    if ANNOT:
        for i, dataset in enumerate(data):
            sample_size = len(dataset)
            axes.text(i , bottom, fr'n = {sample_size}', ha='center', size='x-small')

#     plt.show()

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

def calc_statistics(PCI_states, ax):
    y_min, y_max = ax.get_ylim()
    Post_hoc=sp.posthoc_conover(PCI_states, p_adjust = 'holm') # the stat for the first

    alpha = 0.05  # Significance level
    xtic = np.arange(1,5)
    y_max =max(max(PCI_states[0]), max(PCI_states[1]),max(PCI_states[2]),max(PCI_states[3]))
    for j in range(1,5): # iterate nodes
        posthoc = Post_hoc[j]
        grouptaub80 = PCI_states[j-1]
        for i in range(j+1,5):
            grouptaue = PCI_states[i-1]
            
            
            p_value = posthoc[i]
            ns = convert_pvalue_to_asterisks(p_value)
            if p_value < alpha:
                print("significant")
                ax.plot([xtic[j-1], xtic[i-1]], [y_max + ((j-1)*0.01)+((i-1)*0.02)+(0.1), y_max + ((j-1)*0.01)+((i-1)*0.02)+(0.1)], linewidth=1.5, color='black')
                ax.text((xtic[j-1]+ xtic[i-1])/2,( y_max + ((j-1)*0.01)+((i-1)*0.02)+(0.1)), ns, fontsize=12, ha='center')
    ax.relim()
    ax.autoscale_view()

def load_pci_results(cond, ampstim, b_e, Iexts, tau, E_L_e, E_L_i, i_trials, n_trials, local_folder=False):
    """
    local_folder: True if you want to take the lionel files from the local folder in the git repository (or also if you work from laptop)
    """
    if cond == 'tau_e':
        ffolder_root = './result_tau_e//evoked/eli_-65_ele_-64/Iext_0.315' 
        string = '_tau_e_'
    elif cond == 'tau_i':
        ffolder_root = './result_tau_i//evoked/eli_-65_ele_-64/Iext_0.315' 
        string = '_tau_i_'
    elif cond == 'sleep':
        ffolder_root = './result_sleep_extra//evoked/eli_-65_ele_-64/Iext_0.315' 
        string = '_tau_e_'


    folder_path =  ffolder_root + '/' + \
                'stim_'+str(ampstim)+'_b_e_'+str(b_e)\
                + '_Iext_'+str(Iexts[0])+ string + str(tau)\
                +"_El_e_" + str(E_L_e)+ "_El_i_" + str(E_L_i)\
                +'_seed_'+str(i_trials*n_trials+4)+'/'
    
            
    file_name = folder_path + 'LionelJune2020_Params_PCI_bE_' + str(b_e) + '_stim_'+ str(ampstim) \
    + string + str(tau)+'_trial_'+ str(i_trials)+ "_pers_stims" + '.npy'

    if local_folder:
        ffolder_root = './TVB/pers_stim/Lionel/' 
        file_name = ffolder_root + 'LionelJune2020_Params_PCI_bE_' + str(b_e) + '_stim_'+ str(ampstim) \
    + string + str(tau)+'_trial_'+ str(i_trials)+ "_pers_stims" + '.npy'

    try:
        data_curr = np.load(file_name, encoding = 'latin1', allow_pickle = True).item()
    except FileNotFoundError:
        print("not existing file: ", file_name)
        pass

    return data_curr

def load_pci_results_pipeline(parameters, i_trials, n_trials, stimval=1e-3, b_e=5, Iext=0.000315, tau_e=5.0, tau_i=5.0, local_folder=False):
    """
    local_folder: True if you want to take the lionel files from the local folder in the git repository (or also if you work from laptop)
    """
    if local_folder:
        ffolder_root = './TVB/pers_stim/Lionel/' 
        string='_tau_e_'
        file_name = ffolder_root + f'LionelJune2020_Params_PCI_bE_{b_e}_stim_{stimval}{string}{tau_e}_trial_{i_trials}_pers_stims.npy'
        try:
            data_curr = np.load(file_name, encoding = 'latin1', allow_pickle = True).item()
        except FileNotFoundError:
            try:
                # print("trying tau_i")
                string='_tau_i_'
                file_name = ffolder_root + f'LionelJune2020_Params_PCI_bE_{b_e}_stim_{stimval}{string}{tau_e}_trial_{i_trials}_pers_stims.npy'
                data_curr = np.load(file_name, encoding = 'latin1', allow_pickle = True).item()
            except FileNotFoundError:    
                print("not existing file: ", file_name)
                pass
    
    else:            
        sim_name =  f"stim_{stimval}_b_e_{b_e}_tau_e_{tau_e}_tau_i_{tau_i}_Iext_{Iext}_El_e_{parameters.parameter_model['E_L_e']}_El_i_{parameters.parameter_model['E_L_i']}_nseed_{i_trials*n_trials+(n_trials-1)}"
        folder_path = './result/evoked/' + sim_name+'/'
        file_name = folder_path + f'Params_PCI_bE_{b_e}_stim_{stimval}_tau_e_{tau_e}_tau_i_{tau_i}_trial_{i_trials}.npy'

        try:
            data_curr = np.load(file_name, encoding = 'latin1', allow_pickle = True).item()
        except FileNotFoundError:
            print("not existing file: ", file_name)
            pass

    return data_curr

def make_pci_dict(i_trials, n_trials,data_curr):
    seeds_arr  = (i_trials*n_trials)+np.arange(0,5,1)

    if i_trials==0:
        result_dict = {index: item for index, item in zip(seeds_arr, data_curr['PCI'][-n_trials:])}
    else:
        new_dict = {index: item for index, item in zip(seeds_arr, data_curr['PCI'][-n_trials:])}
        result_dict.update(new_dict)
    
    return result_dict

def create_PCI_all(parameters, params, n_trials=5, stimvals = [1e-5, 1e-4, 1e-3],local_folder=False):
    """
    it creates a list with the PCI values

    DICT = False #True if you also want to create the dictionary for the plot with the lines and the seed
    local_folder = True #True if you want the files from './TVB/pers_stim/Lionel/' 

    """ 
 
    PCI_all=[]
    # dict_all = []

    for ampstim in stimvals:
        PCI_states = []
        # dict_PCI = []
        for b_e, tau_e, Nseeds in params: 
            PCI_curr = []
            for i_trials in range(int(Nseeds/n_trials)):
                data_curr = load_pci_results_pipeline(parameters, i_trials, n_trials, stimval=ampstim, b_e=b_e, tau_e=tau_e, local_folder=local_folder)    
                PCI_curr.append(data_curr['PCI'][-n_trials:])          
                # print(i_trials, len(data_curr['PCI']))
            # print("append wake")
            PCI_states.append(np.concatenate(PCI_curr))
        PCI_all.append(PCI_states)
    
    return PCI_all

def create_dataset_for_raincloud(PCI_all, stimvals, conditions= ['wake', 'nmda', 'gaba', 'sleep'] ):
    
    size_seeds = len(PCI_all[0][0])
    cond_count = len(conditions)
    
    condition_arrays = {condition: np.full(size_seeds, condition) for condition in conditions}
    cond_arr = np.concatenate([condition_arrays[condition] for condition in conditions])
    # print(cond_arr.shape)
    seeds = np.arange(0,size_seeds,1)
    seed_arr = np.tile(seeds, cond_count)

    final_arr = []
    for i in range(len(PCI_all)):
        pci_arr = np.hstack(PCI_all[i])
        # print(pci_arr.shape)
        stim_arr = np.full(pci_arr.shape[0], stimvals[i])
        # print(stim_arr.shape)
        all_arr = np.vstack((pci_arr, cond_arr, stim_arr, seed_arr))
        final_arr.append(all_arr)

    final_array = np.hstack(final_arr)

    df=pd.DataFrame(final_array.T, columns = ["PCI", "cond", "stim", "seed"])
    df['PCI'] = df['PCI'].map(lambda x: float(x))
    df['stim'] = df['stim'].map(lambda x: float(x)*1e3)
    df['seed'] = df['seed'].map(lambda x: int(x))

    df_small=df
    df_small.head()

    return df_small

def plot_raincloud_with_stats(parameters, params, n_trials=5, stimvals=[1e-3], pick_stim=1, conditions= ['wake', 'nmda', 'gaba', 'sleep'],  
                              colors = [ "steelblue", '#6d0a26', '#9b6369', '#c0b3b4'],
                              dx='stim', dy='PCI', dhue='cond', ort='v', sigma=0.2, local_folder=False):
    """
    df
    pick_stim: the stimulus for which you will compare
    colors should be at least as many as the conditions
    dx, dy, dhue, ort, sigma: parameters for the raincloud
    """
    if local_folder:
        print("Loading paper params:")
        params = [[5, 5.0, 60], [30, 3.75, 60], [30, 7.0, 60], [120, 5.0, 60]] # b_e, tau, nseeds
        conditions = ['wake', 'nmda', 'gaba', 'sleep'] #conditions that the params describe
        stimvals = [1e-5, 1e-4, 1e-3] #stimvals to load
        n_trials=5
        for i in range(len(conditions)):
            print(f"For {conditions[i]} : b_e={params[i][0]}, tau={params[i][1]}")
        print(f"Seeds = {params[0][2]}, n_trials={n_trials}, stimvals={stimvals}")

    print("Creating PCI_all")
    PCI_all = create_PCI_all(parameters, params, n_trials=n_trials,stimvals = stimvals, local_folder=local_folder)

    print("Creating dataframe")
    df = create_dataset_for_raincloud(PCI_all, stimvals = stimvals, conditions= conditions)

    #Check if there are more than one pick_stims:
    if type(pick_stim) is list and len(pick_stim)>1: 
        df_use = df[df['stim'].isin(pick_stim)]
        if len(np.unique(df_use['stim']))>1:
            plot_all_stimuli(df_use, sigma)
        else:
            print(f"Only stim = {np.unique(df_use['stim'])} exists for these parameters\nTry again with the correct value in pick_stim")
    else:
        if type(pick_stim) is list:
            pick_stim = pick_stim[0]
        df_use = df[df['stim']==pick_stim]
        # df_use = df
        if colors and len(colors)<=len(conditions):
            pal = sns.color_palette(colors)
        else:
            pal = sns.color_palette('tab20')
        
        print("Plotting..")
        f, ax = plt.subplots(figsize=(5,4))
        # pal = 'Set2'
        ax = pt.RainCloud(x = dx, y = dy, hue = dhue, data = df_use,
            palette = pal, bw = sigma, width_viol = 0.6, width_box = .25, ax = ax,linewidth = 0.8, point_size = 3, point_estimator= np.median,legend_title = 'Stimulus(Hz)',
            orient = ort , dodge=True, pointplot = False,  cloud_alpha=0.3, alpha = 0.6,offset = 0.2, box_showfliers=False)

        for patch in ax.patches:
            fc = patch.get_facecolor()
            patch.set_facecolor(mplcol.to_rgba(fc, 0.4))

        ax =stats_rain(df, ax, val_col='PCI', group_col='cond')

        ax.invert_xaxis()
    plt.show()

def plot_all_stimuli(df, sigma):
    colors = ["tan", "darkred", "steelblue"]
    pal = sns.color_palette(colors)

    dx='cond'; dy='PCI'; dhue='stim'; ort='v'
    f, ax = plt.subplots(figsize=(10,4))
    # pal = 'Set2'
    ax = pt.RainCloud(x = dx, y = dy, hue = dhue, data = df,
        palette = pal, bw = sigma, width_viol = 0.6, width_box = .2, ax = ax,linewidth = 0.8, 
        point_size = 3, point_estimator= np.median,legend_title = 'Stimulus(Hz)',
        orient = ort , dodge=True, pointplot = True,  cloud_alpha=0.3, alpha = 0.6,offset = 0.15, box_showfliers=False)

    for patch in ax.patches:
        fc = patch.get_facecolor()
        patch.set_facecolor(mplcol.to_rgba(fc, 0.4))

    # custom_labels =   ["Wakefulness", "$NREM$ sleep", "$NMDA$-blockers", "$GABA_{A}$-potentiators"]

    # plt.xticks(range(4), custom_labels, fontsize=12)
    plt.ylabel("PCI",fontsize=15)
    plt.xlabel("")

    plt.tight_layout()

def load_params_local_folder():
    params = [[5, 5.0, 60], [30, 3.75, 60], [30, 7.0, 60], [120, 5.0, 60]] # b_e, tau, nseeds
    conditions = ['wake', 'nmda', 'gaba', 'sleep'] #conditions that the params describe
    pick_stim = 1 #what stimulus to plot
    stimvals = [1e-5, 1e-4, 1e-3] #stimvals to load
    n_trials=5