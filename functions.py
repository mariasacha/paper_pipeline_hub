from brian2 import *
import os
import scipy.signal as signal
from numba import njit
import itertools
import json
import tvb_model_reference.src.nuu_tools_simulation_human as tools
import pci_v2
import bitarray
import numpy.random as rgn
import tvb.simulator.lab as lab
from IPython.display import clear_output
import builtins 
from scipy.signal import butter, sosfilt
from scipy.stats import zscore
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# prepare firing rate
def bin_array(array, BIN, time_array):
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)

def heaviside(x):
    return 0.5 * (1 + np.sign(x))
   
   
def input_rate(t, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau):
    # t1_exc=10. # time of the maximum of external stimulation
    # tau1_exc=20. # first time constant of perturbation = rising time
    # tau2_exc=50. # decaying time
    # ampl_exc=20. # amplitude of excitation
    inp = ampl_exc * (np.exp(-(t - t1_exc) ** 2 / (2. * tau1_exc ** 2)) * heaviside(-(t - t1_exc)) + \
        heaviside(-(t - (t1_exc+plateau))) * heaviside(t - (t1_exc))+ \
        np.exp(-(t - (t1_exc+plateau)) ** 2 / (2. * tau2_exc ** 2)) * heaviside(t - (t1_exc+plateau)))
    return inp

def detect_UP(train_cut, ratioThreshold=0.4,
              sampling_rate=1., len_state=50.,
              gauss_width_ratio=10., min_for_up=0.2):
    """
    detect UP states from time signal
    (population rate or population spikes or cell voltage trace)
    return start and ends of states.

    Written by Trang-Anh Nghiem. Modified with min_for_up by David Aquilue

    Parameters
    ----------
    train_cut: array
        array of shape (N, ) containing the time trace on which we detect upstates

    ratioThreshold: float
        Over which % of the FR maximum value in the time trace of a region we consider an up-state

    sampling_rate: float
        Sampling rate of the time trace. Usually 1 / dt. In ms**(-1)

    len_state: float
        Minimum length (in ms) of time over threshold to be considered up-state (I think)

    gauss_width_ratio: float
        Width ratio of the Gaussian Kernel used in the filter for detecting up-states.

    min_for_up: float
        A value under which there is no up-state. That way, if we have high relative variations
        near 0 value but the FR is not higher than 0.3 there will be no up-state.
        However, take into account that this will modify the functioning of the algorithm, possibly
        underestimating up state duration.

    Returns
    -------
    idx: array
        indexes where there is a change of state.
    train_shift: array
        time trace of the filtered signal - ratioThreshold * np.max(train_filtered)
    train_bool: array
        array containing 1s when up state and 0s when downstate
    """
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
    diff_remove = np.where(np.diff(idx) < len_state * sampling_rate)[0]
    idx_start_remove = idx[diff_remove]
    idx_end_remove = idx[np.add(diff_remove, 1)] + 1

    for ii_start, ii_end in zip(idx_start_remove, idx_end_remove):
        train_bool[ii_start:ii_end] = np.ones_like(train_bool[ii_start:ii_end]) * train_bool[ii_start - 1]
        # assign to same state as previous long

    idx = np.where(np.diff(train_bool) != 0)[0]
    idx = np.concatenate(([0], idx, [len(train_filt)])) / sampling_rate
    return idx, train_shift, train_bool


@njit
def obtain_updown_durs(train_bool, dt):
    N = train_bool.size
    up_durs = np.empty(0)  # Something of the like up_durs = []
    down_durs = np.empty(0)
    current_up_duration = 0
    current_down_duration = 0
    for k in range(1, N):  # We sweep over all the values of the train_bool signal
        if train_bool[k - 1] == train_bool[k]:  # If 2 consecutive equal values -> increase state duration
            if train_bool[k - 1] == 1:
                current_up_duration += dt
            else:
                current_down_duration += dt
        else:  # If 2 consecutive NOT equal values -> increase state duration + store duration + restore
            if train_bool[k - 1] == 1:
                up_durs = np.append(up_durs, current_up_duration)
                current_up_duration = 0
            else:
                down_durs = np.append(down_durs, current_down_duration)
                current_down_duration = 0
        if k == N - 1:  # Regardless of the value of the last time point, we have to store the last duration.
            if train_bool[k] == 1:
                current_up_duration += dt
                up_durs = np.append(up_durs, current_up_duration)
                current_up_duration = 0
            else:
                current_down_duration += dt
                down_durs = np.append(down_durs, current_down_duration)
                current_down_duration = 0

    if up_durs.size == 0:  # If no up-states, return duration of 0
        mean_up_durs = 0
        total_up_durs = 0
    else:
        mean_up_durs = np.mean(up_durs)
        total_up_durs = np.sum(up_durs)

    if down_durs.size == 0:  # If no down-states, return duration of 0
        mean_down_durs = 0
        total_down_durs = 0
    else:
        mean_down_durs = np.mean(down_durs)
        total_down_durs = np.sum(down_durs)

    return mean_up_durs, mean_down_durs, total_up_durs, total_down_durs

# ----- Raster plot + mean adaptation ------
def plot_raster_meanFR_tau_i(RasG_inh,RasG_exc, TimBinned, popRateG_inh, popRateG_exc, Pu, sim_name,  b_e, tau_e, tau_i, EL_i, EL_e, Iext, path):
    fig=figure(figsize=(8,12))
    ax1=fig.add_subplot(211)
    ax3=fig.add_subplot(212)
    ax2 = ax3.twinx()


    ax1.plot(RasG_inh[0], RasG_inh[1], ',r')
    ax1.plot(RasG_exc[0], RasG_exc[1], ',g')

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Neuron index')

    ax3.plot(TimBinned/1000,popRateG_inh, 'r')
    ax3.plot(TimBinned/1000,popRateG_exc, 'SteelBlue')
    ax2.plot(TimBinned/1000,(Pu/8000), 'orange')
    ax2.set_ylabel('mean w (pA)')
    #ax2.set_ylim(0.0, 0.045)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('population Firing Rate')

    fig.suptitle(f'b_e={b_e}, tau_e={tau_e}, tau_i={tau_i}, EL_i = {EL_i}, EL_e = {EL_e}, Iext = {Iext}')

    #/DATA/Maria/Anesthetics/network_sims/big_loop_tau_i/figures/eli_-64.0_ele_-63.0/Iext_0.4/ +sim_name
    fol_name = path + "figures/" + f"eli_{int(EL_i)}_ele_{int(EL_e)}/Iext_{Iext}/"

    try:
        os.listdir(fol_name)
    except:
        os.makedirs(fol_name)

    fig.savefig(fol_name + sim_name + '.png')

    plt.show()

    #
    # fig.savefig(fol_name + sim_name + '_raster_plot_mean_w' + '.png')
    # plt.close()

def plot_raster_meanFR(RasG_inh,RasG_exc, TimBinned, popRateG_inh, popRateG_exc, Pu, axes, sim_name):
    
    # fig=figure(figsize=(8,12))
    ax1 = axes[0]
    ax3 = axes[1]
    # ax1=fig.add_subplot(211)
    # ax3=fig.add_subplot(212)
    ax2 = ax3.twinx()

    ax1.plot(RasG_inh[0], RasG_inh[1], ',r')
    ax1.plot(RasG_exc[0], RasG_exc[1], ',g')

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Neuron index')

    ax3.plot(TimBinned/1000,popRateG_inh, 'r', label='Inh')
    ax3.plot(TimBinned/1000,popRateG_exc, 'SteelBlue', label='Exc')
    ax2.plot(TimBinned/1000,(Pu/8000), 'orange', label='W')
    ax2.set_ylabel('mean w (pA)')
    #ax2.set_ylim(0.0, 0.045)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('population Firing Rate')

   # ask matplotlib for the plotted objects and their labels
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    ax1.set_title(sim_name)
    plt.show()

def bin_array(array, BIN, time_array):
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)
#---------------------------------------------------------------------#
# prepare firing rate
def prepare_FR(TotTime,DT, FRG_exc, FRG_inh, P2mon ):
    def bin_array(array, BIN, time_array):
        N0 = int(BIN/(time_array[1]-time_array[0]))
        N1 = int((time_array[-1]-time_array[0])/BIN)
        return array[:N0*N1].reshape((N1,N0)).mean(axis=1)

    BIN=5
    time_array = arange(int(TotTime/DT))*DT

    LfrG_exc = array(FRG_exc.rate/Hz)
    TimBinned,popRateG_exc = bin_array(time_array, BIN, time_array),bin_array(LfrG_exc, BIN, time_array)

    LfrG_inh = array(FRG_inh.rate/Hz)
    TimBinned,popRateG_inh = bin_array(time_array, BIN, time_array),bin_array(LfrG_inh, BIN, time_array)

    TimBinned,Pu = bin_array(time_array, BIN, time_array),bin_array(P2mon[0].P, BIN, time_array)

    return TimBinned, popRateG_exc, popRateG_inh, Pu

def create_simname_tau_i(b_e, tau_i, Iext, EL_i, EL_e):
    # _b_0_tau_i_5.0_Iext_0.4_eli_-64_ele_-63
    b_e = int(b_e)
    EL_i = int(EL_i)
    EL_e = int(EL_e)

    sim_name = f'_b_{b_e}_tau_i_{tau_i}_Iext_{Iext}_eli_{EL_i}_ele_{EL_e}'

    return sim_name

def create_simname_both_taus(b_e, tau_e, tau_i, Iext, EL_i, EL_e):
    # _b_0_tau_i_5.0_Iext_0.4_eli_-64_ele_-63
    b_e = int(b_e)
    EL_i = int(EL_i)
    EL_e = int(EL_e)


    sim_name = f'_b_{b_e}_tau_e_{tau_e}_tau_i_{tau_i}_Iext_{Iext}_eli_{EL_i}_ele_{EL_e}'

    return sim_name


def create_simname_tau_e(b_e, tau_e, Iext, EL_i, EL_e):

    b_e = int(b_e*1e+12)
    tau_e = round(tau_e * 1e+3, 1)
    EL_i = int(EL_i * 1e+3)
    EL_e = int(EL_e * 1e+3)

    sim_name = f'_b_{b_e}_tau_e_{tau_e}_Iext_{Iext}_eli_{EL_i}_ele_{EL_e}'

    return sim_name

def create_folder(b_e, tau_e, Iext, EL_i, EL_e, folder_root):


    sim_name = f'_b_{b_e}_tau_e_{tau_e}_Iext_{Iext}_eli_{EL_i}_ele_{EL_e}'

    subfolder = folder_root + '/' + f"eli_{EL_i}_ele_{EL_e}" + '/'

    try:
        os.listdir(subfolder)
    except:
        os.mkdir(subfolder)

    subsubfolder = subfolder + '/' + f"Iext_{Iext}" + '/'

    try:
        os.listdir(subsubfolder)
    except:
        os.mkdir(subsubfolder)

    folder_name = subsubfolder + sim_name + '/'
    print(folder_name)

    try:
        os.listdir(folder_name)
        #print("path existing:" + folder_name)
    except:
        os.mkdir(folder_name)
        print("path created:" + folder_name)

    return folder_name, sim_name

def create_folder_2(b_e, tau_e, Iext, EL_i, EL_e, folder_root ):


    sim_name = f'_b_{b_e}_tau_e_{tau_e}_Iext_{Iext}_eli_{EL_i}_ele_{EL_e}'
    print(sim_name)

    subfolder = folder_root + '\\' + f"eli_{EL_i}_ele_{EL_e}" + '\\'

    try:
        os.listdir(subfolder)
    except:
        os.mkdir(subfolder)

    subsubfolder = subfolder + '\\' + f"Iext_{Iext}" + '\\'

    try:
        os.listdir(subsubfolder)
    except:
        os.mkdir(subsubfolder)

    folder_name = subsubfolder + sim_name + '\\'
    print(folder_name)

    try:
        os.listdir(folder_name)
        #print("path existing:" + folder_name)
    except:
        os.mkdir(folder_name)
        print("path created:" + folder_name)

    return folder_name, sim_name


def create_combination(bvals, tau_es, EL_es, EL_is, Iexts, neglect_silence = True):

    lst = [bvals, tau_es, EL_es, EL_is, Iexts]

    combinaison = np.array(list(itertools.product(*lst)))

    if neglect_silence:
        idx_keep = combinaison[:, 3] <= combinaison[:, 2]  # We keep E_L_e > E_L_i - thresh_silence
        combinaison = combinaison[idx_keep, :]  # And eliminate those combinations that are not needed.

        idx_not = []
        for idx in range(len(combinaison)):
            if combinaison[idx, 1] == 4 and combinaison[idx, 4] == 0.4 and combinaison[idx, 0] != 0 and (
                    combinaison[idx, 3] == -67 or combinaison[idx, 3] == -65):
                idx_not.append(False)
            else:
                idx_not.append(True)

        combinaison = combinaison[idx_not, :]
    return combinaison


def create_combination_neglect_only(bvals, tau_es, EL_es, EL_is, Iexts, neglect=True):
    # create the combination with the values that were neglected
    lst = [bvals, tau_es, EL_es, EL_is, Iexts]

    combinaison = np.array(list(itertools.product(*lst)))

    idx_not = []

    idx_keep = combinaison[:, 3] <= combinaison[:, 2]  # We keep E_L_e > E_L_i - thresh_silence
    combinaison = combinaison[idx_keep, :]  # And eliminate those combinations that are not needed.

    if neglect:
        for idx in range(len(combinaison)):
            if combinaison[idx, 1] == 4 and combinaison[idx, 4] == 0.4 and combinaison[idx, 0] != 0 and (
                    combinaison[idx, 3] == -67 or combinaison[idx, 3] == -65):
                idx_not.append(True)
            else:
                idx_not.append(False)

        combinaison = combinaison[idx_not, :]
    return combinaison

def create_combination_iext04(bvals, tau, EL_es, EL_is, Iexts):

    lst = [bvals, tau, EL_es, EL_is, Iexts]

    combinaison = np.array(list(itertools.product(*lst)))

    # we want the neglect on the Ele and Eli
    idx_keep = combinaison[:, 3] <= combinaison[:, 2]  # We keep E_L_e > E_L_i - thresh_silence
    combinaison = combinaison[idx_keep, :]  # And eliminate those combinations that are not needed.

    return combinaison

def calculate_psd_fmax(popRateG_exc, popRateG_inh, TimBinned):
    time_s = TimBinned * 0.001  # time has to be in seconds and here it is in ms
    FR_exc = popRateG_exc
    FR_inh = popRateG_inh

    f_sampling = 1. * len(time_s) / time_s[-1]  # time in seconds, f_sampling in Hz
    frq = np.fft.fftfreq(len(time_s), 1 / f_sampling)

    Esig = np.transpose(FR_exc)
    Isig = np.transpose(FR_inh)

    pwr_region_E = np.abs(np.fft.fft(Esig)) ** 2
    pwr_region_I = np.abs(np.fft.fft(Isig)) ** 2

    good_idxs = frq > 0
    frq_good = frq[good_idxs]
    pwr_region_E_good = pwr_region_E[good_idxs]
    pwr_region_I_good = pwr_region_I[good_idxs]
    max_idx = pwr_region_E_good == pwr_region_E_good.max()
    frq_max = frq_good[max_idx][0]

    return frq_max,frq_good, pwr_region_E_good, pwr_region_I_good

def plot_psd(frq_max, frq_good, pwr_region_E_good, pwr_region_I_good):
    fig, axes = plt.subplots(1, 1, figsize=(16, 8))
    plt.rcParams.update({'font.size': 14})

    axes.loglog(frq_good, pwr_region_I_good, "-", color='darkred', alpha=0.9, label='Inh.')
    axes.loglog(frq_good, pwr_region_E_good, "-", color='SteelBlue', label='Exc.')
    axes.axvline(x=frq_max, color='b', label='fmax = {:.2f}'.format(frq_max))

    axes.set_xlabel('Frequency (Hz)')
    axes.set_ylabel('Power')

    axes.legend()

    plt.tight_layout()
    plt.show()

def adjust_parameters(parameters, b_e = 5, tau_e = 5.0, tau_i = 5.0, Iext = 0.000315, 
                      stimval = 0,stimdur = 50,stimtime_mean = 2500. ,stim_region = 5, n_nodes=68, 
                      cut_transient=2000, run_sim=5000, nseed=10, additional_path_folder=''):
    """
    assign the desired b_e, tau_e, tau_i, iext, stimval, stimdur, stim_region
    if needed to change other parameters, it can be done manually
    additional_path_folder : add in the form of 'Bold/' in case you are adding other monitors
    """

    if stimval:
        folder_root= './result/evoked'
        sim_name =  f"stim_{stimval}_b_e_{b_e}_tau_e_{tau_e}_tau_i_{tau_i}_Iext_{Iext}_El_e_{parameters.parameter_model['E_L_e']}_El_i_{parameters.parameter_model['E_L_i']}_nseed_{nseed}"

    else:
        folder_root = './result/synch'
        sim_name =  f"_b_e_{b_e}_tau_e_{tau_e}_tau_i_{tau_i}_Iext_{Iext}_El_e_{parameters.parameter_model['E_L_e']}_El_i_{parameters.parameter_model['E_L_i']}_nseed_{nseed}"
    
    print(sim_name)
    parameters.parameter_simulation['path_result'] = folder_root + '/' + sim_name + '/' +additional_path_folder

    try:
        os.listdir(parameters.parameter_simulation['path_result'])
    except:
        os.makedirs(parameters.parameter_simulation['path_result'])

    parameters.parameter_model['b_e'] = b_e

    parameters.parameter_model['tau_e'] = tau_e
    parameters.parameter_model['tau_i'] = tau_i

    parameters.parameter_model['external_input_ex_ex']=Iext
    parameters.parameter_model['external_input_in_ex']=Iext

    # parameters for stimulus
    weight = list(np.zeros(n_nodes))
    weight[stim_region] = stimval # region and stimulation strength of the region 0 

    parameters.parameter_stimulus["tau"]= stimdur # stimulus duration [ms]
    parameters.parameter_stimulus["T"]= 1e9 # interstimulus interval [ms]
    parameters.parameter_stimulus["weights"]= weight
    parameters.parameter_stimulus["variables"]=[0] #variable to kick - it is the FR_exc

    parameters.parameter_stimulus['onset'] = cut_transient + 0.5*(run_sim-cut_transient)
    stim_time = parameters.parameter_stimulus['onset']
    stim_steps = stim_time*10 #number of steps until stimulus

    return parameters

def get_result(parameters,time_begin,time_end, prints=0, b_e = 5, tau_e = 5.0, tau_i = 5.0, 
               Iext = 0.000315,nseed=10, vars_int = ['E', 'I', 'W_e'], additional_path_folder=''):
    '''
    return the result of the simulation between the wanted time
    :parameters: the parameter variable
    :time_begin: the start time for the result (basically to discard cut_transient)
    :time_end:  the ending time for the result
    :vars_int: the variables of interest to load
     for AdEx:
        'E': excitatory FR
        'I': inhibitory FR
        'C_ee': covariance exc
        'C_ei': covariance exc-inh
        'C_ii': covariance inh
        'W_e': adaptatioin exc
        'W_i': adaptation inh
        'noise': noise 
    :return: result of all monitor
        it will be a list with length equal to the monitors
        each element of the list will be an array containing the variables of interest
        this array will have shape (var_int, time, n_nodes)
    '''
    folder_root = './result/synch'
    sim_name =  f"_b_e_{b_e}_tau_e_{tau_e}_tau_i_{tau_i}_Iext_{Iext}_El_e_{parameters.parameter_model['E_L_e']}_El_i_{parameters.parameter_model['E_L_i']}_nseed_{nseed}"
    
    print("Loading: ", sim_name)
    path = folder_root + '/' + sim_name + '/' + additional_path_folder

    with open(path + '/parameter.json') as f:
        parameters = json.load(f)
    parameter_simulation = parameters['parameter_simulation']
    parameter_monitor = parameters['parameter_monitor']
    # print("parameter monitor: ", parameter_monitor)
    count_begin = int(time_begin/parameter_simulation['save_time'])
    count_end = int(time_end/parameter_simulation['save_time'])+1
    nb_monitor = parameter_monitor['Raw'] + parameter_monitor['TemporalAverage'] + parameter_monitor['Bold'] + parameter_monitor['Ca'] #nuu added Ca monitor
    if 'Afferent_coupling' in parameter_monitor.keys() and parameter_monitor['Afferent_coupling']:
        nb_monitor+=1
    output =[]
    print("monitors:", nb_monitor)
    for count in range(count_begin,count_end):
        if prints:
            print ('count {0} from {1}'.format(count,count_end))
        
        result = np.load(path+'/step_'+str(count)+'.npy',allow_pickle=True)
        for i in range(result.shape[0]):
            tmp = np.array(result[i])
            if len(tmp) != 0:
                tmp = tmp[np.where((time_begin <= tmp[:,0]) &  (tmp[:,0]<= time_end)),:]
                tmp_time = tmp[0][:,0]
                if tmp_time.shape[0] != 0:
                    one = tmp[0][:,1][0]
                    tmp_value = np.concatenate(tmp[0][:,1]).reshape(tmp_time.shape[0],one.shape[0],one.shape[1])
                    if len(output) == nb_monitor:
                        output[i]=[np.concatenate([output[i][0],tmp_time]),np.concatenate([output[i][1],tmp_value])]
                    else:
                        output.append([tmp_time,tmp_value])
    
    # indices of each variabel in the result
    dict_var_int = {'E':0 ,'I': 1 ,'C_ee': 2 ,'C_ei': 3,'C_ii': 4,'W_e': 5,'W_i': 6,'noise': 7}
    len_var = len(vars_int)

    #first iterate the monitors
    result = []
    for i in range(nb_monitor):
        time_s = output[i][0]
        n_nodes = output[i][1][:,0,:].shape[1]
        #create empty array with shape (number of variables of interest, time)
        var_arr = np.zeros((len_var, time_s.shape[0], n_nodes))

        c = 0
        for var in vars_int:
            index_var = dict_var_int[var] #get the index of the variables
            if index_var < 2 or index_var==7: #if it is the exc, inh FR or noise, transform from KHz to Hz
                res = output[i][1][:,index_var,:]*1e3
                var_arr[c] = res
            else:
                res = output[i][1][:,index_var,:]
                var_arr[c] = res
            c+=1 
        
        result.append(var_arr)
    shape = np.shape(result[0][0])
    del output
    # access_results(parameter_monitor,vars_int,shape)
    return result, (parameter_monitor,vars_int,shape)

def access_results(for_explan, bvals, tau_es, change_of='tau_e'):
    """
    print how the results are indexed
    result = this array will have shape (var_int, time, n_nodes)

    """
    (parameter_monitor,vars_int,shape) = for_explan

    print("\nExplaining the indices in result:")
    print('The result has a length equal to you different parameter combinations, i.e:')
    for i in range(len(bvals)):    
        print(f'result[{i}]: for b_e = {bvals[i]} and {change_of}= {tau_es[i]}')

    print('\nThe result[i] is a list of arrays, every element of a list corresponds to a monitor:')

    true_monitors = [key for key in parameter_monitor.keys() if not key.startswith("param") and parameter_monitor[key]]
    list_monitors = []
    c= 0
    for key in parameter_monitor.keys():
        if parameter_monitor[key] is True:
            print(f'{key} monitor : result[i][{c}]')
            list_monitors.append(key)
            c+=1

    print('\nEach monitor contains an array with the selected variables of interest, for all the time points and nodes')
    print(f"For example for {list_monitors[0]} monitor:")

    k=0
    for var in vars_int:
        print(f'For {var} : result[i][0][{k}]')
        k+=1

    print(f'\nThese arrays have shape: time_points x number_of_nodes: {shape}')

def get_np_arange(value):
   """
   solution to input np.arange in the argparser 
   """
   try:
       values = [float(i) for i in value.split(',')]
       assert len(values) in (1, 3)
   except (ValueError, AssertionError):
       raise argparse.ArgumentTypeError(
           'Provide a CSV list of 1 or 3 integers'
       )

   # return our value as is if there is only one
   if len(values) == 1:
       return np.array(values)

   # if there are three - return a range
   return np.arange(*values)

def create_dicts(parameters,param, result, monitor, for_explan, var_select, change_of='tau_e', 
               Iext = 0.000315,nseed=10, additional_path_folder='', return_TR=False):
    """
    parameters: the parameters of the model
    param : tuple, in the form of (i, [b_e, tau])
    result: the result with all the parametrizations, and the selected monitors and selected variables
    monitor: the selected monitor (string)
    for_explan: tuple, from the get_result function, to catch all the monitors that have been used in the get_result (parameter_monitor,vars_int,Nnodes)
    var_select: the variables to be plotted
    change_of: str, 'tau_e' if you change values of tau_e, 'tau_i' otherwise
    return a dictionary with key the selected variables, 
    """
    #Take the parameters of interest
    (i, [b_e, tau_e]) = param
    result = result[i]
    if change_of=='tau_i':
        tau_i = tau_e
        tau_e = 5.0
    else:
        tau_i = 5.0

    #Take the monitor of interest
    folder_root = './result/synch'
    sim_name =  f"_b_e_{b_e}_tau_e_{tau_e}_tau_i_{tau_i}_Iext_{Iext}_El_e_{parameters.parameter_model['E_L_e']}_El_i_{parameters.parameter_model['E_L_i']}_nseed_{nseed}"
    path = folder_root + '/' + sim_name + '/' +additional_path_folder
    with open(path + '/parameter.json') as f:
        parameters = json.load(f)
    parameter_monitor = parameters['parameter_monitor']

    list_monitors = {} # {Raw : 0, Temporal:1, etc}
    c= 0
    for key in parameter_monitor.keys():
        if parameter_monitor[key] is True:
            list_monitors[key] = c
            c+=1
    
    result = result[list_monitors[monitor]] #take the wanted monitor

    #Take the variables of interest
    (_,vars_int,_) = for_explan

    list_vars = {}
    k=0
    for var in vars_int:
        list_vars[var] = k
        k+=1
    
    result_fin = {}

    for var in var_select:
        result_fin[var] = result[list_vars[var]]

    if return_TR:
        TR= parameter_monitor["parameter_Bold"]["period"]
        return result_fin,TR
    else:
        return result_fin

def plot_tvb_results(parameters,params, result, monitor, for_explan, var_select,cut_transient, run_sim, change_of='tau_e', 
               Iext = 0.000315,nseed=10, additional_path_folder='',figsize=(8,5), desired_time=0):
    """
    desired_time: in s, plot from a specific time point
    """
    rows =int(len(params))
    cols = len(var_select) 
    if 'E' and 'I' in var_select:
        cols = cols-1 #put E and I in the same plot
    single_plot=False
    if cols == 1:
        if rows>1:
            cols=rows
            rows=1
        else:
            cols=2
            single_plot=True

    fig, axes = plt.subplots(rows,cols,figsize=figsize)
    
    for param in enumerate(params):
        #load results for the params
        (i, [b_e, tau]) = param 
        result_fin = create_dicts(parameters,param, result, monitor, for_explan, var_select, change_of=change_of, Iext = Iext,nseed=nseed, additional_path_folder=additional_path_folder)

        #create list with the indices for each variable
        var_ind_list = {}
        j =0
        for var in var_select:
            if len(axes.shape) ==1:
                ax_index = i
                i +=1 
            else: 
                ax_index= (i, j )#that means that you have one row or one col
            if (var == 'E' and 'I' in var_ind_list.keys()) or (var == 'I' and 'E' in var_ind_list.keys()):
                continue
            else:
                var_ind_list[var] = ax_index
                j+=1
        #if E and I in the vars, plot them in the same subplot
        if 'E' and 'I' in result_fin.keys():
            try:
                ax_ind= var_ind_list['E']
                del var_ind_list['E']
            except KeyError:
                ax_ind = var_ind_list['I']
                del var_ind_list['I']

            time_s = np.linspace(cut_transient, run_sim, result_fin['E'].shape[0])/1000
            closest_index = np.argmin(np.abs(time_s - desired_time)) # index of the time point closest to the desired time
            Li = axes[ax_ind].plot(time_s[closest_index:],result_fin['I'][closest_index:],color='darkred', label='Inh') # [times, regions]
            Le = axes[ax_ind].plot(time_s[closest_index:],result_fin['E'][closest_index:],color='SteelBlue', label='Exc') # [times, regions]
            axes[ax_ind].set_ylabel('Firing rate (Hz)', fontsize=16)
            axes[ax_ind].set_title(change_of+ f'= {tau} ms, b_e={b_e}', fontsize=16)
            axes[ax_ind].legend([Li[0], Le[0]], ['Inh.','Exc.'], loc='upper right', fontsize='xx-small')


            for var in var_ind_list.keys():
                ax_ind= var_ind_list[var]
                time_s = np.linspace(cut_transient, run_sim,result_fin[var].shape[0])/1000    
                closest_index = np.argmin(np.abs(time_s - desired_time)) # index of the time point closest to the desired time
                Li = axes[ax_ind].plot(time_s[closest_index:],result_fin[var][closest_index:], label=var) # [times, regions]
                axes[ax_ind].set_ylabel(var, fontsize=16)
                axes[ax_ind].set_title(change_of+ f'= {tau} ms, b_e={b_e}', fontsize=16)
                axes[ax_ind].legend([Li[0]], [var], loc='upper right', fontsize='xx-small') 
        
        #else plot all the variables separately
        else:
            for var in var_ind_list.keys():
                ax_ind= var_ind_list[var]
                time_s = np.linspace(cut_transient, run_sim,result_fin[var].shape[0])/1000    
                closest_index = np.argmin(np.abs(time_s - desired_time)) # index of the time point closest to the desired time
                Li = axes[ax_ind].plot(time_s[closest_index:],result_fin[var][closest_index:], label=var) # [times, regions]
                axes[ax_ind].set_ylabel(var, fontsize=16)
                axes[ax_ind].set_title(change_of+ f'= {tau} ms, b_e={b_e}', fontsize=16)
                axes[ax_ind].legend([Li[0]], [var], loc='upper right', fontsize='xx-small')    
    
    for ax in axes.reshape(-1):
        ax.set_xlabel('Time (s)')

    if single_plot:
        fig.delaxes(axes[1])

    plt.tight_layout()
    plt.show()

def calculate_PCI(parameters, n_seeds, run_sim, cut_transient, stimval=1e-3, b_e=5, tau_e=5.0, tau_i=5.0, Iext=0.000315, n_trials = 5 ):
    # Perturbational Complexity Index (PCI) computation and saving
    # number of independent random seeds and simulations
    # n_trials: number of simulations/realisations to analyse for one PCI value

    sim_names = np.arange(0,n_seeds,n_trials) 
    for sim_curr in sim_names:   #0,  5, 10, 15, 20, 25, 30, 35
        print(sim_curr)
        entropy_trials = []
        LZ_trials = []
        PCI_trials = []

        sig_cut_analysis = []
        t_stim_onsets = []
        for i_trials in range(sim_curr, sim_curr + n_trials): 
            print(i_trials)
            # if we had 40 seeds and n_trials=5 then 
            #for i_trials in range(0, 5)
            # for i_trials in range (5,10)
            times_l = []
            rateE_m = []
            nstep = int(run_sim/1000) # number of saved files

            sim_name =  f"stim_{stimval}_b_e_{b_e}_tau_e_{tau_e}_tau_i_{tau_i}_Iext_{Iext}_El_e_{parameters.parameter_model['E_L_e']}_El_i_{parameters.parameter_model['E_L_i']}_nseed_{i_trials}"

            folder_path = './result/evoked/' + sim_name+'/'

            for i_step in range(nstep):
                raw_curr = np.load(folder_path + 'step_'+str(i_step)+'.npy',
                encoding = 'latin1', allow_pickle=True)
                for i_time in range(len(raw_curr[0])): 
                    times_l.append(raw_curr[0][i_time][0])
                    rateE_m.append(np.concatenate(raw_curr[0][i_time][1][0]))

            times_l = np.array(times_l) # in ms
            rateE_m = np.array(rateE_m) # matrix of size nbins*nregions

            # choosing variable of interest
            var_of_interest = rateE_m
            
            # discard transient
            nbins_transient = int(cut_transient/times_l[0]) # to discard in analysis   
            sig_region_all = var_of_interest[nbins_transient:,:] 
            sig_region_all = np.transpose(sig_region_all) # now formatted as regions*times

            # load t_onset
            with open(folder_path+"parameter.json", 'r') as json_file:
                data = json.load(json_file)
            onset_value = data['parameter_stimulation']['onset']
            t_stim_bins = int((onset_value - cut_transient)/times_l[0])
            
            #save all the onsets:
            t_stim_onsets.append(t_stim_bins)

            t_analysis = 300 #ms
            nbins_analysis =  int(t_analysis/times_l[0])
            
            sig_cut_region =  sig_region_all[:,t_stim_bins - nbins_analysis:t_stim_bins + nbins_analysis]
            
            # append directly the sig_cut_analysis
            sig_cut_analysis.append(sig_cut_region)

        sig_all_binary = tools.binarise_signals(np.array(sig_cut_analysis), int(t_analysis/times_l[0]), 
                                        nshuffles = 10, percentile = 100)
        
        #return entropy
        for ijk in range(n_trials):
            binJ=sig_all_binary.astype(int)[ijk,:,t_analysis:] # CHECK each row is a time series !
            binJs=pci_v2.sort_binJ(binJ) # sort binJ as done in Casali et al. 2013
            source_entropy=pci_v2.source_entropy(binJs)
            print('Entropy', source_entropy)

            # return Lempel-Ziv
            Lempel_Ziv_lst=pci_v2.lz_complexity_2D(binJs)
            print('Lempel-Ziv', Lempel_Ziv_lst)

            #normalization factor 
            norm=pci_v2.pci_norm_factor(binJs)

            # computing perturbational complexity index
            pci_lst = Lempel_Ziv_lst/norm
            print('PCI', pci_lst)

            all_entropy_lst=[source_entropy,pci_lst]

            entropy_trials.append(all_entropy_lst) 
            LZ_trials.append(Lempel_Ziv_lst) 
            PCI_trials.append(pci_lst) 

        # file saving
        save_file_name = folder_path + f'Params_PCI_bE_{b_e}_stim_{stimval}_tau_e_{tau_e}_tau_i_{tau_i}_trial_{int(sim_curr/n_trials)}.npy'
        savefile = {}
        savefile['entropy'] = np.array(entropy_trials)
        savefile['Lempel-Ziv'] = np.array(LZ_trials)
        savefile['PCI'] = np.array(PCI_trials)

        np.save(save_file_name, savefile)
        print(save_file_name)
        print('Seed', sim_curr + ijk, ' done\n')
    clear_output(wait=False)
    print(f"Done: b_e={b_e}, tau_e={tau_e}, tau_i={tau_i}", flush=True)

def sim_init(parameters, initial_condition=None, my_seed = 10):
    '''
    Initialise the simulator with parameter

    :param parameter_simulation: parameters for the simulation
    :param parameter_model: parameters for the model
    :param parameter_connection_between_region: parameters for the connection between nodes
    :param parameter_coupling: parameters for the coupling of equations
    :param parameter_integrator: parameters for the intergator of the equation
    :param parameter_monitor: parameters for the monitors
    :param initial_condition: the possibility to add an initial condition
    :return: the simulator initialize
    '''

    parameter_simulation  = parameters.parameter_simulation
    parameter_model = parameters.parameter_model
    parameter_connection_between_region = parameters.parameter_connection_between_region
    parameter_coupling = parameters.parameter_coupling
    parameter_integrator = parameters.parameter_integrator
    parameter_monitor = parameters.parameter_monitor
    parameter_stimulation = parameters.parameter_stimulus
    ## initialise the random generator
    parameter_simulation['seed'] = my_seed
    rgn.seed(parameter_simulation['seed'])

    if parameter_model['matteo']:
        import tvb_model_reference.src.Zerlaut_matteo as model
    else:
        import tvb_model_reference.src.Zerlaut as model

    ## Model
    if parameter_model['order'] == 1:
        model = model.Zerlaut_adaptation_first_order(variables_of_interest='E I W_e W_i noise'.split())
    elif parameter_model['order'] == 2:
        model = model.Zerlaut_adaptation_second_order(variables_of_interest='E I C_ee C_ei C_ii W_e W_i noise'.split())
    else:
        raise Exception('Bad order for the model')
    # ------- Changed by Maria 
    to_skip=['initial_condition', 'matteo', 'order']
    for key, value in parameters.parameter_model.items():
        if key not in to_skip:
            setattr(model, key, np.array(value))
    for key,val in parameters.parameter_model['initial_condition'].items():
        model.state_variable_range[key] = val
    

    ## Connection
    if parameter_connection_between_region['default']:
        connection = lab.connectivity.Connectivity().from_file()
    elif parameter_connection_between_region['from_file']:
        path = parameter_connection_between_region['path']
        conn_name = parameter_connection_between_region['conn_name']
        connection = lab.connectivity.Connectivity().from_file(path+'/' + conn_name)
    elif parameter_connection_between_region['from_h5']:
        connection = lab.connectivity.Connectivity().from_file(parameter_connection_between_region['path'])
    elif parameter_connection_between_region['from_folder']:
        # mandatory file 
        tract_lengths = np.loadtxt(parameter_connection_between_region['path']+'/tract_lengths.txt')
        weights = np.loadtxt(parameter_connection_between_region['path']+'/weights.txt')
        # optional file
        if os.path.exists(parameter_connection_between_region['path']+'/region_labels.txt'):
            region_labels = np.loadtxt(parameter_connection_between_region['path']+'/region_labels.txt', dtype=str)
        else:
            region_labels = np.array([], dtype=np.dtype('<U128'))
        if os.path.exists(parameter_connection_between_region['path']+'/centres.txt'):
            centers = np.loadtxt(parameter_connection_between_region['path']+'/centres.txt')
        else:
            centers = np.array([])
        if os.path.exists(parameter_connection_between_region['path']+'/cortical.txt'):
            cortical = np.array(np.loadtxt(parameter_connection_between_region['path']+'/cortical.txt'),dtype=np.bool)
        else:
            cortical=None
        connection = lab.connectivity.Connectivity(
                                                   tract_lengths=tract_lengths,
                                                   weights=weights,
                                                   region_labels=region_labels,
                                                   centres=centers.T,
                                                   cortical=cortical)
    else:
        connection = lab.connectivity.Connectivity(
                                                number_of_regions=parameter_connection_between_region['number_of_regions'],
                                               tract_lengths=np.array(parameter_connection_between_region['tract_lengths']),
                                               weights=np.array(parameter_connection_between_region['weights']),
            region_labels=np.arange(0, parameter_connection_between_region['number_of_regions'], 1, dtype='U128'),#TODO need to replace by parameter
            centres=np.arange(0, parameter_connection_between_region['number_of_regions'], 1),#TODO need to replace by parameter
        )

    if 'normalised'in parameter_connection_between_region.keys() and parameter_connection_between_region['normalised']:
        connection.weights = connection.weights/(np.sum(connection.weights,axis=0)+1e-12)
    connection.speed = np.array(parameter_connection_between_region['speed'])


    ## Stimulus: added by TA and Jen
    if parameter_stimulation['weights'] is None or any(parameter_stimulation['weights'])== 0.0: #changed by Maria
        stimulation = None
    else:
        eqn_t = lab.equations.PulseTrain()
        eqn_t.parameters["onset"] = np.array(parameter_stimulation["onset"]) # ms
        eqn_t.parameters["tau"]   = np.array(parameter_stimulation["tau"]) # ms
        eqn_t.parameters["T"]     = np.array(parameter_stimulation["T"]) # ms; # 0.02kHz repetition frequency
        stimulation = lab.patterns.StimuliRegion(temporal=eqn_t,
                                          connectivity=connection,
                                          weight=np.array(parameter_stimulation['weights']))
        model.stvar = parameter_stimulation['variables']
    ## end add

    ## Coupling
    if parameter_coupling['type'] == 'Linear':
        coupling = lab.coupling.Linear(a=np.array(parameter_coupling['parameter']['a']),
                                       b=np.array(parameter_coupling['parameter']['b']))
    elif parameter_coupling['type'] == 'Scaling':
        coupling = lab.coupling.Scaling(a=np.array(parameter_coupling['parameter']['a']))
    elif parameter_coupling['type'] == 'HyperbolicTangent':
        coupling = lab.coupling.HyperbolicTangent(a=np.array(parameter_coupling['parameter']['a']),
                                       b=np.array(parameter_coupling['parameter']['b']),
                                       midpoint=np.array(parameter_coupling['parameter']['midpoint']),
                                       sigma= np.array(parameter_coupling['parameter']['sigma']),)
    elif parameter_coupling['type'] == 'Sigmoidal':
        coupling = lab.coupling.Sigmoidal(a=np.array(parameter_coupling['parameter']['a']),                                       b=parameter_coupling['b'],
                                       midpoint=np.array(parameter_coupling['parameter']['midpoint']),
                                       sigma= np.array(parameter_coupling['parameter']['sigma']),
                                       cmin=np.array(parameter_coupling['parameter']['cmin']),
                                       cmax=np.array(parameter_coupling['parameter']['cmax']))
    elif parameter_coupling['type'] == 'SigmoidalJansenRit':
        coupling = lab.coupling.SigmoidalJansenRit(a=np.array(parameter_coupling['parameter']['a']),                                       b=parameter_coupling['b'],
                                       midpoint=np.array(parameter_coupling['parameter']['midpoint']),
                                       r= np.array(parameter_coupling['parameter']['r']),
                                       cmin=np.array(parameter_coupling['parameter']['cmin']),
                                       cmax=np.array(parameter_coupling['parameter']['cmax']))
    elif parameter_coupling['type'] == 'PreSigmoidal':
        coupling = lab.coupling.PreSigmoidal(H=np.array(parameter_coupling['parameter']['H']),                                       b=parameter_coupling['b'],
                                       Q=np.array(parameter_coupling['parameter']['Q']),
                                       G= np.array(parameter_coupling['parameter']['G']),
                                       P=np.array(parameter_coupling['parameter']['P']),
                                       theta=np.array(parameter_coupling['parameter']['theta']),
                                       dynamic=np.array(parameter_coupling['parameter']['dynamic']),
                                       globalT=np.array(parameter_coupling['parameter']['globalT']),
                                             )
    elif parameter_coupling['type'] == 'Difference':
        coupling = lab.coupling.Difference(a=np.array(parameter_coupling['parameter']['a']))
    elif parameter_coupling['type'] == 'Kuramoto':
        coupling = lab.coupling.Kuramoto(a=np.array(parameter_coupling['parameter']['a']))
    else:
        raise Exception('Bad type for the coupling')

        
    
    ## Integrator
    if not parameter_integrator['stochastic']:
        if parameter_integrator['type'] == 'Heun':
            integrator = lab.integrators.HeunDeterministic(dt=np.array(parameter_integrator['dt']))
        elif parameter_integrator['type'] == 'Euler':
             integrator = lab.integrators.EulerDeterministic(dt=np.array(parameter_integrator['dt']))
        else:
            raise Exception('Bad type for the integrator')
    else:
        if parameter_integrator['noise_type'] == 'Additive':
            noise = lab.noise.Additive(nsig=np.array(parameter_integrator['noise_parameter']['nsig']),
                                        ntau=parameter_integrator['noise_parameter']['ntau'],)
            # print("type of noise: ", type(noise), "\nand noise: ", noise)
            
        else:
            raise Exception('Bad type for the noise')
        noise.random_stream.seed(parameter_simulation['seed'])

        if parameter_integrator['type'] == 'Heun':
            integrator = lab.integrators.HeunStochastic(noise=noise,dt=parameter_integrator['dt'])
        elif parameter_integrator['type'] == 'Euler':
             integrator = lab.integrators.EulerStochastic(noise=noise,dt=parameter_integrator['dt'])
        else:
            raise Exception('Bad type for the integrator')

    ## Monitors
    monitors =[]
    if parameter_monitor['Raw']:
        monitors.append(lab.monitors.Raw())
    if parameter_monitor['TemporalAverage']:
        monitor_TAVG = lab.monitors.TemporalAverage(
            variables_of_interest=np.array(parameter_monitor['parameter_TemporalAverage']['variables_of_interest']),
            period=parameter_monitor['parameter_TemporalAverage']['period'])
        monitors.append(monitor_TAVG)
    if parameter_monitor['Bold']:
        monitor_Bold = lab.monitors.Bold(
            variables_of_interest=np.array(parameter_monitor['parameter_Bold']['variables_of_interest']),
            period=parameter_monitor['parameter_Bold']['period'])
        monitors.append(monitor_Bold)
    if 'Afferent_coupling' in parameter_monitor.keys() and parameter_monitor['Afferent_coupling']:
        monitor_Afferent_coupling = lab.monitors.AfferentCoupling(variables_of_interest=None)
        monitors.append(monitor_Afferent_coupling)
    if parameter_monitor['Ca']:
        monitor_Ca = lab.monitors.Ca(
            variables_of_interest=np.array(parameter_monitor['parameter_Ca']['variables_of_interest']),
            tau_rise=parameter_monitor['parameter_Ca']['tau_rise'],
            tau_decay=parameter_monitor['parameter_Ca']['tau_decay'])
        monitors.append(monitor_Ca)
    
    #save the parameters in on file
    if not os.path.exists(parameter_simulation['path_result']):
        os.makedirs(parameter_simulation['path_result'])
        f = open(parameter_simulation['path_result']+'/parameter.json',"w")
        f.write("{\n")
        for name,dic in [('parameter_simulation',parameter_simulation),
                        ('parameter_model',parameter_model),
                        ('parameter_connection_between_region',parameter_connection_between_region),
                        ('parameter_coupling',parameter_coupling),
                        ('parameter_integrator',parameter_integrator),
                        ('parameter_monitor',parameter_monitor)]:
            f.write('"'+name+'" : ')
            try:
                json.dump(dic, f)
                f.write(",\n")
            except TypeError:
                print("not serialisable")
        if stimulation is not None:
            f.write('"parameter_stimulation" : ')
            json.dump(parameter_stimulation, f)
            f.write(",\n")

        f.write('"myseed":'+str(my_seed)+"\n}\n")
        f.close()
    elif os.path.exists(parameter_simulation['path_result']+'/parameter.json'):
        # Prompt  input
        response = builtins.input("This path exists already \nYou can see the data (press D), you can stop and choose another path(press N), or you can overwrite (press Y) \nWhat do you want to do? (D/N/Y): ").strip().lower()
        
        # Check if the response is valid
        while response not in ['y', 'n']:
            if response == 'd':
                with open(parameter_simulation['path_result']+'/parameter.json', "r") as f:
                    data = json.load(f)
                list_param = [parameter_model,parameter_connection_between_region ,parameter_coupling ,parameter_integrator,parameter_monitor] 
                list_data = [data['parameter_model'],data['parameter_connection_between_region'],data['parameter_coupling'],
                             data['parameter_integrator'], data['parameter_monitor']]
                list_names = ['model', 'connectivity', 'coupling', 'integrator', 'monitor']
                #compare to see if indeed they have the same model parameters
                for dic_param, dic_data, name in zip(list_param,list_data, list_names):
                    no_diff = print_dict_differences(dic_param, dic_data)
                    if no_diff:
                        print('No differences in ', name)
                print('If there are no differences you can safely continue: press Y')
                response = builtins.input("Do you want to continue (and overwrite)? (Y/N): ").strip().lower()
            else:
                print("Invalid input. Please enter Y, D or N.")
                response = builtins.input("Do you want to continue? (Y/D/N): ").strip().lower()
        
        # Check the user's response
        if response == 'n':
            raise Exception("Try a different path")
        elif response == 'y':
            print("Overwriting...")
            f = open(parameter_simulation['path_result']+'/parameter.json',"w")
            f.write("{\n")
            for name,dic in [('parameter_simulation',parameter_simulation),
                            ('parameter_model',parameter_model),
                            ('parameter_connection_between_region',parameter_connection_between_region),
                            ('parameter_coupling',parameter_coupling),
                            ('parameter_integrator',parameter_integrator),
                            ('parameter_monitor',parameter_monitor)]:
                f.write('"'+name+'" : ')
                try:
                    json.dump(dic, f)
                    f.write(",\n")
                except TypeError:
                    print("not serialisable")
            if stimulation is not None:
                f.write('"parameter_stimulation" : ')
                json.dump(parameter_stimulation, f)
                f.write(",\n")

            f.write('"myseed":'+str(my_seed)+"\n}\n")
            f.close()

    #initialize the simulator: edited by TA and Jen, added stimulation argument, try removing surface
    if initial_condition == None:
        simulator = lab.simulator.Simulator(model = model, connectivity = connection,
                          coupling = coupling, integrator = integrator, monitors = monitors,
                                            stimulus=stimulation)
    else:
        simulator = lab.simulator.Simulator(model = model, connectivity = connection,
                                            coupling = coupling, integrator = integrator,
                                            monitors = monitors, initial_conditions=initial_condition,
                                            stimulus=stimulation)
    simulator.configure()
    if initial_condition == None:
        # save the initial condition
        np.save(parameter_simulation['path_result']+'/step_init.npy',simulator.history.buffer)
        # end edit
    return simulator

def print_dict_differences(dict1, dict2):
    # Iterate through keys of the first dictionary
    for key in dict1:
        # Check if the key exists in the second dictionary
        if key in dict2:
            # Check if the values are different
            if dict1[key] != dict2[key]:
                print(f"Difference in key '{key}':")
                print(f"   New dictionary value: {dict1[key]}")
                print(f"   Preexisting dictionary value: {dict2[key]}")
            else:
                no_diff = True
        else:
            print(f"Key '{key}' not found in preexisting dictionary")
            no_diff=False
    # Check for keys in the second dictionary not present in the first
    for key in dict2:
        if key not in dict1:
            print(f"Key '{key}' not found in new dictionary")
            no_diff=False
    return no_diff

def compare_dicts(dict1, dict2):
    """
    Compare two dictionaries for equality.
    """
    if len(dict1) != len(dict2):
        return False
    
    for key in dict1:
        if key not in dict2:
            return False
        if dict1[key] != dict2[key]:
            return False
    
    return True

def run_simulation_all(parameters, b_e = 5, tau_e = 5.0, tau_i = 5.0, Iext = 0.000315, 
                      stimval = 0,stimdur = 50,stimtime_mean = 2500 ,stim_region = 5, n_nodes=68, 
                      cut_transient=2000, run_sim=5000, nseed=10, additional_path_folder=''):
                      
    #                   parameters, b_e = b_e, tau_e = tau_e , n_nodes=Nnodes,
    # cut_transient=cut_transient, run_sim=run_sim, additional_path_folder='Bold/', time, parameter_simulation,parameter_monitor):
    '''
    run a simulation
    :param simulator: the simulator already initialize
    :param time: the time of simulation
    :param parameter_simulation: the parameter for the simulation
    :param parameter_monitor: the parameter for the monitor
    '''
    print('Adjust Parameters')
    parameters = adjust_parameters(parameters, b_e = b_e, tau_e = tau_e, tau_i = tau_i, Iext = Iext, stimval = stimval,
                                   stimdur = stimdur,stimtime_mean = stimtime_mean ,stim_region = stim_region, n_nodes=n_nodes, 
                      cut_transient=cut_transient, run_sim=run_sim, nseed=nseed, additional_path_folder=additional_path_folder)
    
    print('Initialize Simulator')
    simulator = sim_init(parameters)
    
    print('Start Simulation')
    parameter_simulation,parameter_monitor= parameters.parameter_simulation, parameters.parameter_monitor
    time=run_sim

    if stimval:
        print ('    Stimulating for {1} ms, {2} nS in the {0}\n'.format(simulator.connectivity.region_labels[stim_region],parameters.parameter_stimulus['tau'],stimval))

    nb_monitor = parameter_monitor['Raw'] + parameter_monitor['TemporalAverage'] + parameter_monitor['Bold'] + parameter_monitor['Ca']
    if 'Afferent_coupling' in parameter_monitor.keys() and parameter_monitor['Afferent_coupling']:
        nb_monitor+=1
    # initialise the variable for the saving the result
    save_result =[]
    for i in range(nb_monitor):
        save_result.append([])
    # run the simulation
    count = 0
    for result in simulator(simulation_length=time):
        for i in range(nb_monitor):
            if result[i] is not None:
                save_result[i].append(result[i])
        #save the result in file
        if result[0][0] >= parameter_simulation['save_time']*(count+1): #check if the time for saving at some time step
            print('simulation time :'+str(result[0][0])+'\r')
            np.save(parameter_simulation['path_result']+'/step_'+str(count)+'.npy',np.array(save_result, dtype='object'), allow_pickle = True)
            save_result =[]
            for i in range(nb_monitor):
                save_result.append([])
            count +=1
    
    # save the last part
    np.save(parameter_simulation['path_result']+'/step_'+str(count)+'.npy',np.array(save_result, dtype='object') , allow_pickle = True)
    if count < int(time/parameter_simulation['save_time'])+1:
        np.save(parameter_simulation['path_result']+'/step_'+str(count+1)+'.npy',np.array([], dtype='object'))
    
    clear_output(wait=True)
    print(f"Simulation Completed successfully", flush=True)

def butter_bandpass(lowend, highend, TR, order=5):
        fnq = 0.5/ TR 
        Wn = [lowend / fnq, highend / fnq]
        sos = butter(order,Wn, analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, TR, order=5):
        sos = butter_bandpass(lowcut, highcut, TR, order=order)
        y = sosfilt(sos, data)
        return y

def bandpass_timeseries(ts, TR, lowend=0.04, highend=0.07, order = 5):
    """
    Bandpass BOLD signal timeseries

    Parameters:
    - ts: BOLD signal data with shape (N, T)
    - TR: repetition time in seconds
    - lowend: lowest frequency in Hz; default 0.008
    - highend: highest frequency in Hz; default 0.09

    Returns:
    - filtered_timeseries: bandpass-filtered timeseries with shape (N, T)
    """
    filtered_timeseries = np.zeros_like(ts, dtype=float)

    
    for ROI in range(ts.shape[0]):
        filtered_timeseries[ROI, :] = butter_bandpass_filter(ts[ROI, :], lowend, highend, TR, order=order) 
    
    return filtered_timeseries

def preprocess_bold(ts, TR, apply_bandpass_YN=True):
    """
    ts: time series in the shape (nodes, time_points)
    """
    # Get the dimensions of the BOLD signal data
    # ts=ts.T
    T,N= ts.shape
    print(T,N)
    # Step 1 - Narrowband filter
    if apply_bandpass_YN:
        ts = bandpass_timeseries(ts, TR)

    # Step 2 - Z-score
    Zscored_timeseries = np.zeros_like(ts)
    for roi in range(N):
        Zscored_timeseries[roi, :] = zscore(ts[roi, :]) 
        # Zscored_timeseries[roi, :] =(ts[roi, :] - np.mean(ts[roi, :])) / np.std(ts[roi, :])
    return Zscored_timeseries

def corr_sc_fc(BOLD_signal, TR, SC):

    signal = preprocess_bold(BOLD_signal, TR, apply_bandpass_YN=False) #check the shape, maybe you need to T
    # signal= BOLD_signal
    FC=np.corrcoef(signal.T)

    pearson_FCSC = np.corrcoef(FC.flatten(), SC.flatten()) #this will be a matrix
    coef = pearson_FCSC[0, 1]

    return FC, coef

def plot_FC_SC(parameters,params, result,  for_explan, cut_transient, run_sim,SC, var_select = 'E', monitor = 'Bold', change_of='tau_e', 
               Iext = 0.000315,nseed=10, additional_path_folder='',figsize=(8,5), desired_time=0):
    """
    desired_time: in s, plot from a specific time point
    """
    rows =int(len(params))
    cols = 2 
    # simulator = sim_init(parameters)
    # SC=simulator.connectivity.weights
    fig, axes = plt.subplots(rows,cols,figsize=figsize)
    
    for param in enumerate(params):
        #load results for the params
        (i, [b_e, tau]) = param 
        result_fin,TR = create_dicts(parameters,param, result, monitor, for_explan, var_select, change_of=change_of, Iext = Iext,
                                  nseed=nseed, additional_path_folder=additional_path_folder, return_TR=True)
        
        TR = TR*1e-3
        time_s = np.linspace(cut_transient, run_sim,result_fin[var_select].shape[0])/1000
        closest_index = np.argmin(np.abs(time_s - desired_time)) 
        bold_sig = result_fin[var_select][closest_index:]

        #Take FC and coeff
        FC, coef = corr_sc_fc(bold_sig, TR, SC)

        #create list with the indices for each variable

        # for var in var_select:
        if len(axes.shape) ==1:
            ax_index_fc =i
            ax_index_sc = 1
            im1 = axes[ax_index_fc].imshow(FC, cmap = "seismic", vmin = -1, vmax = 1, 
                   interpolation = 'nearest', origin='lower');
            im2 = axes[ax_index_sc].imshow(SC, cmap = "jet");
            
        else: 
            ax_index_fc= (i, 0)#that means that you have one row or one col
            ax_index_sc= (i, 1)
            im1 = axes[ax_index_fc].imshow(FC, cmap = "seismic", vmin = -1, vmax = 1, 
                   interpolation = 'nearest', origin='lower');
            im2 = axes[ax_index_sc].imshow(SC, cmap = "jet");
        
        if i ==0:
            axes[ax_index_fc].set_title("FC of Simulated Bold\n"+change_of+ f'= {tau} ms, b_e={b_e}' , fontsize=13)
            axes[ax_index_sc].set_title('Structural Connactivity'+f'\nPcoef_SCFC={round(coef,3)}', fontsize=13)
        else:
            axes[ax_index_fc].set_title(change_of+ f'= {tau} ms, b_e={b_e}' , fontsize=13)
            axes[ax_index_sc].set_title(f'\nPcoef_SCFC={round(coef,3)}', fontsize=13)
        fig.colorbar(im1, ax=axes[ax_index_fc])
        fig.colorbar(im2, ax=axes[ax_index_sc])
    plt.tight_layout()
    plt.show()

def calculate_survival_time(bvals, tau_values, tau_i_iter, Nseeds, save_path ='./network_sims/', 
                            BIN = 5, AmpStim = 1,offset_index= 61, load_until = 399  ):
    """
    calculate the survival time

    bvals : values of b_e, list or array
    tau_values : values of tau_e (then tau_i_iter=False) or tau_i (then tau_i_iter=True)
    tau_i_iter: (bool) True for iterating the tau_i
    Nseeds: list or array
    path: where the network sims were saved
    BIN: int,  used for the saving / binning of network simulations
    AmpStim: float, the amplitude of the kick
    offset_index: int, time that the stimulus stops : (time_of_peek + plateau + BIN)/BIN
    load_until: int, to make sure that you load the same length of all the arrays (here duration/BIN)

    the result is an array with shape (tau_vals, bvals) which contains the average survival time for
    each combination of tau/b_e

    original values for the script:
    bvals = np.arange(0,30,step=1)
    tauEv = np.arange(5.,7.,step=10)
    tauIv= np.arange(3.,9.,step=0.1)
    Nseeds = np.arange(0,100,5)
    """
    if tau_i_iter:
        tauv = tau_values
        tau_str = 'tau_i'
    else:
        tauv = tau_values
        tau_str = 'tau_e'


    allseeds = []

    for nseed in Nseeds:
        print("Seed = ", nseed)
        dur_dead_tau = []

        for tau in tauv:
            print(f"loop of {tau_str} = ", tau)
            dur_dead = []

            for b_ad in bvals:

                if tau_i_iter:
                    tau_E = 5.0
                    tau_I = tau
                else:
                    tau_E = tau
                    tau_I = 5.0

                sim_name = f"b_{b_ad}_tau_i_{round(tau_I,1)}_tau_e_{round(tau_E,1)}_ampst_{AmpStim}_seed_{nseed}"
                name_exc = save_path + '/network_sims/' + sim_name + '_exc.npy'
                name_inh = save_path + sim_name + '_inh.npy'

                #Using the exc FR but can use the inh instead
                try:
                    popRateG_exc = np.load(name_exc)[:load_until]
                except FileNotFoundError:
                    try:
                        sim_name = f"b_{float(b_ad)}_tau_i_{round(tau_I,1)}_tau_e_{round(tau_E,1)}_ampst_{AmpStim}_seed_{float(nseed)}"
                        name_exc = save_path + '/network_sims/' + sim_name + '_exc.npy'
                        popRateG_exc = np.load(name_exc)[:load_until]
                    except FileNotFoundError:
                        sim_name = f"b_{float(b_ad)}_tau_i_{round(tau_I,1)}_tau_e_{round(tau_E,1)}_ampst_{AmpStim}_seed_{nseed}"
                        name_exc = save_path + '/network_sims/' + sim_name + '_exc.npy'
                        popRateG_exc = np.load(name_exc)[:load_until]

                # popRateG_inh = np.load(path + name_inh)[:load_until]
                
                thresh = popRateG_exc[offset_index] / 10
                consecutive_count = sum(1 for value in popRateG_exc[offset_index:] if value > thresh)
                dur_until_dead = consecutive_count * BIN  # bin=5ms

                dur_dead.append(dur_until_dead)

            dur_dead_tau.append(dur_dead)

        dead_dur = np.array(dur_dead_tau).T
        allseeds.append(dead_dur)

    allseeds_arr = np.array(allseeds)

    mean_array = np.mean(allseeds_arr, axis=0)


    np.save(save_path + f"{tau_str}_mean_array.npy", mean_array)
    np.save(save_path + f"{tau_str}_heatmap_bvals.npy", bvals)
    np.save(save_path + f"{tau_str}_heatmap_taus.npy", tauv)

    clear_output(wait=False)
    print("Done! Saved in :", save_path)

def load_survival( load = 'tau_e', precalc=False, save_path = './'):
    if precalc:
        if load == 'tau_e':
            mean_array = np.load('./dynamical_precalc/mean_array_tau_e.npy')
            taus = list(np.load('./dynamical_precalc/taues_bcrit.npy'))
            bthr = list(np.load('./dynamical_precalc/bthr_taues_bcrit.npy'))

            tau_v = np.arange(3.5,7.0,0.1)
            bvals = np.arange(0,30,1)
        elif load == 'tau_i':
            mean_array = np.load('./dynamical_precalc/mean_array_tau_i.npy')
            taus = list(np.load('./dynamical_precalc/tauis_bcrit.npy'))
            bthr = list(np.load('./dynamical_precalc/bthr_tauis_bcrit.npy'))

            bvals = np.arange(0,25,1)
            tau_v = np.arange(3.,9.,0.1)
    else:
        mean_array = np.load(save_path + f'{load}_mean_array.npy')
        bthr = list(np.load(save_path + f'b_thresh_{load}.npy')[:,-1])
        tau_v = np.load(save_path + f'{load}_heatmap_taus.npy')
        bvals = np.load(save_path + f'{load}_heatmap_bvals.npy')
        if load == 'tau_e':
            taus = list(np.load(save_path + f'b_thresh_{load}.npy')[:,0])
        if load == 'tau_i':
            taus = list(np.load(save_path + f'b_thresh_{load}.npy')[:,1])
            taus = [i for i in taus if i <= tau_v.max()]
    return mean_array,taus, bthr, tau_v, bvals

def plot_heatmap_survival(mean_array, tauis, tau_v, bvals , bthr, load ,file_path = './' , precalc =False, save_im=False):
    
    if load == 'tau_i': 
        colorscale = [ [0, 'black'], [400/1000, 'royalblue'],[1000/1000, 'white'],[1, 'white']]
        x_heat = tau_v
        title_fig ='τᵢ (ms)'
        if precalc: 
            x_trace=tauis[17:]  
            y_trace=bthr[17:]
            x_ticks = 12
            y_ticks = 12
        else:
            x_trace=tauis  
            y_trace=bthr
            x_ticks = int(len(x_trace))
            y_ticks = int(len(y_trace))                
    elif load=='tau_e':
        colorscale = 'hot'
        x_heat = tau_v
        title_fig='τₑ (ms)'
        if precalc:
            x_trace = tauis[3:-15]
            y_trace = bthr[3:-15]
            x_ticks = 16
            y_ticks = 10
        else:
            x_trace=tauis  
            y_trace=bthr
            x_ticks = int(len(x_trace))
            y_ticks = int(len(y_trace)) 

    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Heatmap(
            z=mean_array, zmin=mean_array.min(), zmax=1000,
        x = x_heat, y=bvals, 
            colorscale=colorscale, colorbar=dict(tickfont=dict(size=19), tickcolor='black')))

    fig.add_trace(go.Scatter(
        mode='lines', 
        x=x_trace, 
        y=y_trace,
        line=dict(color='white',width=2)), secondary_y=False,)

    matplotlib_figsize = (6, 3.5)
    inch_to_pixels = 80

    plotly_width = matplotlib_figsize[0] * inch_to_pixels
    plotly_height = matplotlib_figsize[1] * inch_to_pixels


    fig.update_layout( 
            height = plotly_height, width = plotly_width,
            xaxis=dict(
                title=title_fig,
                showgrid=False,
                tickfont=dict(size=19, color='black'),  # Customize tick label fontsize
                nticks=x_ticks),
            yaxis=dict(
                title='bₑ (pA)',  # Update the y-axis title
                tickfont=dict(size=19, color='black'),  # Customize tick label fontsize
                nticks=y_ticks),
            coloraxis_colorbar_title_side="right",
            coloraxis_colorbar_x=-0.55,  # Change the horizontal position (0 to 1)
            coloraxis_colorbar_y=0.35
                        )

    fig.update_xaxes(title_text=title_fig,title_font=dict(size=20,color='black', family='Arial, sans-serif'), showgrid=False)
    fig.update_yaxes(title_text='bₑ (pA)',title_font=dict(size=20,color='black', family='Arial, sans-serif'), showgrid=False)


    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),  # Set all margins to 0 to remove white space
    )
    custom_margins = dict(l=0, r=0, t=0.1, b=0)

    if save_im:
        pio.write_image(fig, file_path)

    fig.show()