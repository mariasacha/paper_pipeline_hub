from brian2 import *
import os
import scipy.signal as signal
from numba import njit
import itertools

# prepare firing rate
def bin_array(array, BIN, time_array):
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)

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

def plot_raster_meanFR(RasG_inh,RasG_exc, TimBinned, popRateG_inh, popRateG_exc, Pu, axes):
    
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

    ax_all = [ax2, ax3]
    for axi in ax_all:
        axi.legend()
    
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