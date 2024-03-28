import matplotlib.pyplot as plt

from functions import *
import os
path = "/DATA/Maria/Anesthetics/network_sims/big_loop_tau_i/"


def heaviside(x):
    return 0.5 * (1 + np.sign(x))


def input_rate(t, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau):
    # t1_exc=10. # time of the maximum of external stimulation
    # tau1_exc=20. # first time constant of perturbation = rising time
    # tau2_exc=50. # decaying time
    # ampl_exc=20. # amplitude of excitation
    inp = ampl_exc * (np.exp(-(t - t1_exc) ** 2 / (2. * tau1_exc ** 2)) * heaviside(-(t - t1_exc)) + \
                      heaviside(-(t - (t1_exc + plateau))) * heaviside(t - (t1_exc)) + \
                      np.exp(-(t - (t1_exc + plateau)) ** 2 / (2. * tau2_exc ** 2)) * heaviside(t - (t1_exc + plateau)))
    return inp


start_scope()

DT=0.1 # time step
defaultclock.dt = DT*ms
N1 = 2000 # number of inhibitory neurons
N2 = 8000 # number of excitatory neurons 

TotTime=4000 #Simulation duration (ms)
duration = TotTime*ms


C = 200*pF
gL = 10*nS
#EL = -65*mV
#VT = -50.4*mV
#DeltaT = 2*mV
tauw = 500*ms
a =0.0*nS# 4*nS
#b = 0.08*nA
I = 0.*nA
Ee=0*mV
Ei=-80*mV

b_e = 60
tau_e = 5.0

tau_i = 5.
Iext = 0.5

EL_i = -65.0
EL_e = -64.0
#CHECK THE SIMNAME ALWAYS

seed(9) #9,11,25

#
AmpStim = 1
plat = 100
TauP=20
t2 = np.arange(0, TotTime, DT)
test_input = []

for ji in t2:
	test_input.append(0. + input_rate(ji, 200., TauP, 1, AmpStim, plat))

Input_Stim = TimedArray(test_input * Hz, dt=DT * ms)


# sim_name = create_simname_both_taus(b_e, tau_e, tau_i, Iext, EL_i, EL_e)
sim_name = f'_b_{b_e}_tau_e_{tau_e}_tau_i_{tau_i}_eli_{int(EL_i)}_ele_{int(EL_e)}_iext_{Iext}'

# folder_result = "/DATA/Maria/Anesthetics/network_sims/big_loop_tau_i/results/"
subfolder_result = r"C:\Users\maria\Documents\tvb\good_ones\results_FR\\"

# subfolder_result = folder_result + sim_name + '/'

eqs = """
dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)-GsynE*(vm-Ee)-GsynI*(vm-Ei)+I-w)/C : volt (unless refractory)
dw/dt=(a*(vm-EL)-w)/tauw : amp
dGsynI/dt = -GsynI/TsynI : siemens
dGsynE/dt = -GsynE/TsynE : siemens
TsynI:second
TsynE:second
Vr:volt
b:amp
DeltaT:volt
Vcut:volt
VT:volt
EL:volt
"""


# Population 1 - Fast Spiking

G_inh = NeuronGroup(N1, model=eqs, threshold='vm > Vcut',refractory=5*ms,
                     reset="vm = Vr; w += b", method='heun')
G_inh.vm = -60*mV#EL
G_inh.w = a * (G_inh.vm - G_inh.EL)
G_inh.Vr = -65*mV #
G_inh.TsynI =tau_i*ms
G_inh.TsynE =tau_e*ms
G_inh.b=0*pA
G_inh.DeltaT=0.5*mV
G_inh.VT=-50.*mV
# G_inh.Vcut=G_inh.VT + 5 * G_inh.DeltaT
G_inh.Vcut=-30*mV
G_inh.EL=EL_i*mV


# Population 2 - Regular Spiking

G_exc = NeuronGroup(N2, model=eqs, threshold='vm > Vcut',refractory=5*ms,
                     reset="vm = Vr; w += b", method='heun')
G_exc.vm = -60*mV#EL
G_exc.w = a * (G_exc.vm - G_exc.EL)
G_exc.Vr = -65*mV 
G_exc.TsynI =tau_i*ms
G_exc.TsynE =tau_e*ms
G_exc.b=b_e*pA
G_exc.DeltaT=2*mV
G_exc.VT=-50.*mV
# G_exc.Vcut=G_exc.VT + 5 * G_exc.DeltaT
G_exc.Vcut = -30*mV
G_exc.EL=EL_e*mV

# external drive--------------------------------------------------------------------------

P_ed=PoissonGroup(N2, rates=Iext*Hz)
# P_ed = PoissonGroup(N2, rates='Input_Stim(t)')  # P_ed=PoissonGroup(N2, rates=0.315*Hz)

# Network-----------------------------------------------------------------------------

# connections-----------------------------------------------------------------------------
#seed(0)
try:
    Qi=5.0*nS
    Qe=1.5*nS

    prbC=.05 #0.05
 

    S_12 = Synapses(G_inh, G_exc, on_pre='GsynI_post+=Qi') #'v_post -= 1.*mV')
    S_12.connect('i!=j', p=prbC)

    S_11 = Synapses(G_inh, G_inh, on_pre='GsynI_post+=Qi')
    S_11.connect('i!=j',p=prbC)

    S_21 = Synapses(G_exc, G_inh, on_pre='GsynE_post+=Qe')
    S_21.connect('i!=j',p=prbC)

    S_22 = Synapses(G_exc, G_exc, on_pre='GsynE_post+=Qe')
    S_22.connect('i!=j', p=prbC)

    S_ed_in = Synapses(P_ed, G_inh, on_pre='GsynE_post+=Qe')
    S_ed_in.connect(p=prbC)

    S_ed_ex = Synapses(P_ed, G_exc, on_pre='GsynE_post+=Qe')
    S_ed_ex.connect(p=prbC)


    PgroupE = NeuronGroup(1, 'P:amp', method='heun')

    PE=Synapses(G_exc, PgroupE, 'P_post = w_pre : amp (summed)')
    PE.connect(p=1)
    P2mon = StateMonitor(PgroupE, 'P', record=0)


    PgroupMuVE = NeuronGroup(1, 'Pv:volt', method='heun')

    PmuE=Synapses(G_exc, PgroupMuVE, 'Pv_post = vm_pre : volt (summed)')
    PmuE.connect(p=1)
    P2MuVemon = StateMonitor(PgroupMuVE, 'Pv', record=0)


    # Recording tools -------------------------------------------------------------------------------
    rec1=2000
    rec2=8000

    M1G_inh = SpikeMonitor(G_inh)
    FRG_inh = PopulationRateMonitor(G_inh)
    M1G_exc = SpikeMonitor(G_exc)
    FRG_exc = PopulationRateMonitor(G_exc)

    # M2G1 = StateMonitor(G_inh, 'vm', record=range(rec1))
    # M3G1 = StateMonitor(G_inh, 'w', record=range(rec1))
    # M4G1 = StateMonitor(G_inh, 'GsynE', record=range(rec1))
    # M5G1 = StateMonitor(G_inh, 'GsynI', record=range(rec1))

    # M2G2 = StateMonitor(G_exc, 'vm', record=range(rec2))
    # M3G2 = StateMonitor(G_exc, 'w', record=range(rec2))
    # M4G2 = StateMonitor(G_exc, 'GsynE', record=range(rec2))
    # M5G2 = StateMonitor(G_exc, 'GsynI', record=range(rec2))


    # Run simulation -------------------------------------------------------------------------------

    print('--##Start simulation##--')
    print(f"b_e = {b_e}, tau_e = {tau_e}, Iext= {Iext} , eli_{EL_i}_ele_{EL_e}")
    run(duration)
    print('--##End simulation##--')


    # Plots -------------------------------------------------------------------------------

    # average_membrane_potential_exc = M2G2.vm.mean()
    # average_membrane_potential_inh = M2G1.vm.mean()


    # prepare raster plot
    RasG_inh = array([M1G_inh.t/ms, [i+N2 for i in M1G_inh.i]])
    RasG_exc = array([M1G_exc.t/ms, M1G_exc.i])
    TimBinned, popRateG_exc, popRateG_inh, Pu = prepare_FR(TotTime,DT, FRG_exc, FRG_inh, P2mon)

    # np.save(subfolder_result + sim_name + "_m2g1.npy", M2G1.vm)
    # np.save(subfolder_result + sim_name + "_m2g2.npy", M2G2.vm)
    # np.save(sim_name + "_frexc.npy", popRateG_exc)
    # sp_tr = M1G_exc.spike_trains()
    
    # import pickle
    
    # with open(subfolder_result + sim_name + '_spike_train.pkl', 'wb') as fp:
    #     pickle.dump(sp_tr, fp)
    #     print('dictionary saved successfully to file')

    # save
    
    # try:
    #     os.listdir(subfolder_result)
    # except:
    #     os.makedirs(subfolder_result)
    SAVE = True
    if SAVE:
        # #HERE
        # subfolder_result = "/DATA/Maria/Anesthetics/figures_analysis_dur/results_FRs/"
        filename_1 = subfolder_result + sim_name + '_poprate_exc_kick.npy'
        np.save(filename_1, np.array(popRateG_exc))

        filename_2 = subfolder_result + sim_name + '_poprate_inh_kick.npy'
        np.save(filename_2, np.array(popRateG_inh))



        filename_3 = subfolder_result + sim_name + '_pu.npy'
        np.save(filename_3, np.array(Pu))
        
        filename_4 = subfolder_result + sim_name + '_ras_inh.npy'
        np.save(filename_4, np.array(RasG_inh))
        
        filename_5 = subfolder_result + sim_name + '_ras_exc.npy'
        np.save(filename_5, np.array(RasG_exc))
    
    
    # # HERE
    # input_bin = bin_array(np.array(test_input), 5, np.arange(int(TotTime / DT)) * DT)
    # filename_6 = subfolder_result + sim_name + '_input_bin.npy'
    # np.save(filename_6, input_bin)

    # filename_7 = subfolder_result + sim_name + '_timbin_kick.npy'
    # np.save(filename_7, TimBinned)


    # # ----- Raster plot + mean adaptation ------

    plot_raster_meanFR_tau_i(RasG_inh, RasG_exc, TimBinned, popRateG_inh, popRateG_exc, Pu, sim_name, b_e,tau_e, tau_i, EL_i,
                             EL_e, Iext, path)

    
#     fig=figure(figsize=(9,7.5))
#     ax1=fig.add_subplot(211)
#     ax3=fig.add_subplot(212)
#     ax2 = ax3.twinx()
#
# ###################### --- RASTER PLOT --- ######################
#     ax1.plot(RasG_inh[0], RasG_inh[1], ',r')
#     ax1.plot(RasG_exc[0], RasG_exc[1], ',g')
#
#     ax1.set_xlabel('Time (ms)')
#     ax1.set_ylabel('Neuron index')
#     ##################################################################
#
#     ax3.plot(TimBinned/1000,popRateG_inh, 'r')
#     ax3.plot(TimBinned/1000,popRateG_exc, 'SteelBlue')
#     ax2.plot(TimBinned/1000,(Pu/8000), 'orange')
#
#     # input_bin = bin_array(np.array(test_input), 5, np.arange(int(TotTime / DT)) * DT)
#     # ax3.plot(TimBinned / 1000, input_bin, 'green')
#
#     ax2.set_ylabel('mean w (pA)')
#     #ax2.set_ylim(0.0, 0.045)
#     ax3.set_xlabel('Time (s)')
#     ax3.set_ylabel('population Firing Rate')
#
#     # fig.suptitle(f'b_e={b_e}, tau_e={tau_e}, tau_i={tau_i}, EL_i = {EL_i}, EL_e = {EL_e}, Iext = {Iext}')
#
#     #/DATA/Maria/Anesthetics/network_sims/big_loop_tau_i/figures/eli_-64.0_ele_-63.0/Iext_0.4/ +sim_name
#     fol_name = path + "figures/" + f"eli_{int(EL_i)}_ele_{int(EL_e)}/Iext_{Iext}/"
#
#     try:
#         os.listdir(fol_name)
#     except:
#         os.makedirs(fol_name)
#
#     # fig.savefig(fol_name + sim_name + '.png')
#     plt.tight_layout()
#     plt.show()

    print(f"{sim_name} done")
    plt.show()
except Exception:
    print("ERROR")
    folder_indicator = "/DATA/Maria/Anesthetics/network_sims/big_loop/errors/"
    sim_name_2 = f'_b_{b_e}_tau_e_{tau_e}_Iext_{Iext}_eli_{EL_i}_ele_{EL_e}'
    ind_filename = folder_indicator + sim_name_2 + '_Error.txt'
    open(ind_filename, 'a').close()




