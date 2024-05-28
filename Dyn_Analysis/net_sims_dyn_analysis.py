import numpy as np
from brian2 import *
import os
from functions import *
import argparse 

start_scope()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#range of values
parser.add_argument('--b_e_range', type=get_np_arange , default='0,30,1', help='b_e range of values')
parser.add_argument('--tau_e_range', type=get_np_arange , default='5.,7.,10', help='tau_e range of values \
					- if you iterate tau_i then set  tau_e_range=np.arange(5.,10.,step=500)')
parser.add_argument('--tau_i_range', type=get_np_arange , default='3.,9.,0.1', help='tau_i range of values \
					- if you iterate tau_e then set tau_i_range=np.arange(5.,10.,step=500)')
parser.add_argument('--nseeds', type=get_np_arange , default='0,100,5', help='seed values')

#Other
parser.add_argument('--time', type=float, default=2000, help='Total Time of simulation (ms)')
parser.add_argument('--save_path', type=str, default='/trials', help='Path to save the results of the simulations')
parser.add_argument('--overwrite', type=bool, default=False, help='If True it will overwrite the existant paths')
parser.add_argument('--surv_time_calc', type=bool, default=False, help='If True calculate the survival time and save it')

args = parser.parse_args()

surv_time_calc = args.surv_time_calc
TotTime = args.time
duration = TotTime*ms
save_path = "./Dyn_Analysis" +  args.save_path + "/"
OVERWRITE = args.overwrite
BIN=5
#Choose the values of the scan
tauIv=args.tau_i_range
tauEv=args.tau_e_range
bvals = args.b_e_range
Nseeds= args.nseeds
if len(tauIv) > 1 and len(tauEv)>1:
	print("Iterations for both tau_e and tau_i")
	raise Exception("Change the tau_e_range or tau_i_range ")

 
#Create the kick
DT=0.1
AmpStim = 1
plat = 100
TauP=20
t2 = np.arange(0, TotTime, DT)
test_input = []
time_peek = 200.
for ji in t2:
	test_input.append(0. + input_rate(ji, time_peek, TauP, 1, AmpStim, plat))
Input_Stim = TimedArray(test_input * Hz, dt=DT * ms)


DT=0.1 # time step
defaultclock.dt = DT*ms
N1 = 2000 # number of inhibitory neurons
N2 = 8000 # number of excitatory neurons

#Parameters of network
C = 200*pF
gL = 10*nS
tauw = 500*ms
a =0.0*nS# 4*nS
I = 0.*nA
Ee=0*mV
Ei=-80*mV
eli=-65
ele=-64

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

ii=0


#Start the loop
combinations = [(nseed, tau_I, b_ad, tau_E) for b_ad in bvals for tau_I in tauIv for tau_E in tauEv for nseed in Nseeds]

for nseed, tau_I, b_ad, tau_E in combinations:
	print(f"b_e={b_ad}, tau_e={tau_E}, tau_i={tau_I}, seed={nseed}")

	seed(int(nseed))

	sim_name = f"b_{b_ad}_tau_i_{round(tau_I,1)}_tau_e_{round(tau_E,1)}_ampst_{AmpStim}_seed_{nseed}"
	str1 =  save_path + 'network_sims/' +sim_name + '_exc.npy'
	str2 =  save_path + 'network_sims/' +sim_name + '_inh.npy'

	try:
		os.listdir(save_path+'network_sims/')
	except:
		os.makedirs(save_path+'network_sims/')

	#check if the path exists already
	if os.path.exists(str1) and os.path.exists(str2) and not OVERWRITE:
		print(str1, ": exists")
		continue
	else:
		pass
	# Population 1 - Fast Spiking

	G_inh = NeuronGroup(N1, model=eqs, threshold='vm > Vcut',refractory=5*ms,
					reset="vm = Vr; w += b", method='heun')
	G_inh.vm = -60*mV#EL
	G_inh.w = a * (G_inh.vm - G_inh.EL)
	G_inh.Vr = -65*mV #
	G_inh.TsynI =tau_I*ms#5.0*ms
	G_inh.TsynE =tau_E*ms#5.0*ms
	G_inh.b=0*pA
	G_inh.DeltaT=0.5*mV
	G_inh.VT=-50.*mV
	G_inh.Vcut=G_inh.VT + 5 * G_inh.DeltaT
	G_inh.EL=eli*mV#-65*mV#-67*mV


	# Population 2 - Regular Spiking

	G_exc = NeuronGroup(N2, model=eqs, threshold='vm > Vcut',refractory=5*ms,
					reset="vm = Vr; w += b", method='heun')
	G_exc.vm = -60*mV#EL
	G_exc.w = a * (G_exc.vm - G_exc.EL)
	G_exc.Vr = -65*mV
	G_exc.TsynI =tau_I*ms#5.0*ms
	G_exc.TsynE =tau_E*ms#5.0*ms
	G_exc.b=b_ad*pA#0*pA#60*pA
	G_exc.DeltaT=2*mV
	G_exc.VT=-50.*mV
	G_exc.Vcut=G_exc.VT + 5 * G_exc.DeltaT
	G_exc.EL=ele*mV#-65*mV#-63*mV

	# external drive--------------------------------------------------------------------------

	P_ed=PoissonGroup(N2, rates='Input_Stim(t)')#P_ed=PoissonGroup(N2, rates=0.315*Hz)

	# Network-----------------------------------------------------------------------------

	# connections-----------------------------------------------------------------------------
	#seed(0)
	Qi=5.0*nS
	Qe=1.5*nS

	prbC=.05 #0.05
	prbC = 500/(N1 + N2)

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


	# Recording tools -------------------------------------------------------------------------------


	FRG_inh = PopulationRateMonitor(G_inh)
	FRG_exc = PopulationRateMonitor(G_exc)


	# Run simulation -------------------------------------------------------------------------------

	#print('--##Start simulation##--')
	run(duration)
	#print('--##End simulation##--')

	# Plots -------------------------------------------------------------------------------
	time_array = arange(int(TotTime/DT))*DT


	LfrG_exc=array(FRG_exc.rate/Hz)
	TimBinned,popRateG_exc=bin_array(time_array, BIN, time_array),bin_array(LfrG_exc, BIN, time_array)

	LfrG_inh=array(FRG_inh.rate/Hz)
	TimBinned,popRateG_inh=bin_array(time_array, BIN, time_array),bin_array(LfrG_inh, BIN, time_array)

	np.save(str1,popRateG_exc )
	np.save(str2,popRateG_inh )

#Save also the time_array
if os.path.exists(save_path + 'network_sims/'+ sim_name + '_time.npy') and not OVERWRITE:
	print("binned time array exists")
	pass
else:
	np.save(save_path + 'network_sims/'+ sim_name + '_time.npy', TimBinned)

if surv_time_calc:
	print("calculating survival times")

	offset_index = int((plat + time_peek + BIN)/BIN)
	load_until = int((TotTime/BIN)-BIN)

	if len(tauEv)==1 and len(tauIv)>1:
		tau_i_iter = True
		tau_values = tauIv
		tau_str = "tau_i"
	elif len(tauEv)>1 and len(tauIv)==1:
		tau_i_iter = False
		tau_values = tauEv
		tau_str = "tau_e"

	calculate_survival_time(bvals, tau_values, tau_i_iter, Nseeds, save_path=save_path, 
                            BIN = BIN, AmpStim = AmpStim, offset_index= offset_index, load_until = load_until)

clear_output(wait=False)
print(f"Done! Network simulations are saved in {save_path}/network_sims/ \nsurvival time ({tau_str}_mean_array.npy) \nwith the respective values of b_e ({tau_str}_heatmap_bvals.npy) and \n{tau_str}s ({tau_str}_heatmap_taus.npy) in {save_path}")