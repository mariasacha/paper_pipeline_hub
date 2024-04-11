import matplotlib.pyplot as plt
from brian2 import *
import time
start_scope()
start_time = time.time()

DT=0.1 # time step
defaultclock.dt = DT*ms

TotTime=5000 #Simulation duration (ms)
duration = TotTime*ms


FRout_inh=[]
FRout_exc=[]
muve=[]
muvi=[]

Adapt=[]
Npts=60 #Resolution of the measured TF
Npts2=60
i=0
finh = np.array([0.6779661 , 1.3559322 , 2.03389831, 2.71186441, 3.38983051,
       4.06779661, 4.74576271, 5.42372881, 6.10169492, 6.77966102,
       7.45762712, 8.13559322, 8.81355932, 9.49152542])

for rate_exc in linspace(0.1,10, 30):
	print("rate exc =", rate_exc)
	FRout_inh.append([])
	FRout_exc.append([])
	Adapt.append([])
	muve.append([])	
	muvi.append([])
	for rate_inh in finh:
		print("rate inh =", rate_inh)
		#(0.04*v**2+5*v+140-u-GsynE*(v-Ee)-GsynI*(v-Ei)-I)/Tn:1
		eqs="""
		dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)-GsynE*(vm-Ee)-GsynI*(vm-Ei)+Is-w)/Cm :volt (unless refractory)
		dw/dt=(a*(vm-EL)-w)/tauw : amp
		dGsynI/dt = -GsynI/TsynI : siemens
		dGsynE/dt = -GsynE/TsynE : siemens
		TsynI:second
		TsynE:second
		Is:amp
		Cm:farad
		gL:siemens
		Vr:volt
		b:amp
		DeltaT:volt
		Vcut:volt
		VT:volt
		EL:volt
		a:siemens
		tauw:second
		Dt:volt
		Ee:volt
		Ei:volt
		"""
		#Pvar:1
		#Qi:1
		#Qe:1
		#"""

		# Population 1 - Fast Spiking

		G_inh = NeuronGroup(1, eqs, threshold='vm > Vcut',refractory=5*ms, reset="vm = Vr; w += b", method='heun')
		G_inh.vm = -60 * mV  # EL
		G_inh.w = 0.*nS * (G_inh.vm - G_inh.EL)
		G_inh.Vr = -65 * mV  #
		G_inh.TsynI = 5. * ms  # 5.0*ms
		G_inh.TsynE = 4. * ms  # 5.0*ms
		G_inh.b = 0 * pA
		G_inh.DeltaT = 0.5 * mV
		G_inh.VT = -50. * mV
		G_inh.Vcut = G_inh.VT + 5 * G_inh.DeltaT
		G_inh.EL = -65 * mV  # -65*mV#-67*mV
		G_inh.GsynI=0.0*nS
		G_inh.GsynE=0.0*nS
		G_inh.Ee=0.*mV
		G_inh.Ei=-80.*mV
		G_inh.Cm = 200.*pF
		G_inh.gL = 10.*nS
		G_inh.tauw = 500*ms #1000.*ms
		G_inh.Is = 0.0*nA #2.50*nA #[0.0 for i in range(N2)]*nA


		# Population 2 - Regular Spiking
		G_exc = NeuronGroup(1, eqs, threshold='vm > Vcut', refractory=5*ms, reset="vm = Vr; w += b", method='heun')
		G_exc.vm = -60*mV#EL
		# G_exc.a = 4.0*nS
		G_exc.w = 0.*nS * (G_exc.vm - G_exc.EL)
		G_exc.Vr = -65*mV
		G_exc.TsynI =5.*ms#5.0*ms
		G_exc.TsynE =4.*ms#5.0*ms
		G_exc.b=30*pA#60*pA
		G_exc.DeltaT=2*mV
		G_exc.VT=-50.*mV
		G_exc.Vcut=G_exc.VT + 5 * G_exc.DeltaT
		G_exc.EL=-64*mV#-65*mV#-63*mV
		G_exc.GsynI=0.0*nS
		G_exc.GsynE=0.0*nS
		G_exc.Ee=0.*mV
		G_exc.Ei=-80.*mV
		G_exc.Cm = 200.*pF
		G_exc.gL = 10.*nS
		G_exc.tauw = 500*ms #1000.*ms
		G_exc.Is = 0.0*nA #2.50*nA #[0.0 for i in range(N2)]*nA
			

		# external drive--------------------------------------------------------------------------

		P_ed_inh = PoissonGroup(100, rates=rate_inh*Hz) #5% of 2000 inh cells
		P_ed_exc = PoissonGroup(400, rates=rate_exc*Hz) #5% of 8000 exc cells


		# Network-----------------------------------------------------------------------------

		# connections-----------------------------------------------------------------------------
		#seed(0)
		Qi=5.0*nS#e-6 #*nS
		Qe=1.5*nS#e-6 #*nS

		prbC=1. #1


		S_edin_in = Synapses(P_ed_inh, G_inh, on_pre='GsynI_post+=Qi')
		S_edin_in.connect(p=prbC)

		S_edin_ex = Synapses(P_ed_inh, G_exc, on_pre='GsynI_post+=Qi')
		S_edin_ex.connect(p=prbC)

		S_edex_in = Synapses(P_ed_exc, G_inh, on_pre='GsynE_post+=Qe')
		S_edex_in.connect(p=prbC)

		S_edex_ex = Synapses(P_ed_exc, G_exc, on_pre='GsynE_post+=Qe')
		S_edex_ex.connect(p=prbC)



		# Recording tools -------------------------------------------------------------------------------
		M2G2_vi = StateMonitor(G_inh, 'vm', record=0)
		M2G2_v = StateMonitor(G_exc, 'vm', record=0)
		M3G2 = StateMonitor(G_exc, 'w', record=0)
		FRG_inh = PopulationRateMonitor(G_inh)
		FRG_exc = PopulationRateMonitor(G_exc)


		# Run simulation -------------------------------------------------------------------------------

		#print('--##Start simulation##--')
		run(duration)
		#print('--##End simulation##--')


		#  -------------------------------------------------------------------------------


		# prepare firing rate

		BIN=5
		time_array = arange(int(TotTime/DT))*DT

		print(test)
		
		LfrG_exc=array(FRG_exc.rate/Hz)
		TimBinned,popRateG_exc=bin_array(time_array, BIN, time_array),bin_array(LfrG_exc, BIN, time_array)

		LfrG_inh=array(FRG_inh.rate/Hz)
		TimBinned,popRateG_inh=bin_array(time_array, BIN, time_array),bin_array(LfrG_inh, BIN, time_array)



		uu=M3G2[0].w
		vv=M2G2_v[0].vm
		vvi=M2G2_vi[0].vm
		#print(mean(popRateG_inh[150::]))
		#print(mean(popRateG_exc[150::]))
		FRout_inh[i].append(mean(popRateG_inh[int(0.8*len(popRateG_inh))::]))
		FRout_exc[i].append(mean(popRateG_exc[int(0.8*len(popRateG_exc))::]))
		Adapt[i].append(mean(uu[int(0.8*len(uu)):len(uu)]))
		muve[i].append(mean(vv[int(0.8*len(uu)):len(uu)]))
		muvi[i].append(mean(vvi[int(0.8*len(uu)):len(uu)]))


	i=i+1
	
	if i % 1 == 0:
		print(i)

#
np.save('/DATA/Maria/fede_tau/data/ExpTF_Adapt_Nstp60_tau_e_4_b_30_vol3.npy', Adapt)
np.save('/DATA/Maria/fede_tau/data/ExpTF_inh_Nstp60_tau_e_4_b_30_vol3.npy', FRout_inh)
np.save('/DATA/Maria/fede_tau/data/ExpTF_exc_Nstp60_tau_e_4_b_30_vol3.npy', FRout_exc)
np.save('/DATA/Maria/fede_tau/data/ExpTF_muve_Nstp60_tau_e_4_b_30_vol3.npy', muve)
np.save('/DATA/Maria/fede_tau/data/ExpTF_muvi_Nstp60_tau_e_4_b_30_vol3.npy', muvi)


# np.save('/DATA/Maria/fede_tau/data/lol.npy', Adapt)
# np.save('/DATA/Maria/fede_tau/data/jkj.npy', FRout_inh)
# np.save('/DATA/Maria/fede_tau/data/fg.npy', FRout_exc)
# np.save('/DATA/Maria/fede_tau/data/hjg.npy', muve)
# np.save('/DATA/Maria/fede_tau/data/tyt.npy', muvi)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")






