from brian2 import *
prefs.codegen.target = "numpy"

start_scope()

DT=0.1 # time step
defaultclock.dt = DT*ms
N_inh = 500 # number of inhibitory neurons
N_exc = 500 # number of excitatory neurons
N_ped = 8000 # external pop

TotTime=1000 #Simulation duration (ms)
duration = TotTime*ms


FRout_inh=[]
FRout_exc=[]
Npts = 40  #Resolution of the measured TF - !!! 20 is reserved !!!
i = 0
j = 1
for rate_exc in linspace(0, 40, Npts):
	FRout_inh.append([])
	FRout_exc.append([])
	for rate_inh in linspace(0, 170, Npts):
		
		eqs='''
		dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w + Is)/Cm : volt (unless refractory)
		dw/dt = (a*(v-El)-w)/tau_w:ampere
		dGsynI/dt = -GsynI/Tsyn : siemens
		dGsynE/dt = -GsynE/Tsyn : siemens
		Is:ampere
		Cm:farad
		gl:siemens
		El:volt
		a:siemens
		tau_w:second
		Dt:volt
		Vt:volt
		Ee:volt
		Ei:volt
		Tsyn:second
		'''

		# Population 1 [inhibitory] - RE - Reticular

		b_inh = 10.*pA
		G_inh = NeuronGroup(500, eqs, threshold='v > -20*mV', reset='v = -55*mV; w += b_inh', refractory='5*ms', method='heun')
		# init:
		G_inh.v = -55.*mV
		G_inh.w = 0.*pA
		# synaptic parameters
		G_inh.GsynI = 0.0*nS
		G_inh.GsynE = 0.0*nS
		G_inh.Ee = 0.*mV
		G_inh.Ei = -80.*mV
		G_inh.Tsyn = 5.*ms
		# cell parameters
		G_inh.Cm = 200.*pF
		G_inh.gl = 10.*nS
		G_inh.Vt = -45.*mV
		G_inh.Dt = 2.5*mV
		G_inh.tau_w = 200.*ms
		G_inh.Is = 0.0*nA # external input
		G_inh.El = -75.*mV
		G_inh.a = 8.0*nS


		# Population 2 [excitatory] - TC - Thalamocortical

		b_exc = 10*pA
		G_exc = NeuronGroup(500, eqs, threshold='v > -20.0*mV', reset='v = -50*mV; w += b_exc', refractory='5*ms',  method='heun')
		# init
		G_exc.v = -50.*mV
		G_exc.w = 0.*pA
		# synaptic parameters
		G_exc.GsynI = 0.0*nS
		G_exc.GsynE = 0.0*nS
		G_exc.Ee = 0.*mV
		G_exc.Ei = -80.*mV
		G_exc.Tsyn = 5.*ms
		# cell parameters
		G_exc.Cm = 160.*pF
		G_exc.gl = 10.*nS
		G_exc.Vt = -50.*mV
		G_exc.Dt = 4.5*mV
		G_exc.tau_w = 200.*ms
		G_exc.Is = 0.0*nA # ext inp
		G_exc.El = -65.*mV # -55
		G_exc.a = 0.*nS


		# external drive--------------------------------------------------------------------------

		P_re_inh = PoissonGroup(N_inh, rates=rate_inh*Hz) # RE pop
		P_ed_exc = PoissonGroup(N_ped, rates=rate_exc*Hz) # external pop
		# P_tc_exc = PoissonGroup(N_exc, rates=rate_exc*Hz) # TC pop


		# Network-----------------------------------------------------------------------------

		# quantal increment in synaptic conductances:
		Qpe = 1*nS # from P_ed to G_exc (p -> e)
		Qpi = 4*nS
		# Qei = 4*nS
		Qii = 1*nS
		Qie = 6*nS

		# probability of connection
		prbC= 0.05

		
		S_edin_ex = Synapses(P_ed_exc, G_inh, on_pre='GsynE_post+=Qpi')
		S_edin_ex.connect(p=prbC)

		# S_tcin_in = Synapses(P_tc_exc, G_inh, on_pre='GsynE_post+=Qei')
		# S_tcin_in.connect(p=prbC)

		S_reex_in = Synapses(P_re_inh, G_exc, on_pre='GsynI_post+=Qie')
		S_reex_in.connect(p=prbC)

		S_rein_in = Synapses(P_re_inh, G_inh, on_pre='GsynI_post+=Qii')
		S_rein_in.connect(p=prbC*6)

		S_edex_ex = Synapses(P_ed_exc, G_exc, on_pre='GsynE_post+=Qpe')
		S_edex_ex.connect(p=prbC*2)


		# Recording tools -------------------------------------------------------------------------------

		FRG_inh = PopulationRateMonitor(G_inh)
		FRG_exc = PopulationRateMonitor(G_exc)


		# Run simulation -------------------------------------------------------------------------------

		run(duration)

		# Plots -------------------------------------------------------------------------------



		# prepare firing rate
		def bin_array(array, BIN, time_array):
		    N0 = int(BIN/(time_array[1]-time_array[0]))
		    N1 = int((time_array[-1]-time_array[0])/BIN)
		    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)

		BIN=5
		time_array = arange(int(TotTime/DT))*DT



		LfrG_exc=array(FRG_exc.rate/Hz)
		TimBinned,popRateG_exc=bin_array(time_array, BIN, time_array),bin_array(LfrG_exc, BIN, time_array)

		LfrG_inh=array(FRG_inh.rate/Hz)
		TimBinned,popRateG_inh=bin_array(time_array, BIN, time_array),bin_array(LfrG_inh, BIN, time_array)

		#print(mean(popRateG_inh[150::]))
		#print(mean(popRateG_exc[150::]))
		FRout_inh[i].append(mean(popRateG_inh[150::]))
		FRout_exc[i].append(mean(popRateG_exc[150::]))

		print(f' {j}/{Npts**2}', end='\r')
		j=j+1
	i=i+1


np.save(f'data\\ExpTF_inh_Nstp{Npts}_new.npy', FRout_inh)
np.save(f'data\\ExpTF_exc_Nstp{Npts}_new.npy', FRout_exc)
plt.figure()
plt.imshow(FRout_inh)
plt.figure()
plt.imshow(FRout_exc)
plt.show()

print()
