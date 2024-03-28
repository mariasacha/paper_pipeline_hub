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
G_inh = NeuronGroup(1, eqs, threshold='v > -20*mV', reset='v = -55*mV; w += b_inh', refractory='5*ms', method='heun')
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
G_exc = NeuronGroup(1, eqs, threshold='v > -20.0*mV', reset='v = -50*mV; w += b_exc', refractory='5*ms',  method='heun')
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

P_re_inh = PoissonGroup(N_inh, rates=4*Hz) # RE pop
P_ed_exc = PoissonGroup(N_ped, rates=30*Hz) # external pop
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

# Useful trick to record global variables ------------------------------------------------------

Gw_inh = NeuronGroup(1, 'Wtot : ampere', method='rk4')
Gw_exc = NeuronGroup(1, 'Wtot : ampere', method='rk4')

SwInh1=Synapses(G_inh, Gw_inh, 'Wtot_post = w_pre : ampere (summed)')
SwInh1.connect(p=1)
SwExc1=Synapses(G_exc, Gw_exc, 'Wtot_post = w_pre : ampere (summed)')
SwExc1.connect(p=1)

MWinh = StateMonitor(Gw_inh, 'Wtot', record=0)
MWexc = StateMonitor(Gw_exc, 'Wtot', record=0)



GV_inh = NeuronGroup(1, 'Vtot : volt', method='rk4')
GV_exc = NeuronGroup(1, 'Vtot : volt', method='rk4')

SvInh1=Synapses(G_inh, GV_inh, 'Vtot_post = v_pre : volt (summed)')
SvInh1.connect(p=1)
SvExc1=Synapses(G_exc, GV_exc, 'Vtot_post = v_pre : volt (summed)')
SvExc1.connect(p=1)

MVinh = StateMonitor(GV_inh, 'Vtot', record=0)
MVexc = StateMonitor(GV_exc, 'Vtot', record=0)


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



# np.save('Wtot.npy',[np.array(MWinh.Wtot[0]/mamp),np.array(MWexc.Wtot[0]/mamp)])

# np.save('Vtot.npy',[np.array(MVinh.Vtot[0]/mV),np.array(MVexc.Vtot[0]/mV)])


plt.plot(np.array(MVexc.Vtot[0]/mV),c='b')
plt.plot(np.array(MVinh.Vtot[0]/mV),c='r')
# plt.plot(np.array(MWexc.Wtot[0]/mamp),c='b')
# plt.plot(np.array(MWinh.Wtot[0]/mamp),c='r')
plt.show()