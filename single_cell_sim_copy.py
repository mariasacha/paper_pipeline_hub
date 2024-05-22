from brian2 import *
import matplotlib.pyplot as plt
from functions import *
import argparse
import numpy as np

start_scope()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--type', type=str, default='RS', help='type of cell (RS or FS)')
parser.add_argument('--b_e', type=float, default=30, help='adaptation (pA)')
parser.add_argument('--iext', type=float, default=0.3, help='input current (nA)')
parser.add_argument('--rate_inh', type=float, default=0.3, help='input current (nA)')
parser.add_argument('--rate_exc', type=float, default=0.3, help='input current (nA)')

parser.add_argument('--tau_e', type=float, default=5.0, help='excitatory synaptic decay (ms)')
parser.add_argument('--tau_i', type=float, default=5.0, help='inhibitory synaptic decay (ms)')
parser.add_argument('--time', type=float, default=1000, help='Total Time of simulation (ms)')
args = parser.parse_args()


b_e = args.b_e
Iext = args.iext
tau_e = args.tau_e
tau_i = args.tau_e
TotTime = args.time
type = args.type
rate_inh = args.rate_inh
rate_exc = args.rate_exc

tau = 10*ms
C = 200*pF
gL = 10*nS
tauw = 500*ms
a =.0*nS# 4*nS
Ee=0*mV
Ei=-80*mV
I = 0*nA

DT=0.1 # time step
defaultclock.dt = DT*ms

EL_i = -65.0
EL_e = -64.0

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

G_exc = NeuronGroup(1, model=eqs, threshold='vm > Vcut',refractory=5*ms,
                     reset="vm = Vr; w += b", method='heun')
G_exc.vm = -65*mV#EL
G_exc.w = a * (G_exc.vm - G_exc.EL)
G_exc.Vr = -65*mV 
G_exc.b=b_e*pA
G_exc.DeltaT=2*mV
G_exc.VT=-50.*mV
# G_exc.Vcut=G_exc.VT + 5 * G_exc.DeltaT
G_exc.Vcut = -30*mV
G_exc.EL=EL_e*mV
G_exc.TsynI =5*ms
G_exc.TsynE =5*ms



G_inh = NeuronGroup(1, model=eqs, threshold='vm > Vcut',refractory=5*ms,
                     reset="vm = Vr; w += b", method='heun')
G_inh.vm = -65*mV#EL
G_inh.w = a * (G_inh.vm - G_inh.EL)
G_inh.Vr = -65*mV #
G_inh.TsynI =5*ms
G_inh.TsynE =5*ms
G_inh.b=0*pA
G_inh.DeltaT=0.5*mV
G_inh.VT=-50.*mV
# G_inh.Vcut=G_inh.VT + 5 * G_inh.DeltaT
G_inh.Vcut=-30*mV
G_inh.EL=EL_i*mV

p_con = 0.05
gei = 0.2
Ntot = 10000

P_inh = int(p_con*gei*Ntot)
P_exc = int(p_con*(1- gei)*Ntot)

P_ed_inh = PoissonGroup(P_inh, rates=rate_inh*Hz) #5% of 2000 inh cells
P_ed_exc = PoissonGroup(P_exc, rates=rate_exc*Hz) #5% of 8000 exc cells

Qi=5*nS
Qe=1.5*nS

prbC=1. #1


S_edin_in = Synapses(P_ed_inh, G_inh, on_pre='GsynI_post+=Qi')
S_edin_in.connect(p=prbC)

S_edin_ex = Synapses(P_ed_inh, G_exc, on_pre='GsynI_post+=Qi')
S_edin_ex.connect(p=prbC)

S_edex_in = Synapses(P_ed_exc, G_inh, on_pre='GsynE_post+=Qe')
S_edex_in.connect(p=prbC)

S_edex_ex = Synapses(P_ed_exc, G_exc, on_pre='GsynE_post+=Qe')
S_edex_ex.connect(p=prbC)


if type == 'RS':
    statemon_exc = StateMonitor(G_exc, 'vm', record=0)
    spikemon_exc = SpikeMonitor(G_exc)
    wmonitor = StateMonitor(G_exc, 'w', record=0)
    FRG_exc = PopulationRateMonitor(G_exc)

    
    run(TotTime*ms)

    # plot(statemon_exc.t/ms, statemon_exc.vm[0])

    mon_vm = statemon_exc[0].vm
    LfrG_exc=array(FRG_exc.rate/Hz)
    
    BIN=5
    time_array = arange(int(TotTime/DT))*DT
    TimBinned,popRateG_exc=bin_array(time_array, BIN, time_array),bin_array(LfrG_exc, BIN, time_array)
    
    plt.plot(TimBinned,popRateG_exc)

    print(mean(mon_vm[int(0.8*len(mon_vm)):len(mon_vm)]))
    print(mean(popRateG_exc[int(0.8*len(popRateG_exc))::]))
    # for t in spikemon_exc.t:
        # axvline(t/ms, ls='--', c='C1', lw=3)
elif type =='FS':
    statemon_inh = StateMonitor(G_inh, 'vm', record=0)
    spikemon_inh = SpikeMonitor(G_inh)
    FRG_inh = PopulationRateMonitor(G_inh)

    run(TotTime*ms)

    LfrG_inh=array(FRG_inh.rate/Hz)
    BIN=5
    time_array = arange(int(TotTime/DT))*DT
    TimBinned,popRateG_inh=bin_array(time_array, BIN, time_array),bin_array(LfrG_inh, BIN, time_array)

    plot(TimBinned,popRateG_inh)
    print(mean(popRateG_inh))

    # plot(statemon_inh.t/ms, statemon_inh.vm[0])
    # for t in spikemon_inh.t:
    #     axvline(t/ms, ls='--', c='C1', lw=3)
xlabel('Time (ms)')
ylabel('vm');