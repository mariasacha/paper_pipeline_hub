from brian2 import *
import matplotlib.pyplot as plt
from functions import *
import argparse
import numpy as np

start_scope()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--type', type=str, default='RS', help='type of cell (RS or FS)')
parser.add_argument('--b_e', type=float, default=60, help='adaptation (pA)')
parser.add_argument('--iext', type=float, default=0.3, help='input current (nA)')

parser.add_argument('--tau_e', type=float, default=5.0, help='excitatory synaptic decay (ms)')
parser.add_argument('--tau_i', type=float, default=5.0, help='inhibitory synaptic decay (ms)')
parser.add_argument('--time', type=float, default=200, help='Total Time of simulation (ms)')
args = parser.parse_args()

b_e = args.b_e
Iext = args.iext
tau_e = args.tau_e
tau_i = args.tau_e
TotTime = args.time
type = args.type

tau = 10*ms
C = 200*pF
gL = 10*nS
tauw = 500*ms
a =.0*nS# 4*nS
Ee=0*mV
Ei=-80*mV
I = Iext*nA


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




if type == 'RS':
    statemon_exc = StateMonitor(G_exc, 'vm', record=0)
    spikemon_exc = SpikeMonitor(G_exc)
    
    run(TotTime*ms)

    plot(statemon_exc.t/ms, statemon_exc.vm[0])
    for t in spikemon_exc.t:
        axvline(t/ms, ls='--', c='C1', lw=3)
elif type =='FS':
    statemon_inh = StateMonitor(G_inh, 'vm', record=0)
    spikemon_inh = SpikeMonitor(G_inh)

    run(TotTime*ms)

    plot(statemon_inh.t/ms, statemon_inh.vm[0])
    for t in spikemon_inh.t:
        axvline(t/ms, ls='--', c='C1', lw=3)
xlabel('Time (ms)')
ylabel('vm');