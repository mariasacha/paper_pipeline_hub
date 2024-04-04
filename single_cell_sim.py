from brian2 import *
import matplotlib.pyplot as plt
from functions import *
import argparse

start_scope()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

statemon = StateMonitor(G_exc, 'vm', record=0)
spikemon = SpikeMonitor(G_exc)

run(TotTime*ms)

plot(statemon.t/ms, statemon.vm[0])
for t in spikemon.t:
    axvline(t/ms, ls='--', c='C1', lw=3)
xlabel('Time (ms)')
ylabel('vm');