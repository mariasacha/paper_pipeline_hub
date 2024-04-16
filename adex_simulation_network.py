from brian2 import *
import matplotlib.pyplot as plt
from functions import *
import argparse
# from Tf_calc.model_library import get_model
from Tf_calc.cell_library import get_neuron_params_double_cell

start_scope()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--b_e', type=float, default=0.0, help='adaptation (pA)')
parser.add_argument('--iext', type=float, default=0.5, help='external input (Hz)')
# parser.add_argument('--model', type=str, default='adex', help='model to run')
parser.add_argument('--cells', type=str, default='FS-RS', help='cell types of the populations')

parser.add_argument('--tau_e', type=float, default=5.0, help='excitatory synaptic decay (ms)')
parser.add_argument('--tau_i', type=float, default=5.0, help='inhibitory synaptic decay (ms)')
parser.add_argument('--time', type=float, default=1000, help='Total Time of simulation (ms)')
args = parser.parse_args()


MODEL = args.model
CELLS = args.cells

# eqs = get_model(MODEL)
params = get_neuron_params_double_cell(CELLS)

# Extract values from params for each key
extracted_values = {}
for key in params.keys():
    extracted_values[key] = params[key]
# Unpack extracted values into variables
locals().update(extracted_values)

b_e = args.b_e
Iext = args.iext
tau_e = args.tau_e
tau_i = args.tau_e
TotTime = args.time

N1 = int(gei*Ntot)
N2 = int((1-gei)*Ntot)

DT=0.1 # time step
# N1 = 2000 # number of inhibitory neurons
# N2 = 8000 # number of excitatory neurons 

duration = TotTime*ms

C = Cm*pF
gL = Gl*nS
tauw = tau_w*ms
a =0.0*nS# 4*nS
#b = 0.08*nA
I = 0.*nA
Ee=E_e*mV
Ei=E_i*mV

# EL_i = -65.0
# EL_e = -64.0

seed(9) #9,11,25

sim_name = f'_b_{b_e}_tau_e_{tau_e}_tau_i_{tau_i}_eli_{int(EL_i)}_ele_{int(EL_e)}_iext_{Iext}'

# print("V_m={} , a_i={}, a_e={}, V_r={}, tau_i={}, tau_e={}, b_i={}, b_e={}, delta_i={}, delta_e={}, V_th={}, EL_i={}, EL_e={}, Vcut_i={}, Vcut_e={}".format(V_m, a_i, a_e, V_r, tau_i, tau_e, b_i, b_e, delta_i, delta_e, V_th, EL_i, EL_e,Vcut_i,Vcut_e) )


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
G_inh.vm = V_m * mV 
G_inh.w = a_i*nS * (G_inh.vm - G_inh.EL)
G_inh.Vr = V_r * mV  
G_inh.TsynI = tau_i * ms  
G_inh.TsynE = tau_e * ms  
G_inh.b = b_i * pA
G_inh.DeltaT = delta_i * mV
G_inh.VT = V_th * mV
# G_inh.Vcut = G_inh.VT + 5 * G_inh.DeltaT
G_inh.Vcut = Vcut_i * mV
G_inh.EL = EL_i * mV

# Population 2 - Regular Spiking

G_exc = NeuronGroup(N2, model=eqs, threshold='vm > Vcut',refractory=5*ms,
                     reset="vm = Vr; w += b", method='heun')
G_exc.vm = V_m*mV
G_exc.w = a_e*nS * (G_exc.vm - G_exc.EL)
G_exc.Vr = V_r*mV
G_exc.TsynI =tau_i*ms
G_exc.TsynE =tau_e*ms
G_exc.b=b_e*pA
G_exc.DeltaT=delta_e*mV
G_exc.VT=V_th*mV
G_exc.Vcut = Vcut_e * mV
# G_exc.Vcut=G_exc.VT + 5 * G_exc.DeltaT
G_exc.EL=EL_e*mV

# external drive--------------------------------------------------------------------------

P_ed=PoissonGroup(N2, rates=Iext*Hz)

# Network-----------------------------------------------------------------------------

# connections-----------------------------------------------------------------------------
#seed(0)
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

# Recording tools -------------------------------------------------------------------------------
rec1=2000
rec2=8000

M1G_inh = SpikeMonitor(G_inh)
FRG_inh = PopulationRateMonitor(G_inh)
M1G_exc = SpikeMonitor(G_exc)
FRG_exc = PopulationRateMonitor(G_exc)

# Run simulation -------------------------------------------------------------------------------

print('--##Start simulation##--')
run(duration)
print('--##End simulation##--')

# Plots -------------------------------------------------------------------------------

# prepare raster plot
RasG_inh = array([M1G_inh.t/ms, [i+N2 for i in M1G_inh.i]])
RasG_exc = array([M1G_exc.t/ms, M1G_exc.i])
TimBinned, popRateG_exc, popRateG_inh, Pu = prepare_FR(TotTime,DT, FRG_exc, FRG_inh, P2mon)


# # ----- Raster plot + mean adaptation ------
# fig, axes = figure.add_subplots(2,1,figsize=(8,12))
fig, axes = plt.subplots(2,1,figsize=(5,8))

plot_raster_meanFR(RasG_inh,RasG_exc, TimBinned, popRateG_inh, popRateG_exc, Pu, axes, sim_name)


print(f" done")
plt.show()

