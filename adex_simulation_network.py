from brian2 import *
import matplotlib.pyplot as plt
from functions import *
import argparse
import ast
import numpy as np
import os

# from Tf_calc.model_library import get_model
from Tf_calc.cell_library import get_neuron_params_double_cell

def convert_values(input_dict: dict) -> dict:
    """
    Convert string values in the input dictionary to appropriate types.
    """
    converted_dict = {}
    for key, value in input_dict.items():
        # Check for boolean values
        if value.lower() == 'true':
            converted_dict[key] = True
        elif value.lower() == 'false':
            converted_dict[key] = False
        # Check for integer values
        elif value.isdigit():
            converted_dict[key] = int(value)
        # Add more type checks if needed (e.g., for floats)
        else:
            converted_dict[key] = value  # Keep the value as is if no conversion is needed
    return converted_dict

# Define the custom action to parse keyword arguments
class ParseKwargs(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: list, option_string: str = None):
        """
        Parse keyword arguments from the command line.
        """
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

start_scope()

# Set up argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cells', type=str, default='FS-RS', help='cell types of the populations - do not pass single cells')

parser.add_argument('--iext', type=float, default=0.5, help='external input rate (in Hz) to a group of neurons that fire according to a Poisson process.')
parser.add_argument('--input', type=float, default=0, help='Stable input amplitude (Hz)')
parser.add_argument('--plat_dur', type=float, default=0, help='If 0 the input will be applied for the whole duration of the simulation')

parser.add_argument('--time', type=float, default=1000, help='Total Time of simulation (ms)')
parser.add_argument('--save_path', default=None, help='save path ')
parser.add_argument('--save_mean', default=True, help='save mean firing rate (if save_path is provided)')
parser.add_argument('--save_all', default=False, help='save the whole simulation (if save_path is provided)')
parser.add_argument('--kwargs', nargs='*', required=True, action=ParseKwargs, help='String representation of kwargs - change the first argument to "use": True before adding your kwargs, e.g: "{"use": True, "b": 60}"')

args = parser.parse_args()


CELLS = args.cells

# Get neuron parameters for the specified cell types
params = get_neuron_params_double_cell(CELLS)


# Use the parameters that are passed in kwargs
kwargs1 = args.kwargs
# print(kwargs1)
kwargs = convert_values(kwargs1)


if kwargs['use']: #only if use=True
    for key in kwargs.keys():
        if key in params.keys():
            params[key] = kwargs[key]
        elif key == 'use':
            continue
        else:
            raise Exception(f"Key '{key}' not in the valid keys \nValid keys: {params.keys()}")
# Update locals
locals().update(params)



save_path = args.save_path
save_mean = args.save_mean
save_all = args.save_all

TotTime = args.time
Iext = args.iext

# Calculate the number of neurons in each population
N1 = int(gei*Ntot)
N2 = int((1-gei)*Ntot)

DT=0.1 # time step

#Create the kick
AmpStim = args.input #0
time_peek = 200.
TauP=20 #20

if not args.plat_dur:
    plat = TotTime - time_peek - TauP #100
else:
    plat = args.plat_dur
t2 = np.arange(0, TotTime, DT)
test_input = []

for ji in t2:
    test_input.append(0. + input_rate(ji, time_peek, TauP, 1, AmpStim, plat))
Input_Stim = TimedArray(test_input * Hz, dt=DT * ms)

duration = TotTime*ms

# Define neuron parameters
C = Cm*pF
gL = Gl*nS
tauw = tau_w*ms

I = 0.*nA
Ee=E_e*mV
Ei=E_i*mV


seed(9) #9,11,25

sim_name = f'_b_{b_e}_tau_e_{tau_e}_tau_i_{tau_i}_eli_{int(EL_i)}_ele_{int(EL_e)}_iext_{Iext}'


print('b_e= ', b_e)
eqs = """
dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)-GsynE*(vm-Ee)-GsynI*(vm-Ei)+I-w)/C : volt (unless refractory)
dw/dt=(a*(vm-EL)-w)/tauw : amp
dGsynI/dt = -GsynI/TsynI : siemens
dGsynE/dt = -GsynE/TsynE : siemens
TsynI:second
TsynE:second
Vr:volt
b:amp
a:siemens
DeltaT:volt
Vcut:volt
VT:volt
EL:volt
"""

# Population 1 - Fast Spiking

G_inh = NeuronGroup(N1, model=eqs, threshold='vm > Vcut',refractory=5*ms,
                     reset="vm = Vr; w += b", method='heun')
G_inh.vm = V_m * mV 
G_inh.a = a_i * nS 
G_inh.w = G_inh.a * (G_inh.vm - G_inh.EL)
G_inh.Vr = V_r * mV  
G_inh.TsynI = tau_i * ms  
G_inh.TsynE = tau_e * ms  
G_inh.b = b_i * pA
G_inh.DeltaT = delta_i * mV
G_inh.VT = V_th * mV
G_inh.Vcut = V_cut * mV
G_inh.EL = EL_i * mV

# Population 2 - Regular Spiking

G_exc = NeuronGroup(N2, model=eqs, threshold='vm > Vcut',refractory=5*ms,
                     reset="vm = Vr; w += b", method='heun')
G_exc.vm = V_m*mV
G_exc.a = a_e * nS 
G_exc.w = G_exc.a * (G_exc.vm - G_exc.EL)
G_exc.Vr = V_r*mV
G_exc.TsynI =tau_i*ms
G_exc.TsynE =tau_e*ms
G_exc.b=b_e*pA
G_exc.DeltaT=delta_e*mV
G_exc.VT=V_th*mV
G_exc.Vcut = V_cut * mV
G_exc.EL=EL_e*mV

# external drive--------------------------------------------------------------------------
if AmpStim > 0:
    print("Input =", AmpStim, 'plat=', plat)
    P_ed=PoissonGroup(N2, rates='Input_Stim(t)')
else:
    P_ed=PoissonGroup(N2, rates=Iext*Hz)

# Network-----------------------------------------------------------------------------

# connections-----------------------------------------------------------------------------
#seed(0)
Qi=Q_i*nS
Qe=Q_e*nS

prbC=p_con #0.05

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
RasG_inh = np.array([M1G_inh.t / ms, [i + N2 for i in M1G_inh.i]])
RasG_exc = np.array([M1G_exc.t / ms, M1G_exc.i])
if AmpStim > 0:
    time_array = np.arange(int(TotTime / DT)) * DT
    input_bin = bin_array(np.array(test_input), 5, time_array)
else:
    input_bin = np.full(1, np.nan)
TimBinned, popRateG_exc, popRateG_inh, Pu = prepare_FR(TotTime, DT, FRG_exc, FRG_inh, P2mon, BIN=5)
if save_path:
    try:
        os.listdir(save_path)
    except:
        os.makedirs(save_path)
    
    if save_mean:
        print("Saving the mean")
        # print("Exc=", np.mean(popRateG_exc[int(len(popRateG_exc)/2):]), "Inh=",np.mean(popRateG_inh[int(len(popRateG_inh)/2):]))
        np.save(save_path + f'{CELLS}_mean_exc_amp_{AmpStim}.npy', np.array([np.mean(popRateG_exc[int(len(popRateG_exc)/2):]),AmpStim, params], dtype=object))
        np.save(save_path + f'{CELLS}_mean_inh_amp_{AmpStim}.npy', np.array([np.mean(popRateG_inh[int(len(popRateG_inh)/2):]), AmpStim, params], dtype=object))
    if save_all:
        print("Saving the whole simulation")
        np.save(save_path + f'{CELLS}_inh_amp_{AmpStim}.npy', np.array([popRateG_inh, AmpStim, params], dtype=object))
        np.save(save_path + f'{CELLS}_exc_amp_{AmpStim}.npy', np.array([popRateG_exc,AmpStim, params], dtype=object))

# # ----- Raster plot + mean adaptation ------

fig, axes = plt.subplots(2, 1, figsize=(5, 8))

plot_raster_meanFR(RasG_inh, RasG_exc, TimBinned, popRateG_inh, popRateG_exc, Pu, axes, sim_name, input_bin)


print(f" done")
plt.show()

