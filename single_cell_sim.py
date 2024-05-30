from brian2 import *
import matplotlib.pyplot as plt
from functions import *
import argparse
import numpy as np
import ast  
from Tf_calc.cell_library import get_neuron_params_double_cell

def convert_values(input_dict):
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

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


# def parse_kwargs(kwargs_str):
#     try:
#         kwargs = ast.literal_eval(kwargs_str)
#         if not isinstance(kwargs, dict):
#             raise ValueError("Invalid dictionary format")
#         return kwargs
#     except (SyntaxError, ValueError) as e:
#         print(f"Error parsing kwargs: {e}")
#         return None



start_scope()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--cell', default='RS', help='type of cell (RS or FS)')
# parser.add_argument('--kwargs', type=str, default='{"use": False}', help='String representation of kwargs - change the first argument to "use": True before adding your kwargs, e.g: "{"use": True, "b": 60}"')
parser.add_argument('--kwargs', nargs='*', required=True, action=ParseKwargs, help='String representation of kwargs - change the first argument to "use": True before adding your kwargs, e.g: "{"use": True, "b": 60}"')

parser.add_argument('--iext', type=float, default=0.3, help='input current (nA)')
parser.add_argument('--time', type=float, default=200, help='Total Time of simulation (ms)')

args = parser.parse_args()

CELLS = args.cell

params = get_neuron_params_double_cell(CELLS)


kwargs1 = args.kwargs
kwargs = convert_values(kwargs1)
print(kwargs)

# Extract values from params for each key
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



Iext = args.iext
TotTime = args.time

C = Cm*pF
gL = Gl*nS
tauw = tau_w*ms
a =a*nS# 4*nS
Ee=E_e*mV
Ei=E_i*mV
I = Iext*nA


eqs = """
dvm/dt=(gL*(El-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)-GsynE*(vm-Ee)-GsynI*(vm-Ei)+I-w)/C : volt (unless refractory)
dw/dt=(a*(vm-El)-w)/tauw : amp
dGsynI/dt = -GsynI/TsynI : siemens
dGsynE/dt = -GsynE/TsynE : siemens
TsynI:second
TsynE:second
Vr:volt
b_e:amp
DeltaT:volt
Vcut:volt
VT:volt
El:volt
"""

G_exc = NeuronGroup(1, model=eqs, threshold='vm > Vcut',refractory=5*ms,
                     reset="vm = Vr; w += b_e", method='heun')
G_exc.vm = V_m*mV#EL
G_exc.w = a * (G_exc.vm - G_exc.El)
G_exc.Vr = V_r*mV 
G_exc.b_e=b*pA
G_exc.DeltaT=delta*mV
G_exc.VT=V_th*mV
# G_exc.Vcut=G_exc.VT + 5 * G_exc.DeltaT
G_exc.Vcut = V_cut*mV
G_exc.El=EL*mV
G_exc.TsynI =tau_e*ms
G_exc.TsynE =tau_i*ms



statemon_exc = StateMonitor(G_exc, 'vm', record=0)
spikemon_exc = SpikeMonitor(G_exc)

run(TotTime*ms)

plot(statemon_exc.t/ms, statemon_exc.vm[0])
for t in spikemon_exc.t:
    axvline(t/ms, ls='--', c='C1', lw=3)

xlabel('Time (ms)')
ylabel('vm');