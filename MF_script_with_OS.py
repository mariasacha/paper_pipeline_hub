# Model from  Di Volo et al. Neural Comp. 2019
# %matplotlib widget

import numpy as np
import matplotlib.pylab as plt
from math import erf
import argparse
from functions import *
from Tf_calc.cell_library import get_neuron_params_double_cell
from Tf_calc.theoretical_tools import convert_params
import ast
import json


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


def TF(P,fexc,finh,adapt, El):
    #Transfer Function 

    fe = fexc*(1.-gei)*pconnec*Ntot;
    fi = finh*gei*pconnec*Ntot;
    
    muGi = Qi*Ti*fi;
    muGe = Qe*Te*fe;
    muG = Gl+muGe+muGi;
    muV = (muGe*Ee+muGi*Ei+Gl*El-adapt)/muG;
    
    Tm = Cm/muG;
    
    Ue =  Qe/muG*(Ee-muV);
    Ui = Qi/muG*(Ei-muV);
    sV = np.sqrt(fe*(Ue*Te)*(Ue*Te)/2./(Te+Tm)+fi*(Ui*Ti)*(Ui*Ti)/2./(Ti+Tm));
    
    fe= fe+1e-9;
    fi=fi+1e-9;
    Tv = ( fe*(Ue*Te)*(Ue*Te) + fi*(Qi*Ui)*(Qi*Ui)) /( fe*(Ue*Te)*(Ue*Te)/(Te+Tm) + fi*(Qi*Ui)*(Qi*Ui)/(Ti+Tm) );
    TvN = Tv*Gl/Cm;
    
    muV0=-60e-3;
    DmuV0 = 10e-3;
    sV0 =4e-3;
    DsV0= 6e-3;
    TvN0=0.5;
    DTvN0 = 1.;

    #Effective threshold
    
    vthr=P[0]+P[1]*(muV-muV0)/DmuV0+P[2]*(sV-sV0)/DsV0+P[3]*(TvN-TvN0)/DTvN0+P[4]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0)+P[5]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0)+P[6]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0)+P[7]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+P[8]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+P[9]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0;

    frout=.5/TvN*Gl/Cm*(1-erf((vthr-muV)/np.sqrt(2)/sV));
    
    return frout;


def OU(tfin):
    # Ornstein-Ulhenbeck process
    
    theta = 1/(5*1.e-3 )  # Mean reversion rate
    mu = 0     # Mean of the process
    sigma = 1   # Volatility or standard deviation
    dt = 0.1*1.e-3    # Time increment
    T = tfin        # Total time period

    # Initialize the variables
    t = np.arange(0, T, dt)         # Time vector
    n = len(t)                      # Number of time steps
    x = np.zeros(n)                 # Array to store the process values
    x[0] = 0  # Initial value

    # Generate the process using the Euler-Maruyama method
    for i in range(1, n):
        dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
        x[i] = x[i-1] + dx
    return x

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cells', type=str, default='FS-RS', help='cell types of the populations')

# parser.add_argument('--b_e', type=float, default=0.0, help='adaptation - in pA')

parser.add_argument('--iext', type=float, default=0.3, help='external input - in Hz')
parser.add_argument('--input', type=float, default=0, help='Stable input amplitude (Hz)')
parser.add_argument('--plat_dur', type=float, default=0, help='If 0 the input will be applied for the whole duration of the simulation')

# parser.add_argument('--tau_e', type=float, default=5.0, help='excitatory synaptic decay - in ms')
# parser.add_argument('--tau_i', type=float, default=5.0, help='inhibitory synaptic decay - in ms')
parser.add_argument('--T', type=float, default=20.0, help='time constant - in ms')
# parser.add_argument('--kwargs', type=str, default='{"use": False}', help='String representation of kwargs - change the first argument to "use": True before adding your kwargs, e.g: "{"use": True, "b": 60}"')
parser.add_argument('--kwargs', nargs='*', required=True, action=ParseKwargs, help='String representation of kwargs - change the first argument to "use": True before adding your kwargs, e.g: "{"use": True, "b": 60}"')

parser.add_argument('--time', type=float, default=10, help='Total Time of simulation - in s')

parser.add_argument('--file_fs',  default='FS-cell_CONFIG1_fit.npy', help='fit for fs')
parser.add_argument('--file_rs',  default='RS-cell0_CONFIG1_fit.npy', help='fit for rs')


args = parser.parse_args()

kwargs1 = args.kwargs
kwargs = convert_values(kwargs1)

CELLS = args.cells
params = get_neuron_params_double_cell(CELLS)

if kwargs['use']: #only if use=True
    for key in kwargs.keys():
        if key in params.keys():
            params[key] = kwargs[key]
        elif key == 'use':
            continue
        else:
            raise Exception(f"Key '{key}' not in the valid keys \nValid keys: {params.keys()}")
# Update locals
params = convert_params(params)
locals().update(params)

p = params

Iext = args.iext
TotTime = args.time

#Model parameters
Gl=p['Gl']; #leak conductance
Cm=p['Cm']; #capacitance

Qe=p['Q_e']; #excitatory quantal conductance
Qi=p['Q_i']; #inhibitory quantal conductance

Ee=p['E_e']; #excitatory reversal potential
Ei=p['E_i']; #inhibitory reversal

twRS=p['tau_w']; #adaptation time constant 

#Network parameters
pconnec= p['p_con']; #probability of connection
gei=p['gei']; #percentage of inhibitory cells
Ntot=p['Ntot']; #total number of cells


#Fitting coefficients
PRS=np.load(args.file_rs)
PFS=np.load(args.file_fs)

#Time
tfinal=TotTime
dt=0.0001
t = np.linspace(0, tfinal, int(tfinal/dt))

# Additive Noise
v_drive = Iext
sigma=3.5
sigma=1
os_noise = sigma*OU(tfinal) + v_drive

#Create the kick
AmpStim = args.input #0
time_peek = 200.
TauP=20 #20

if not args.plat_dur:
    plat = TotTime*1000 - time_peek - TauP #100
else:
    plat = args.plat_dur

test_input = []
t2 = np.arange(0, TotTime*1e3, 0.1)
for ji in t2:
    test_input.append(0. + input_rate(ji, time_peek, TauP, 1, AmpStim, plat))

# print(max(os_noise), min(os_noise))
# print(max(test_input), min(os_noise))

#To adjust
bRS = p['b_e']; #adaptation 
Te=p['tau_e']; #excitatory synaptic decay
Ti=p['tau_i']; #inhibitory synaptic decay

Ele =p['EL_e'] #leak reversal (exc)
Eli = p['EL_i'] #leak reversal (inh)
T = args.T*1e-3 # time constant
# T = 5*1e-3 # time constant

#Initial Conditions
fecont=5;
ficont=12;
w=fecont*bRS*twRS

LSw=[]
LSfe=[]
LSfi=[]
if AmpStim>0:
    print("Input = ", AmpStim, "plat=", plat)
print("starting")
for i in range(len(t)):

    if AmpStim>0:
        # print("Input = ", AmpStim)
        external_input = test_input[i]
    else:
        external_input =os_noise[i]
    fecontold=fecont

    FEX = fecont + external_input
    FINH = fecontold + external_input

    if FEX < 0:
        FEX = 0
    if FINH < 0:
        FINH = 0

    fecont+=dt/T*(TF(PRS, FEX,ficont,w, Ele)-fecont)
    w+=dt*( -w/twRS+(bRS)*fecontold) 
    ficont+=dt/T*(TF(PFS,FINH,ficont,0., Eli)-ficont)

    LSfe.append(float(fecont))
    LSfi.append(float(ficont))
    LSw.append(float(w))


sim_name = f"_b_{int(bRS * 1.e12)}_tau_e_{Te * 1.e3}_tau_i_{Ti * 1.e3}_eli_{int(Eli * 1.e3)}_ele_{int(Ele * 1.e3)}_iext_{v_drive}"

print("done")

fig=plt.figure(figsize=(8,6))
ax3=fig.add_subplot(211)
ax2 = ax3.twinx()

t_st = 0

ax3.plot(t[t_st:], LSfe[t_st:],'steelblue', label="Exc")
ax3.plot(t[t_st:], LSfi[t_st:],'r', label="Inh")
ax2.plot(t[t_st:], LSw[t_st:], 'orange' , label="W")
if AmpStim>0:
    ax3.plot(t[t_st:], test_input[t_st:], 'green' , label="input")
ax2.set_ylabel('mean w (pA)')
#ax2.set_ylim(0.0, 0.045)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('population Firing Rate')

# ask matplotlib for the plotted objects and their labels
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

# print(PRS)
# # print(PFS)

# print(np.mean(LSfe))
# print(np.mean(LSfi))
plt.suptitle(sim_name)

# ax2.legend()
# ax3.legend()

plt.show()
#f.close()
