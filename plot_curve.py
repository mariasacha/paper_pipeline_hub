# Model from  Di Volo et al. Neural Comp. 2019
import numpy as np
from math import erf
import argparse
from scipy import optimize
import os
from IPython.display import clear_output
from Tf_calc.cell_library import get_neuron_params_double_cell
import matplotlib.pyplot as plt

def TF_2(finh,fexc,fext,fextin,P,adapt,El):


    fe = (fexc+fext)*(1.-gei)*pconnec*Ntot;
    fi = (finh+fextin)*gei*pconnec*Ntot;

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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#range of values

#Other
parser.add_argument('--cell',  default='FS-RS', help='parameters to load')
parser.add_argument('--fit_fs',  default='FS-cell_CONFIG1_fit_2.npy', help='fit for fs')
parser.add_argument('--fit_rs',  default='RS-cell0_CONFIG1_fit_2.npy', help='fit for rs')

args = parser.parse_args()

NAME = args.cell
file_rs = args.fit_rs
file_fs = args.fit_fs


#Model parameters
# Gl=10*1.e-9; #leak conductance
# Cm=200*1.e-12; #capacitance

# Qe=1.5*1.e-9; #excitatory quantal conductance
# Qi=5.*1.e-9; #inhibitory quantal conductance

# Ee=0; #excitatory reversal potential
# Ei=-80*1.e-3; #inhibitory reversal

# twRS=.5; #adaptation time constant 

#Network parameters
# pconnec=0.05; #probability of connection
# gei=0.2; #percentage of inhibitory cells
# Ntot=10000; #total number of cells

#Fitting coefficients
PRS=np.load(file_rs)
PFS=np.load(file_fs)

# Ele =-64*1e-3 #leak reversal (exc)
# Eli = -65*1e-3 #leak reversal (inh)
T = 20*1e-3 # time constant

#Initial Conditions
fecont=2;
ficont=10;

LSfe=[]
nuev=np.arange(0.00000001,20,step=0.1)


nuext = 0 
nuextin = 0.

# bRS = b_e*1e-12 
# Ti = tau_I*1.e-3
# Te = tau_E*1.e-3

params = get_neuron_params_double_cell(NAME, SI_units=False)
p = params
Qe,Qi,Te,Ti,Ee,Ei,Cm,twRS,Gl, gei, Ntot, pconnec = p['Q_e']*1e-9,p['Q_i']*1e-9,p['tau_e']*1e-3,p['tau_i']*1e-3,p['E_e']*1e-3,p['E_i']*1e-3, p['Cm']*1e-12, p['tau_w']*1e-3,p['Gl']*1e-9, p['gei'], p['Ntot'], p['p_con']

a,bRS,Ele = p['a_e']*1e-9, p['b_e']*1e-12, p['EL_e']*1e-3

a,bFS,Eli = p['a_i']*1e-9, p['b_i']*1e-12, p['EL_i']*1e-3

bRS = 0

# print("running: ", sim_name)
for nue in nuev:

    w=nue*bRS*twRS
    try:
        nui_fix = optimize.fixed_point(TF_2, [1.0], args=(nue, nuext, nuextin, PFS, 0., Eli))
    except RuntimeError:
        print(f"runtime error: b_e = {b_e}, tau_e={tau_I}, tau_i={tau_E}")
    TFe_fix = TF_2(nui_fix, nue, nuext, 0., PRS, w, Ele)
    LSfe.append(float(TFe_fix))

plt.plot(nuev, LSfe, "b-", label=f"b={bRS*1.e+12}")
plt.plot(nuev, nuev, "k--")
plt.legend()
plt.show() 

# Define a function that returns TF_2(nue) - nue
def func(nue):
    w = nue * bRS * twRS
    nui_fix = optimize.fixed_point(TF_2, [1.0], args=(nue, nuext, nuextin, PFS, 0., Eli), xtol = 1.e-9, maxiter=1500)
    TFe_fix = TF_2(nui_fix, nue, nuext, nuextin, PRS, w, Ele)
    return TFe_fix - nue




# Use fsolve to find the roots of the difference equation
initial_guess = 4  # Initial guess for the root
intersection_point = optimize.fsolve(func, initial_guess)
print("solution = ", intersection_point)