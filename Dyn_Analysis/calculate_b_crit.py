# Model from  Di Volo et al. Neural Comp. 2019
import numpy as np
from math import erf
import argparse
from scipy import optimize
import os
from IPython.display import clear_output
import sys
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)


"""
returns array of type [[tau_e_1, tau_i_1, b_crit_1], [tau_e_2, tau_i_2, b_crit_2], ..., [tau_e_n, tau_i_n, b_crit_n]]
"""
def get_np_arange(value):
   """
   solution to input np.arange in the argparser 
   """
   try:
       values = [float(i) for i in value.split(',')]
       assert len(values) in (1, 3)
   except (ValueError, AssertionError):
       raise argparse.ArgumentTypeError(
           'Provide a CSV list of 1 or 3 integers'
       )

   # return our value as is if there is only one
   if len(values) == 1:
       return np.array(values)

   # if there are three - return a range
   return np.arange(*values)

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
parser.add_argument('--b_e_range', type=get_np_arange , default='0,60,1', help='b_e range of values')
parser.add_argument('--tau_e_range', type=get_np_arange , default='5,7,10',help='tau_e range of values \
					- if you iterate tau_i then set  tau_e_range=np.arange(5.,10.,step=500)')
parser.add_argument('--tau_i_range', type=get_np_arange , default= '3,9,0.1', help='tau_i range of values \
					- if you iterate tau_e then set tau_i_range=np.arange(5.,10.,step=500)')

#Other
parser.add_argument('--save_path',  default='./', help='Path to save the results of the simulations')
parser.add_argument('--overwrite', type=bool, default=False, help='If True it will overwrite the existant paths')

args = parser.parse_args()

save_path = "./Dyn_Analysis" +  args.save_path + "/"
OVERWRITE = args.overwrite

#Choose the values of the scan
tauIv=args.tau_i_range
tauEv=args.tau_e_range
bvals = args.b_e_range


if len(tauIv) > 1 and len(tauEv)>1:
    print("Iterations for both tau_e and tau_i")
    raise Exception("Change the tau_e_range or tau_i_range ")

if len(tauEv)==1 and len(tauIv)>1:
    tau_i_iter = True
    tau_str = 'tau_i'
elif len(tauEv)>1 and len(tauIv)==1:
    tau_i_iter = False
    tau_str = 'tau_e'

file_name = save_path + f'b_thresh_{tau_str}.npy'

try:
    os.listdir(save_path)
except:
    os.makedirs(save_path)

if os.path.exists(file_name) and not OVERWRITE:
    raise Exception("This file already exists. Set overwrite to True or change path")


#Model parameters
Gl=10*1.e-9; #leak conductance
Cm=200*1.e-12; #capacitance

Qe=1.5*1.e-9; #excitatory quantal conductance
Qi=5.*1.e-9; #inhibitory quantal conductance

Ee=0; #excitatory reversal potential
Ei=-80*1.e-3; #inhibitory reversal

twRS=.5; #adaptation time constant 

#Network parameters
pconnec=0.05; #probability of connection
gei=0.2; #percentage of inhibitory cells
Ntot=10000; #total number of cells

#Fitting coefficients
PRS=np.load('./Tf_calc/data/RS-cell0_CONFIG1_fit.npy')
PFS=np.load('./Tf_calc/data/FS-cell_CONFIG1_fit.npy')


Ele =-64*1e-3 #leak reversal (exc)
Eli = -65*1e-3 #leak reversal (inh)
T = 20*1e-3 # time constant

#Initial Conditions
fecont=2;
ficont=10;

LSfe=[]

nuev=np.arange(0.00000001,10,step=0.1)


bvals = np.arange(0, 60, 1)

combinations = [(tau_I, tau_E) for tau_I in tauIv for tau_E in tauEv]

store_npy = []

for tau_I, tau_E in combinations:
    nuext = 0 
    nuextin = 0.
    print(tau_I, tau_E)
    for b_e in bvals:
        
        bRS = b_e*1e-12 
        Ti = tau_I*1.e-3
        Te = tau_E*1.e-3

        # print("running: ", sim_name)
        for nue in nuev:

            w=nue*bRS*twRS
            try:
                nui_fix = optimize.fixed_point(TF_2, [1.0], args=(nue, nuext, nuextin, PFS, 0., Eli))
            except RuntimeError:
                print(f"runtime error: b_e = {b_e}, tau_e={tau_I}, tau_i={tau_E}")
            TFe_fix = TF_2(nui_fix, nue, nuext, 0., PRS, w, Ele)
            LSfe.append(float(TFe_fix))

            #print('nue',nue)
        deltanue=nuev - LSfe

        LSfe=[]

        if all(i >= 0.00000000001 for i in deltanue): #this means that the curve would be below the bisector
            break
    print("crit b = ", b_e)

    store_npy.append([tau_E, tau_I, b_e])


np.save(file_name, np.array(store_npy))

clear_output(wait=False)
print("Done! Saved in ", save_path)
