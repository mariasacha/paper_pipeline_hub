# Model from  Di Volo et al. Neural Comp. 2019
# %matplotlib widget

import numpy as np
import matplotlib.pylab as plt
from math import erf


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
    vthr=P[0]+P[1]*(muV-muV0)/DmuV0+P[2]*(sV-sV0)/DsV0+P[3]*(TvN-TvN0)/DTvN0+P[5]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0)+P[6]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0)+P[7]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0)+P[8]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+P[9]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+P[10]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0;

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
PRS=np.load('RS-cell0_CONFIG1_fit.npy')
PFS=np.load('FS-cell_CONFIG1_fit.npy')

#Time
tfinal=10
dt=0.0001
t = np.linspace(0, tfinal, int(tfinal/dt))

# Additive Noise
v_drive = 0.2
sigma=3.5
os_noise = sigma*OU(tfinal) + v_drive

#To adjust
bRS = 5*1e-12 #adaptation 
Te=5.*1.e-3; #excitatory synaptic decay
Ti=5*1.e-3; #inhibitory synaptic decay

Ele =-64*1e-3 #leak reversal (exc)
Eli = -65*1e-3 #leak reversal (inh)
T = 20*1e-3 # time constant

#Initial Conditions
fecont=8;
ficont=10;
w=fecont*bRS*twRS

LSw=[]
LSfe=[]
LSfi=[]

print("starting")
for i in range(len(t)):
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
ax2.set_ylabel('mean w (pA)')
#ax2.set_ylim(0.0, 0.045)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('population Firing Rate')

plt.suptitle(sim_name)

ax2.legend()
ax3.legend()

plt.show()
#f.close()
