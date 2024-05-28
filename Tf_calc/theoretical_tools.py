import numpy as np
# import scipy.special as sp_spec
# import scipy.integrate as sp_int
from scipy.optimize import minimize, fixed_point, fsolve
# import sys
from scipy.special import erfc, erfcinv
from scipy.ndimage import maximum_filter
from math import erf

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from math import erf
import argparse
from functions import *
from Tf_calc.cell_library import get_neuron_params_double_cell

from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt

#fede functions

def eff_thresh(mu_V, sig_V, tauN_V, params):
    P_0, P_mu, P_sig, P_tau, P_mu2, P_sig2, P_tau2, P_mu_sig, P_mu_tau, P_sig_tau = params
    V0 = P_0
    mu_0 = -60.0*1e-3
    mu_d = 0.01
    sig_0 = 0.004
    sig_d = 0.006
    tau_0 = 0.5
    tau_d = 1.0


    V1 = (P_mu*(mu_V - mu_0) / mu_d +
          P_sig*(sig_V - sig_0) / sig_d +
          P_tau*(tauN_V - tau_0) / tau_d)
    V2 = (P_mu2*((mu_V - mu_0) / mu_d)**2 +
          P_sig2*((sig_V - sig_0) / sig_d)**2 +
          P_tau2*((tauN_V - tau_0) / tau_d)**2 +
          P_mu_sig*((mu_V - mu_0) / mu_d)*((sig_V - sig_0) / sig_d) +
          P_mu_tau*((mu_V - mu_0) / mu_d)*((tauN_V - tau_0) / tau_d) +
          P_sig_tau*((sig_V - sig_0) / sig_d)*((tauN_V - tau_0) / tau_d))
    return V0 + V1 + V2

def mu_sig_tau_func(fexc, finh, fout,w_ad,params,cell_type, w_prec=False):
    p = params
    Q_e,Q_i,tau_e,tau_i,E_e,E_i,C_m,Tw,g_L, gei, ntot, pconnec = p['Q_e']*1e-9,p['Q_i']*1e-9,p['tau_e']*1e-3,p['tau_i']*1e-3,p['E_e']*1e-3,p['E_i']*1e-3, p['Cm']*1e-12, p['tau_w']*1e-3,p['Gl']*1e-9, p['gei'], p['Ntot'], p['p_con']

    if cell_type == "RS":
        try:
            a,b,E_L = p['a_e']*1e-9, p['b_e']*1e-12, p['EL_e']*1e-3
        except KeyError:
            a,b,E_L = p['a']*1e-9, p['b']*1e-12, p['EL']*1e-3
    elif cell_type == "FS":
        try:
            a,b,E_L = p['a_i']*1e-9, p['b_i']*1e-12, p['EL_i']*1e-3
        except KeyError:
            a,b,E_L = p['a']*1e-9, p['b']*1e-12, p['EL']*1e-3
    f_e = fexc*(1.-gei)*pconnec*ntot;
    f_i = finh*gei*pconnec*ntot;
    
    # K_e = (1-gei)*ntot
    # K_i = gei*ntot

    mu_Ge = f_e*tau_e*Q_e
    mu_Gi = f_i*tau_i*Q_i
    mu_G = mu_Ge  + mu_Gi + g_L
    tau_eff = C_m / mu_G
    print(finh.shape)
    if w_prec:
        print(w_ad.shape)
        mu_V = (mu_Ge*E_e +  mu_Gi*E_i + g_L*E_L - w_ad) / mu_G
    else:
        mu_V = (mu_Ge*E_e +  mu_Gi*E_i + g_L*E_L - fout*Tw*b + a*E_L) / mu_G
    
    U_e = Q_e / mu_G*(E_e - mu_V)
    U_i = Q_i / mu_G*(E_i - mu_V)
    
    sig_V = np.sqrt(f_e*(U_e*tau_e)*(U_e*tau_e) / (2*(tau_eff + tau_e)) + f_i*(U_i*tau_i)*(U_i*tau_i) / (2*(tau_eff + tau_i)))
    
    tau_V = ((f_e*(U_e*tau_e)*(U_e*tau_e) + f_i*(U_i*tau_i)*(U_i*tau_i)) / 
        (f_e*(U_e*tau_e)*(U_e*tau_e) / (tau_eff + tau_e)  + f_i*(U_i*tau_i)*(U_i*tau_i) / (tau_eff + tau_i)))
    
    tauN_V = tau_V*g_L / C_m

    return mu_V, sig_V, tau_V, tauN_V

def output_rate(params,mu_V, sig_V, tau_V, tauN_V):
    f_out = erfc((eff_thresh(mu_V, sig_V, tauN_V, params) - mu_V) / (np.sqrt(2)*sig_V)) / (2*tau_V)
    # f_out = 0.5*tau_V* (1-erf((eff_thresh(mu_V, sig_V, tauN_V, params) - mu_V) / (np.sqrt(2)*sig_V)))
    return f_out

def eff_thresh_estimate(ydata, mu_V, sig_V, tau_V):
    Veff_thresh = mu_V + np.sqrt(2)*sig_V*erfcinv(ydata*2*tau_V)
    return Veff_thresh

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
    # vthr=P[0]+P[1]*(muV-muV0)/DmuV0+P[2]*(sV-sV0)/DsV0+P[3]*(TvN-TvN0)/DTvN0+P[5]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0)+P[6]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0)+P[7]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0)+P[8]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+P[9]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+P[10]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0;
    
    vthr=P[0]+P[1]*(muV-muV0)/DmuV0+P[2]*(sV-sV0)/DsV0+P[3]*(TvN-TvN0)/DTvN0+P[4]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0)+P[5]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0)+P[6]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0)+P[7]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+P[8]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+P[9]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0;

    frout=.5/TvN*Gl/Cm*(1-erf((vthr-muV)/np.sqrt(2)/sV));
    # frout=.5*Tv*(1-erf((vthr-muV)/(np.sqrt(2)*sV)));
    
    return frout;
##### Maria's stuff #####
def get_rid_of_nans(vve, vvi, adapt, FF, params, cell_type, return_index=False, w_prec=False):
    """
    w_prec : if True use precalculated adaptation from network (adapt) instead of Tw*b*fout
    """
    ve2 = vve.flatten()
    vi2 = vvi.flatten()
    FF2 = FF.flatten()
    adapt2= adapt.flatten()
    print(ve2.shape, vi2.shape, FF2.shape, adapt2.shape)
    # FF2[FF2<1e-9] = 1e-9 

        # Calculate Veff:
    # muV2, sV2, Tv2, TNv2= mu_sig_tau_func(ve2, vi2, adapt2,params,cell_type) 
    muV2, sV2, Tv2, TNv2= mu_sig_tau_func(ve2, vi2, FF2,adapt2,params,cell_type, w_prec=w_prec) 
    Veff = eff_thresh_estimate(FF2,muV2, sV2, Tv2)

    #delete Nan/Infs
    nanindex=np.where(np.isnan(Veff))
    infindex=np.where(np.isinf(Veff))

    bigindex = np.concatenate([nanindex,infindex],axis=1)

    print("this many nans:", len(bigindex[0]))
    ve2=np.delete(ve2,bigindex)
    vi2=np.delete(vi2,bigindex)
    FF2=np.delete(FF2,bigindex)
    adapt2=np.delete(adapt2,bigindex)

    if return_index:
        return ve2, vi2, FF2, adapt2, bigindex
    else:
        return ve2, vi2, FF2, adapt2

def plot_muv(vve, vvi, FF, adapt,params,cell_type, w_prec=False):
    mu_V, sig_V, tau_V, tauN_V =  mu_sig_tau_func(vve, vvi, FF, adapt,params,cell_type, w_prec=w_prec)
    # muV_2 = muV.reshape(50,50)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(30):
        ax.plot(vve[0], mu_V[i]*1e3,marker = '.',label=f'vi={vvi[i,0]:.2f}Hz' )

    plt.xlabel('ve [Hz]')
    plt.ylabel('muV [mV]')
    plt.show()

def plot_veff(vve, vvi, adapt, FF,params,cell_type, w_prec=False):
    mu_V, sig_V, tau_V, tauN_V =  mu_sig_tau_func(vve, vvi, FF, adapt,params,cell_type, w_prec=w_prec)
    Veff = eff_thresh_estimate(FF, mu_V, sig_V, tau_V)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(vvi[:,0])):
        ax.plot( Veff.reshape(50,50)[i]*1e3,label=f'vi={vvi[i,0]:.2f}Hz')

    ax.set_xlabel('ve [Hz]')
    ax.set_ylabel('Veff [mV]')
    plt.show()

def plot_check_fit(file, param_file, adapt_file ,cell_type, P, w_prec=False):
    
    feSim, fiSim, params = np.load(param_file,allow_pickle=True) 
    adapt = np.load(adapt_file).T
    out_rate = np.load(file).T

    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title(f'Transfer function of {cell_type} cell')
    ax.set_ylabel('Output rate (Hz)')
    ax.set_xlabel('Excitatory input (Hz)')
        
    inp_exc = feSim
    vve, vvi = np.meshgrid(feSim, fiSim) #vve has the range_exc along the row, vvi has it accros the column 
    
    # mu_V, sig_V, tau_V,tauN_V = mu_sig_tau_func(vve, vvi, adapt,params,cell_type)
    mu_V, sig_V, tau_V,tauN_V = mu_sig_tau_func(vve, vvi, out_rate,adapt, params,cell_type,w_prec=w_prec )

    fit_rate = output_rate(P,mu_V, sig_V, tau_V, tauN_V)
    ax.plot(inp_exc, out_rate.T, 'ro', label='data');

    ax.plot(inp_exc, fit_rate.T, 'k.', label='fit');

    

    mean_error = np.nanmean(np.sqrt((out_rate - fit_rate)**2))

    ax.text(0.5, 0.95, f'dev: {mean_error:.2f} Hz', transform=ax.transAxes, ha='center')
    plt.show()

def video_check_fit(file, param_file, adapt_file ,cell_type, P, w_prec=True):
    
    ve, vi, params = np.load(param_file,allow_pickle=True)
    adapt = np.load(adapt_file).T
    out_rate = np.load(file).T

    vve, vvi = np.meshgrid(ve, vi)
    # mu_V, sig_V, tau_V,tauN_V = mu_sig_tau_func(vve, vvi, adapt,params,cell_type)
    mu_V, sig_V, tau_V,tauN_V = mu_sig_tau_func(vve, vvi, out_rate,adapt, params,cell_type,w_prec=w_prec)
    fit_rate = output_rate(P,mu_V, sig_V, tau_V, tauN_V)

    # Define the function to update the plot for each frame
    def update(frame):
        ax_anim.clear()
        i = frame  # iterate over different values of i
        ax_anim.set_title(f'vi = {vi[i]:.2f}Hz')
        ax_anim.plot(ve, out_rate[i], 'o', ms=3, label='data')
        ax_anim.plot(ve, fit_rate[i], label='fit')
        ax_anim.set_xlabel('ve [Hz]')
        ax_anim.set_ylabel('vout [Hz]')
        plt.legend()

    # Create a figure and axis
    fig_anim, ax_anim = plt.subplots()

    # Set any initial settings for the plot (if needed)

    # Create the animation
    ani = FuncAnimation(fig_anim, update, frames=len(vi), interval=200,repeat=True)
    return HTML(ani.to_html5_video())

def plot_example_adjust_range(file, param_file, adapt_file ,cell_type, P,w_prec=False, **kwargs): 

    default_args = {'window': 12, 'thresh_pc': 0.9}

    #update arguments
    default_args.update(kwargs)

    window, thresh_pc=default_args['window'],default_args['thresh_pc']

    ve, vi, params = np.load(param_file,allow_pickle=True)
    adapt = np.load(adapt_file).T 
    out_rate = np.load(file).T

    vve, vvi = np.meshgrid(ve, vi)
    mu_V, sig_V, tau_V,tauN_V = mu_sig_tau_func(vve, vvi,out_rate, adapt,params,cell_type,w_prec=w_prec)
    fit_rate = output_rate(P,mu_V, sig_V, tau_V, tauN_V)
    error = np.sqrt((out_rate - fit_rate)**2).T

    fig, ax = plt.subplots()

    x_ticks = np.arange(0, len(ve), 5)
    y_ticks = np.arange(0, len(vi), 5)
    im= ax.imshow(error)

    fig.colorbar(im, ax=ax, label='Error')

    ax.set_xticks(x_ticks,np.round(ve[list(x_ticks)]));
    ax.set_yticks(y_ticks,np.round(vi[list(y_ticks)]));  

    range_exc, range_inh = find_max_error(out_rate, fit_rate, ve, vi, window=window, thresh_pc = thresh_pc)
    
    ve_st, ve_end = range_exc
    vi_st, vi_end = range_inh
    
    y_start, y_end = np.argmin(np.abs(ve - ve_st)),np.argmin(np.abs(ve - ve_end))+1
    x_start, x_end = np.argmin(np.abs(vi - vi_st)),np.argmin(np.abs(vi - vi_end))+1

    # Plot the rectangular area with the maximum mean error on the image

    rect = plt.Rectangle((y_start, x_start), y_end - y_start, x_end - x_start,
                        linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Show the plot

    ax.set_xlabel("ve (Hz)")
    ax.set_ylabel("vi (Hz)")

    ax.invert_yaxis()
    plt.tight_layout()

    plt.show()

def plot_curve(NAME = 'FS-RS', file_rs ='RS-cell0_CONFIG1_fit_2.npy', file_fs= 'FS-cell_CONFIG1_fit_2.npy', 
               use_new=False, **kwargs):
    
    """
    NAME : str, FS-RS
    file_rs, file_fs : str, P coef for the two types of cells
    use_new : if True you can use new parameters

    kwargs : update parameters eg b_e, tau_e, etc
    """
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
    
    #Fitting coefficients
    PRS=np.load(file_rs)
    PFS=np.load(file_fs)

    print(PRS)
    T = 20*1e-3 # time constant

    #Initial Conditions
    fecont=2;
    ficont=10;

    LSfe=[]
    nuev=np.arange(0.00000001,10,step=0.1)

    nuext = 0 
    nuextin = 0.


    params = get_neuron_params_double_cell(NAME, SI_units=True)
    
    if use_new:
        for key, it in kwargs.items():
            params[key] = it

    p = params

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

    #To adjust
    bRS = p['b_e']; #adaptation 
    Te=p['tau_e']; #excitatory synaptic decay
    Ti=p['tau_i']; #inhibitory synaptic decay

    Ele =p['EL_e'] #leak reversal (exc)
    Eli = p['EL_i'] #leak reversal (inh)
    
    # params = get_neuron_params_double_cell(NAME, SI_units=False)

    # p = params
    # Qe,Qi,Te,Ti,Ee,Ei,Cm,twRS,Gl, gei, Ntot, pconnec = p['Q_e']*1e-9,p['Q_i']*1e-9,p['tau_e']*1e-3,p['tau_i']*1e-3,p['E_e']*1e-3,p['E_i']*1e-3, p['Cm']*1e-12, p['tau_w']*1e-3,p['Gl']*1e-9, p['gei'], p['Ntot'], p['p_con']

    # a,bRS,Ele = p['a_e']*1e-9, p['b_e']*1e-12, p['EL_e']*1e-3

    # a,bFS,Eli = p['a_i']*1e-9, p['b_i']*1e-12, p['EL_i']*1e-3

    # print("running: ", sim_name)
    for nue in nuev:

        w=nue*bRS*twRS
        try:
            nui_fix = fixed_point(TF_2, [1.0], args=(nue, nuext, nuextin, PFS, 0., Eli))
        except RuntimeError:
            print(f"runtime error: b_e = {bRS}, tau_e={Te}, tau_i={Ti}")
        TFe_fix = TF_2(nui_fix, nue, nuext, 0., PRS, w, Ele)
        LSfe.append(float(TFe_fix))

    plt.plot(nuev, LSfe, "b-", label=f"b={bRS*1.e+12}")
    plt.plot(nuev, nuev, "k--")
    plt.legend()
    plt.show() 

    # Define a function that returns TF_2(nue) - nue
    def func(nue):
        w = nue * bRS * twRS
        nui_fix = fixed_point(TF_2, [1.0], args=(nue, nuext, nuextin, PFS, 0., Eli), xtol = 1.e-9, maxiter=3000)
        TFe_fix = TF_2(nui_fix, nue, nuext, nuextin, PRS, w, Ele)
        # print("TFe_fix:", TFe_fix)
        # print("nui_fix:", nui_fix)
        return TFe_fix - nue


    # Use fsolve to find the roots of the difference equation
    initial_guess = 10  # Initial guess for the root
    intersection_point = fsolve(func, initial_guess, maxfev=20000)
    print("solution = ", intersection_point)

    nui_fix = fixed_point(TF_2, [intersection_point], args=(intersection_point, nuext, nuextin, PFS, 0., Eli), xtol = 1.e-9, maxiter=3000)

    print("solution nui_fix = ",nui_fix)

def convert_params(params):
    print('cell parameters in SI units')
    # mV to V
    params['EL_e'], params['EL_i'], params['E_e'], params['E_i'] = 1e-3*params['EL_e'],1e-3*params['EL_i'],1e-3*params['E_e'],1e-3*params['E_i']
    
    params['V_th'], params['V_r'], params['V_m'], params['V_cut'] =\
            1e-3*params['V_th'], 1e-3*params['V_r'], 1e-3*params['V_m'], 1e-3*params['V_cut']
    
    params['delta_e'], params['delta_i'] = 1e-3*params['delta_e'], 1e-3*params['delta_i']
    # ms to s
    params['tau_w'],params['tau_e'],params['tau_i'] = 1e-3*params['tau_w'], 1e-3*params['tau_e'],1e-3*params['tau_i']
    # nS to S
    params['a_e'], params['a_i'],params['Q_e'], params['Q_i'], params['Gl'] = 1e-9*params['a_e'],1e-9*params['a_i'],1e-9*params['Q_e'],1e-9*params['Q_i'], 1e-9*params['Gl']
    # pF to F and pA to A
    params['Cm'], params['b_e'],params['b_i'] = 1e-12*params['Cm'], 1e-12*params['b_e'],1e-12*params['b_i']

    return params

def find_max_error(out_rate, fit_rate, ve, vi, window=12, thresh_pc = 0.9):
    """"
    - caclulate the error between estimated (fit_rate) and simulated (out_rate) firing rate
    - finds the areas with the maximum errors and accordingly returns a range of exc and inh excitatory inputs to focus the fit
    - it prioritizes lower ranges (more relevant in the simulations) as long as their estimated error is not lower than a percentage (thesh_pc) of the max error

    out_rate  : FR from the single cells experiments
    fit_rate  : estimated FR through the fitted TF
    ve, vi    : the exc and inh inputs used for the experiments
    window    : size of the window to calculate local maxima
    thresh_pc : percentage of the max local error that should be exceeded to prioritize lower ranges (=1 if no priorities in lower indices is required)

    return 
    range_exc, range_inh : tuples with the ranges for the exc and inh inputs
    """

    error = np.sqrt((out_rate - fit_rate)**2).T
    print('mean error = ', np.mean(error))

    if window > len(ve)/3: #If window too large readjust
        window = int(len(ve)/3)
    rectangular_size = window  

    # Apply a maximum filter to find local maxima in the error array
    local_maxima = maximum_filter(error, size=rectangular_size)

    # Find the indices of local maxima
    max_indices = np.argwhere(local_maxima == error)

    # Initialize variables to store the maximum mean error and its corresponding rectangular area
    max_mean_error_rect = None
    all_errors = []

    # Iterate over all local maxima
    for i, j in max_indices:
        # Calculate the mean error inside the current rectangular area
        mean_error = np.nanmean(error[max(0, i-rectangular_size):min(error.shape[0], i+rectangular_size+1),
                                max(0, j-rectangular_size):min(error.shape[1], j+rectangular_size+1)])
        max_mean_error_rect = (max(0, i-rectangular_size), max(0, j-rectangular_size),
                                min(error.shape[0], i+rectangular_size+1), min(error.shape[1], j+rectangular_size+1))
        a = sum(max_mean_error_rect)
        
        all_errors.append([mean_error, a, max_mean_error_rect])

    all_errors = np.array(all_errors, dtype='object')

    #sort according to indices 
    y=np.argsort(all_errors[:,1],kind='mergesort')
    all_errors = all_errors[y]

    thresh = np.max(all_errors[:,0]) * thresh_pc #90% of the max error

    #choose lower indices if there is no big difference in the errors
    for i in range(all_errors.shape[0]):
        if all_errors[i,0] > thresh:
            max_mean_error_rect = all_errors[i,2]
            break

    x_start, y_start, x_end, y_end = max_mean_error_rect
    range_exc=(ve[y_start],ve[y_end-1])
    range_inh=(vi[x_start],vi[x_end-1])
    
    return range_exc, range_inh

def adjust_ranges(ve, vi, FF, adapt,params,cell_type, range_inh, range_exc, w_prec=False):
    """
    return mu_V, sig_V, tau_V, tauN_V, FF2 that are calculated according to the new ranges
    """
    vve, vvi = np.meshgrid(ve, vi)

    if range_inh:
        des_start, des_end = range_inh
        start, end = np.argmin(np.abs(vi - des_start)),np.argmin(np.abs(vi - des_end))
        rid = range(start,end)
        additional_values = [0,1,3, 5,  -3, -5,-1,-2] #adding some values from the extremities give better results 
        rid = list(additional_values) + list(rid)

    if range_exc:
        des_start, des_end = range_exc
        start,end = np.argmin(np.abs(ve - des_start)) ,np.argmin(np.abs(ve - des_end))
        additional_values = [0,1,3, 5,  -3, -5,-1,-2]
        red = range(start, end)
        red = list(additional_values) +list(red)
    
    if range_inh and range_exc:
        FF= FF[red][:,rid]
        ve2 = vve[red][:,rid].flatten()
        vi2 = vvi[red][:,rid].flatten()
        FF2 = FF.flatten()
        adapt2 = adapt[red][:,rid].flatten()
    elif range_inh:
        FF= FF[:,rid]
        ve2 = vve[:,rid].flatten()
        vi2 = vvi[:,rid].flatten()
        FF2 = FF.flatten()
        adapt2 = adapt[:,rid].flatten()
    elif range_exc: 
        FF = FF[red]
        ve2 = vve[red].flatten()
        vi2 = vvi[red].flatten()
        FF2 = FF.flatten()
        adapt2 = adapt[red].flatten()

    # mu_V, sig_V, tau_V, tauN_V = mu_sig_tau_func(ve2, vi2, adapt2,params,cell_type)
    mu_V, sig_V, tau_V, tauN_V = mu_sig_tau_func(ve2, vi2, FF2,adapt2, params,cell_type,w_prec=w_prec)

    return mu_V, sig_V, tau_V, tauN_V, FF2

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

def run_MF(CELLS, AmpStim, PRS, PFS, Iext=0, TotTime=2):

    p = get_neuron_params_double_cell(CELLS, SI_units = True)
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
        # vthr=P[0]+P[1]*(muV-muV0)/DmuV0+P[2]*(sV-sV0)/DsV0+P[3]*(TvN-TvN0)/DTvN0+P[5]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0)+P[6]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0)+P[7]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0)+P[8]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+P[9]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+P[10]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0;
        
        vthr=P[0]+P[1]*(muV-muV0)/DmuV0+P[2]*(sV-sV0)/DsV0+P[3]*(TvN-TvN0)/DTvN0+P[4]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0)+P[5]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0)+P[6]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0)+P[7]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+P[8]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+P[9]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0;

        frout=.5/TvN*Gl/Cm*(1-erf((vthr-muV)/np.sqrt(2)/sV));
        
        return frout;
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

    #Time
    tfinal=TotTime
    dt=0.0001
    t = np.linspace(0, tfinal, int(tfinal/dt))

    # Additive Noise
    v_drive = Iext
    sigma=3.5
    os_noise = sigma*OU(tfinal) + v_drive

    #Create the kick
    time_peek = 200.
    TauP=20 #20

    plat = TotTime*1000 - time_peek - TauP #100
    plat = 900
    # print("plat=", plat)s
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
    T = 20*1e-3 # time constant

    #Initial Conditions
    fecont=6;
    ficont=13;
    w=fecont*bRS*twRS

    LSw=[]
    LSfe=[]
    LSfi=[]
    if AmpStim>0:
        print("Input = ", AmpStim)
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
    
    return np.mean(LSfe[int(0.5*len(LSfe)):]), np.mean(LSfi[int(0.5*len(LSfi)):])


################################################################
##### Now fitting to Transfer Function data
################################################################
from scipy.optimize import basinhopping

def make_fit_from_data(DATA,cell_type, params_file, adapt_file, range_exc=None, range_inh=None,w_prec=False, **kwargs):
    """
    DATA                : (str) directory for the npy file containing the firing rates from the simulations
    cell                : (str) cell type to choose the correct parameters of the fit (ex: FS or RS)
    params_file         : directory to the file containing the ranges and the params 
    adapt_file          : directory to the file containing the adaptation from the experiments
    range_exc,range_inh : the ranges to focus the fit of the TF
    w_prec              : if True use precalculated adaptation in calculations of subthreshold metrics (instead of w=Tw*b*FF)

    additional args:
    name_add            : (str) additional name for the file where the P will be saved 
    loop_n              : (int) how many loops to iterate to minimize the mean error (estimated-simulated)**2
    window              : (int) size of the rectangle where the local maxima will be estimated
    thresh_pc           : (float) percentage for the threshold used in the prioritization of the lower input ranges
    tol,maxiter,method  : tolerance, max iterations and method used for the fit of the v threshold (vthr_) and TF (tf_)

    returns the P coefficients
    """
    default_args = {'loop_n': 10,'window': 12,'thresh_pc': 0.9, 'name_add': '',
        'vthr_tol' : 1e-17, 'vtrh_maxiter': 30000, 'vthr_method': 'SLSQP',
        'tf_tol' : 1e-17, 'tf_maxiter': 30000, 'tf_method':'nelder-mead', 'MF_comp': False, 'seed': 10}
    
    #update arguments
    default_args.update(kwargs)

    loop_n, window, thresh_pc, name_add = default_args['loop_n'],default_args['window'],default_args['thresh_pc'], default_args['name_add']
    vthr_tol,vtrh_maxiter,vthr_method = default_args['vthr_tol'],default_args['vtrh_maxiter'],default_args['vthr_method']
    tf_tol, tf_maxiter, tf_method = default_args['tf_tol'],default_args['tf_maxiter'],default_args['tf_method']
    MF_comp, seed = default_args['MF_comp'], default_args['seed']

    
    FF=np.load(DATA).T #has shape ve*vi bur with the transpose you put the ve to change accross the row, and the vi across the column 
    adapt = np.load(adapt_file).T
    ve, vi, params = np.load(params_file,allow_pickle=True) 
    vve, vvi = np.meshgrid(ve, vi)

    #remove nan and inf values
    ve2, vi2, FF2, adapt2 = get_rid_of_nans(vve, vvi, adapt, FF, params, cell_type, w_prec=w_prec)

    #calculate subthresh
    mu_V, sig_V, tau_V, tauN_V = mu_sig_tau_func(ve2, vi2, FF2,adapt2,params,cell_type, w_prec=w_prec)
    Veff_thresh = eff_thresh_estimate(FF2,mu_V, sig_V, tau_V)

    # fitting first order Vthr on the phenomenological threshold space
    print("fitting first order V threshold..")

    params_init = np.ones(10)*1e-3

    def res_func(params):
        vthresh = eff_thresh(mu_V, sig_V, tauN_V, params)
        res = np.mean((Veff_thresh - vthresh)**2)
        return res

    fit = minimize(res_func, params_init, 
               method=vthr_method,tol= vthr_tol, options={ 'disp': True, 'maxiter':vtrh_maxiter, 'seed': seed})
    
    minimizer_kwargs = {
    'method': vthr_method,
    'tol': vthr_tol,
    'options': {'disp': True, 'maxiter': vtrh_maxiter}}

    # Perform basinhopping optimization
    # fit = basinhopping(res_func, params_init, minimizer_kwargs=minimizer_kwargs, niter_success=10)

    print("P = ", fit['x'])
    
    print("Fitting Transfer Function..")

    params_init2 = fit['x']
    
    # LOOP 
    params_all = []
    for i in range(loop_n):
        print("loop n:",i)
        #adjust range
        if range_inh or range_exc:
            mu_V, sig_V, tau_V,tauN_V, FF2 = adjust_ranges(ve, vi, FF,adapt,params,cell_type, range_inh, range_exc, w_prec=w_prec)
        
        def res2_func(params):
            res2 = np.mean((output_rate(params,mu_V, sig_V, tau_V, tauN_V) - FF2)**2)
            return res2

        fit2 = minimize(res2_func, params_init2, 
                        method=tf_method,tol= tf_tol , options={'disp': True, 'maxiter':tf_maxiter, 'seed': seed})
        
        minimizer_kwargs = {
        'method': tf_method,
        'tol': tf_tol,
        'options': {'disp': True, 'maxiter': tf_maxiter}}

        # Perform basinhopping optimization
        # fit2 = basinhopping(res2_func, params_init2, minimizer_kwargs=minimizer_kwargs,niter_success=10, T=10)
        P = fit2['x']

        #originals - calculate mean error
        # muV, sigV, tauV, tauNV = mu_sig_tau_func(vve, vvi, adapt,params,cell_type)
        muV, sigV, tauV, tauNV = mu_sig_tau_func(vve, vvi, FF,adapt, params,cell_type, w_prec=w_prec)
        fit_rate = output_rate(P,muV, sigV, tauV, tauNV)
        mean_error = np.mean(np.sqrt((FF - fit_rate)**2)) 

        #renew ranges and params
        if loop_n>1:
            range_exc, range_inh = find_max_error(FF, fit_rate, ve, vi, window=window, thresh_pc = thresh_pc)
            params_init2 = P

        params_all.append([P, mean_error])
        i+=1
        seed+=10

    params_all = np.array(params_all,dtype='object')
    P = params_all[np.argmin(params_all[:,1])][0] #keep the one with the smallest mean error
    
    file_name = DATA.replace('ExpTF_','')
    filename = file_name.replace('.npy', f'_fit.npy')
    print('coefficients saved in ', filename)
    np.save(filename, np.array(P))

    return P

def load_network_mean(CELLS, path_net):
        #load network
    fr_inh=[]
    fr_exc=[]
    for file in os.listdir(path_net):
        if file.startswith(CELLS + '_mean'):
            if "inh" in file:
                mean_fr, amp, _ = np.load(path_net+file, allow_pickle=True)
                fr_inh.append([mean_fr, amp])
            elif "exc" in file:
                mean_fr, amp, _ = np.load(path_net+file, allow_pickle=True)
                fr_exc.append([mean_fr, amp])
    fr_exc = np.array(sorted(fr_exc, key=lambda x: x[1]))
    fr_inh = np.array(sorted(fr_inh, key=lambda x: x[1]))
    fr_both = np.column_stack((fr_inh[:,0], fr_exc ))
    
    inputs = fr_both[:,-1]

    return fr_both, inputs

def calculate_mf_difference(CELLS, fr_both, inputs, PRS, PFS):
    mean_both =[]
    for AmpStim in inputs:
        mean_exc, mean_inh = run_MF(CELLS, AmpStim, PRS, PFS, Iext=0, TotTime=2)
        mean_both.append([mean_inh, mean_exc, AmpStim])

    dif_arr = np.abs(fr_both - np.array(mean_both))

    if dif_arr[:,-1].any() !=0:
        raise Exception("difference of inputs should be 0 but it is not")

    dif_arr[:,2] = inputs
    print("Whole difference: ", dif_arr)
    print("mean difference exc: ", np.mean(dif_arr[:,1]))
    print("mean difference inh: ", np.mean(dif_arr[:,0]))
    
    dif = np.mean(dif_arr[:,:2])

    return dif 

def fit_with_MF(file, CELLS, param_file, adapt_file,max_iter = 30, dif = 5, range_exc=None, range_inh=None, window=15, loop_n=1, w_prec=True,  path_net= "./net_compar/"):
    
    #Generate random seeds
    random_seeds = np.random.randint(0, 1000, size=max_iter)
    rs = 0

    PRS = make_fit_from_data_fede(file, 'RS', param_file, adapt_file,range_exc=range_exc, range_inh=range_inh, window=window, loop_n=loop_n, w_prec=w_prec, seed=random_seeds[rs])
    PFS = make_fit_from_data_fede(file, 'FS', param_file, adapt_file,range_exc=range_exc, range_inh=range_inh, window=window, loop_n=loop_n, w_prec=w_prec, seed=random_seeds[rs])

    #load network
    fr_both, inputs = load_network_mean(CELLS, path_net)

    # loop until difference is smaller than 2
    iter = 0
    while dif>2:
        # run mf
        dif = calculate_mf_difference(fr_both, inputs, PRS, PFS)

        if iter>max_iter:
            print(f"Max iterations ({max_iter}) reached")
            break
        
        rs+=1
            
        PRS = make_fit_from_data_fede(file, 'RS', param_file, adapt_file,range_exc=None, range_inh=None, window=15, loop_n=1, w_prec=True, seed=random_seeds[rs])
        PFS = make_fit_from_data_fede(file, 'FS', param_file, adapt_file,range_exc=None, range_inh=None, window=15, loop_n=1, w_prec=True, seed=random_seeds[rs])

        
        iter+=1

import argparse
if __name__=='__main__':
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     '=================================================='
     '=====> FIT of the transfer function =============='
     '=== and theoretical objects for the TF relation =='
     '=================================================='
     """,
              formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-f', "--FILE",help="file name of numerical TF data",\
                        default='./data/ExpTF_exc_10x10_trial.npy')
    parser.add_argument("--cell", help="type of cell",\
                    default='RS')
    parser.add_argument("--params_file", help="parameter file",\
                    default='./data/params_range_trial.npy')
    parser.add_argument("--With_Square",help="Add the square terms in the TF formula"+\
                        "\n then we have 7 parameters",\
                         action="store_true")
    args = parser.parse_args()

    make_fit_from_data(args.FILE,  args.cell, args.params_file, with_square_terms=args.With_Square)

    make_fit_from_data_2(args.FILE,  args.cell, args.params_file, with_square_terms=args.With_Square)


