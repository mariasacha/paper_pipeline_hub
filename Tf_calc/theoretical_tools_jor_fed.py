import numpy as np
import scipy.special as sp_spec
import scipy.integrate as sp_int
from scipy.optimize import minimize, curve_fit
import sys
from scipy.special import erf, erfc, erfcinv
from scipy.ndimage import maximum_filter


def MPF(fexc,finh,adapt, params, cell_type):

    p = params
    Qe,Qi,Te,Ti,Ee,Ei,Cm,Tw,Gl, gei, Ntot = p['Q_e']*1e-9,p['Q_i']*1e-9,p['tau_e']*1e-3,p['tau_i']*1e-3,p['E_e']*1e-3,p['E_i']*1e-3, p['Cm']*1e-12, p['tau_w']*1e-3,p['Gl']*1e-9, p['gei'], p['Ntot']

    if cell_type == "RS":
        a,b,El = p['a_e']*1e-9, p['b_e']*1e-12, p['EL_e']*1e-3
    elif cell_type == "FS":
        a,b,El = p['a_i']*1e-9, p['b_i']*1e-12, p['EL_i']*1e-3


    Nexc,Ninh = Ntot*(1-gei), Ntot*gei

    if 'P' in params.keys():
        P = params['P']
    else: # no correction
        P = [-45e-3]
        for i in range(1,11):
            P.append(0)
    
    fexc[fexc<1e-9]=1e-9
    fe = fexc*Nexc
    finh[finh<1e-9]=1e-9
    fi = finh*Ninh

    muGi = Qi*Ti*fi
    muGe = Qe*Te*fe
    muG = Gl+muGe+muGi
    # muV = (muGe*Ee+muGi*Ei+Gl*El - fout*Tw*b + a*El)/(muG+a)
    muV = (muGe*Ee+muGi*Ei+Gl*El - adapt)/(muG+a)
    
    muGn = muG/Gl
    Tm = Cm/muG
    
    Ue =  Qe/muG*(Ee-muV)
    Ui = Qi/muG*(Ei-muV)

    sV = np.sqrt(fe*(Ue*Te)*(Ue*Te)/2./(Te+Tm)+fi*(Ui*Ti)*(Ui*Ti)/2./(Ti+Tm))

    Tv = ( fe*(Ue*Te)*(Ue*Te) + fi*(Qi*Ui)*(Qi*Ui)) /( fe*(Ue*Te)*(Ue*Te)/(Te+Tm) + fi*(Qi*Ui)*(Qi*Ui)/(Ti+Tm) )
    TvN = Tv*Gl/Cm

    return muV, sV, Tv, TvN

def pheV(fout, muV, sV, Tv):
    fout[fout<0]=1e-9
    Tv[Tv<0]=1e-9
    return np.sqrt(2)*sV * erfcinv( 2*Tv*fout ) + muV # Zerlaut 2017
    # return np.sqrt(2)*sV * erfcinv( Tv*fout ) + muV # to widen the definition range


def get_rid_of_nans(vve, vvi, FF, params, cell_type):
    ve2 = vve.flatten()
    vi2 = vvi.flatten()
    FF2 = FF.flatten()

        # Calculate Veff:
    muV2, sV2, Tv2, TvN2 = MPF(ve2, vi2, FF2, params, cell_type)

    Veff = pheV(FF2, muV2, sV2, Tv2)

    #delete Nan/Infs
    nanindex=np.where(np.isnan(Veff))
    infindex=np.where(np.isinf(Veff))

    bigindex = np.concatenate([nanindex,infindex],axis=1)

    ve2=np.delete(ve2,bigindex)
    vi2=np.delete(vi2,bigindex)
    FF2=np.delete(FF2,bigindex)
    print(ve2.shape)

    #Keep the good ones
    muV_fit, sV_fit, Tv_fit, TvN_fit = MPF(ve2, vi2, FF2, params, cell_type)

    Veff_fit = pheV(FF2, muV_fit, sV_fit, Tv_fit)

    nan_still = np.isnan(Veff_fit).any()
    inf_still = np.isinf(Veff_fit).any()

    if nan_still or inf_still:

        print("still nans or infs")

        nanindex=np.where(np.isnan(Veff_fit))
        infindex=np.where(np.isinf(Veff_fit))

        bigindex = np.concatenate([nanindex,infindex],axis=1)

        ve2=np.delete(ve2,bigindex)
        vi2=np.delete(vi2,bigindex)
        FF2=np.delete(FF2,bigindex)
        print(ve2.shape)

        #Keep the good ones
        muV_fit, sV_fit, Tv_fit, TvN_fit = MPF(ve2, vi2, FF2, params, cell_type)

        Veff_fit = pheV(FF2, muV_fit, sV_fit, Tv_fit)
        
        nan_still = np.isnan(Veff_fit).any()
        inf_still = np.isinf(Veff_fit).any()

        if nan_still or inf_still:
            raise Exception("still nans or infs after the second round -- check")
    
    return muV_fit, sV_fit, Tv_fit, TvN_fit, Veff_fit

def TF(P, muV, sV, Tv, TvN):
    # the transfer function

    fout = 1/(2*Tv) * erfc( (Vthre(P, muV, sV, TvN) - muV)/(np.sqrt(2)*sV) )
    
    # fout = np.where(fout<0, 1e-9, fout)
    fout[fout<0]=0
    return fout


def Vthre(P, muV, sV, TvN):
    # calculating the effective threshold potential with a general second order polynomial of the membrane moments (mu,sigma,tau)
    # normalizing moments:
    muV0 = -60e-3;
    DmuV0 = 10e-3;
    sV0 = 4e-3;
    DsV0 = 6e-3;
    TvN0 = 0.5;
    DTvN0 = 1.;
    
    # first order polynomial
    Vo1 = P[0] + P[1]*(muV-muV0)/DmuV0 + P[2]*(sV-sV0)/DsV0 + P[3]*(TvN-TvN0)/DTvN0
    # second order polynomial
    # Vo2 = P[4]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0) + P[5]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0 + P[6]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0 + P[7]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0) + P[8]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0  + P[9]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0);
    Vo2 = P[4]*((muV-muV0)/DmuV0)**2 + P[5]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0 + P[6]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0 + P[7]*((sV-sV0)/DsV0)**2 + P[8]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0  + P[9]*((TvN-TvN0)/DTvN0)**2;

    return Vo1 + Vo2

#fede functions

def eff_thresh(mu_V, sig_V, tauN_V, params):
    P_0, P_mu, P_sig, P_tau, P_mu2, P_sig2, P_tau2, P_mu_sig, P_mu_tau, P_sig_tau = params
    V0 = P_0
    mu_0 = -60.0 * 1e-3
    mu_d = 0.01
    sig_0 = 0.004
    sig_d = 0.006
    tau_0 = 0.5
    tau_d = 1.0

    V1 = (P_mu * (mu_V - mu_0) / mu_d +
          P_sig * (sig_V - sig_0) / sig_d +
          P_tau * (tauN_V - tau_0) / tau_d)
    V2 = (P_mu2 * ((mu_V - mu_0) / mu_d)**2 +
          P_sig2 * ((sig_V - sig_0) / sig_d)**2 +
          P_tau2 * ((tauN_V - tau_0) / tau_d)**2 +
          P_mu_sig * ((mu_V - mu_0) / mu_d) * ((sig_V - sig_0) / sig_d) +
          P_mu_tau * ((mu_V - mu_0) / mu_d) * ((tauN_V - tau_0) / tau_d) +
          P_sig_tau * ((sig_V - sig_0) / sig_d) * ((tauN_V - tau_0) / tau_d))
    return V0 + V1 + V2

def mu_sig_tau_func(f_e, f_i2, w_ad,params,cell_type):
    p = params
    Q_e,Q_i2,tau_e,tau_i,E_e,E_i,C_m,Tw,g_L, gei, ntot = p['Q_e']*1e-9,p['Q_i']*1e-9,p['tau_e']*1e-3,p['tau_i']*1e-3,p['E_e']*1e-3,p['E_i']*1e-3, p['Cm']*1e-12, p['tau_w']*1e-3,p['Gl']*1e-9, p['gei'], p['Ntot']

    if cell_type == "RS":
        a,b,E_L = p['a_e']*1e-9, p['b_e']*1e-12, p['EL_e']*1e-3
    elif cell_type == "FS":
        a,b,E_L = p['a_i']*1e-9, p['b_i']*1e-12, p['EL_i']*1e-3

    K_e = (1-gei)*ntot

    # tau_i = 5. * 1e-3
    tau_i2 = tau_i
    # Q_i2 = 1.5 * 1e-9
    K_i2 = gei*ntot


    mu_Ge = f_e * K_e * tau_e * Q_e
    
    mu_Gi2 = f_i2 * K_i2 * tau_i2 * Q_i2
    
    mu_G = mu_Ge  + mu_Gi2 + g_L
    tau_eff = C_m / mu_G
    
    mu_V = (mu_Ge * E_e +  mu_Gi2 * E_i + g_L * E_L - w_ad) / mu_G
    
    U_e = Q_e / mu_G * (E_e - mu_V)
    U_i2 = Q_i2 / mu_G * (E_i - mu_V)
    
    sig_V = np.sqrt(
        K_e * f_e * (U_e * tau_e)**2 / (2 * (tau_eff + tau_e)) +
        
        K_i2 * f_i2 * (U_i2 * tau_i2)**2 / (2 * (tau_eff + tau_i2))
    )
    
    tau_V = (
        (K_e * f_e * (U_e * tau_e)**2 +
         
         K_i2 * f_i2 * (U_i2 * tau_i2)**2
        ) / 
        (K_e * f_e * (U_e * tau_e)**2 / (tau_eff + tau_e)  +
         K_i2 * f_i2 * (U_i2 * tau_i2)**2 / (tau_eff + tau_i2)
        )
    )

    return mu_V, sig_V, tau_V

def output_rate(params,mu_V, sig_V, tau_V, tauN_V):
    f_out = erfc((eff_thresh(mu_V, sig_V, tauN_V, params) - mu_V) / (np.sqrt(2) * sig_V)) / (2 * tau_V)
    return f_out

def eff_thresh_estimate(ydata, mu_V, sig_V, tau_V):
    Veff_thresh = mu_V + np.sqrt(2) * sig_V * erfcinv(ydata * 2 * tau_V)
    return Veff_thresh

##### plot stuff #####
def plot_muv(vve, vvi, adapt):
    mu_V, sig_V, tau_V =  mu_sig_tau_func(vve, vvi, adapt)
    # muV_2 = muV.reshape(50,50)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(30):
        ax.plot(ve, mu_V[i]*1e3,marker = '.',label=f'vi={vi[i]:.2f}Hz' )

    plt.xlabel('ve [Hz]')
    plt.ylabel('muV [mV]')
    plt.show()

def plot_veff(vve, vvi, adapt, FF):
    mu_V, sig_V, tau_V =  mu_sig_tau_func(vve, vvi, adapt)
    Veff = eff_thresh_estimate(out_rate, mu_V, sig_V, tau_V)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(vi)):
        ax.plot( Veff.reshape(50,50)[i]*1e3,label=f'vi={vi[i]:.2f}Hz')

    ax.set_xlabel('ve [Hz]')
    ax.set_ylabel('Veff [mV]')
    plt.show()

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

def find_max_error(out_rate, fit_rate, ve, vi):

    error = np.sqrt((out_rate - fit_rate)**2).T
    print('mean error = ', np.mean(error))
    rectangular_size = 8  # You can adjust this value as needed

    # Apply a maximum filter to find local maxima in the error array
    local_maxima = maximum_filter(error, size=rectangular_size)

    # Find the indices of local maxima
    max_indices = np.argwhere(local_maxima == error)

    # Initialize variables to store the maximum mean error and its corresponding rectangular area
    max_mean_error = 0
    max_mean_error_rect = None

    # Iterate over all local maxima
    for i, j in max_indices:
        # Calculate the mean error inside the current rectangular area
        mean_error = np.mean(error[max(0, i-rectangular_size):min(error.shape[0], i+rectangular_size+1),
                                max(0, j-rectangular_size):min(error.shape[1], j+rectangular_size+1)])
        
        # Check if the mean error is greater than the current maximum mean error
        if mean_error > max_mean_error:
            max_mean_error = mean_error
            max_mean_error_rect = (max(0, i-rectangular_size), max(0, j-rectangular_size),
                                min(error.shape[0], i+rectangular_size+1), min(error.shape[1], j+rectangular_size+1))

    # Plot the rectangular area with the maximum mean error on the image
    x_start, y_start, x_end, y_end = max_mean_error_rect
    range_exc=(ve[y_start],ve[y_end-1])
    range_inh=(vi[x_start],vi[x_end-1])
    
    return range_exc, range_inh

def adjust_ranges(ve, vi, FF, adapt,params,cell_type, range_inh, range_exc):
    vve, vvi = np.meshgrid(ve, vi)

    if range_inh:
        des_start, des_end = range_inh
        start, end = np.argmin(np.abs(vi - des_start)),np.argmin(np.abs(vi - des_end))
        rid = range(start,end)
        vi_max = vi.max()
        additional_values = [0,1,3, 5,  -3, -5,-1,-2]
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

    mu_V, sig_V, tau_V = mu_sig_tau_func(ve2, vi2, adapt2,params,cell_type)

    # Veff_thresh = eff_thresh_estimate(FF2,mu_V, sig_V, tau_V)

    # #delete Nan/Infs
    # nanindex=np.where(np.isnan(Veff_thresh))
    # infindex=np.where(np.isinf(Veff_thresh))

    # bigindex = np.concatenate([nanindex,infindex],axis=1)

    # ve2=np.delete(ve2,bigindex)
    # vi2=np.delete(vi2,bigindex)
    # FF2=np.delete(FF2,bigindex)
    # adapt2=np.delete(adapt2,bigindex)

    mu_V, sig_V, tau_V = mu_sig_tau_func(ve2, vi2, adapt2,params,cell_type)

    return mu_V, sig_V, tau_V, FF2
################################################################
##### Now fitting to Transfer Function data
################################################################

def make_fit_from_data_fede(DATA,cell_type, params_file, adapt_file, range_exc=None, range_inh=None):

    FF=np.load(DATA) #has shape ve*vi
    adapt = np.load(adapt_file)
    ve, vi, params = np.load(params_file,allow_pickle=True) 
    vve, vvi = np.meshgrid(ve, vi)

    ve2 = vve.flatten()
    vi2 = vvi.flatten()
    FF2 = FF.flatten()
    adapt2 = adapt.flatten()

    mu_V, sig_V, tau_V = mu_sig_tau_func(ve2, vi2, adapt2,params,cell_type)

    Veff_thresh = eff_thresh_estimate(FF2,mu_V, sig_V, tau_V)

    #delete Nan/Infs
    nanindex=np.where(np.isnan(Veff_thresh))
    infindex=np.where(np.isinf(Veff_thresh))

    bigindex = np.concatenate([nanindex,infindex],axis=1)

    ve2=np.delete(ve2,bigindex)
    vi2=np.delete(vi2,bigindex)
    FF2=np.delete(FF2,bigindex)
    adapt2=np.delete(adapt2,bigindex)

    mu_V, sig_V, tau_V = mu_sig_tau_func(ve2, vi2, adapt2,params,cell_type)
    Veff_thresh = eff_thresh_estimate(FF2,mu_V, sig_V, tau_V)

    g_L = 10 * 1e-9   # 30.0  /1.3
    C_m = 200 * 1e-12  # 281.0 /1.1
    tauN_V = tau_V * g_L / C_m

    # fitting first order Vthr on the phenomenological threshold space
    print("fitting first order V threshold..")

    params_init = np.ones(10) * 1e-3

    def res_func(params):
        vthresh = eff_thresh(mu_V, sig_V, tauN_V, params)
        res = np.mean((Veff_thresh - vthresh)**2)
        return res

    fit = minimize(res_func, params_init, 
               method='nelder-mead',tol= 1e-17, options={ 'disp': True, 'maxiter':30000})
    
    print("P = ", fit['x'])
    
    print("Fitting Transfer Function..")

    params_init2 = fit['x']
    
    # LOOP 
    params_all = []
    for i in range(10):
        print("loop n:",i)
        #adjust range
        if range_inh or range_exc:
            mu_V, sig_V, tau_V, FF2 = adjust_ranges(ve, vi, FF,adapt,params,cell_type, range_inh, range_exc)
            tauN_V = tau_V * g_L / C_m

        
        def res2_func(params):
            res2 = np.mean((output_rate(params,mu_V, sig_V, tau_V, tauN_V) - FF2)**2)
            return res2

        fit2 = minimize(res2_func, params_init2, 
                        method='nelder-mead',tol= 1e-17 , options={'disp': True, 'maxiter':10000})
        
        P = fit2['x']

        #originals - calculate mean error
        muV, sigV, tauV = mu_sig_tau_func(vve, vvi, adapt,params,cell_type)
        tauNV = tauV * g_L / C_m
        fit_rate = output_rate(P,muV, sigV, tauV, tauNV)
        mean_error = np.mean(np.sqrt((FF - fit_rate)**2)) 

        #renew ranges and params
        range_exc, range_inh = find_max_error(FF, fit_rate, ve, vi)
        params_init2 = P

        params_all.append([P, mean_error])
        i+=1

    params_all = np.array(params_all,dtype='object')
    P = params_all[np.argmin(params_all[:,1])][0] #keep the one with the smallest mean error
    
    filename = DATA.replace('.npy', f'_{cell_type}_fede_fit.npy')
    print('coefficients saved in ', filename)
    np.save(filename, np.array(P))

    return P


def make_fit_from_data_2(DATA,cell_type, params_file, with_square_terms=False ):

    FF=np.load(DATA)
    ve, vi, params = np.load(params_file,allow_pickle=True) 
    vve, vvi = np.meshgrid(ve, vi)

    muV, sV, Tv, TvN = MPF(vve, vvi, FF, params, cell_type)

    # Veff = pheV(FF, muV, sV, Tv)
    muV_fit, sV_fit, Tv_fit, TvN_fit, Veff_fit = get_rid_of_nans(vve, vvi, FF, params, cell_type)

    # fitting first order Vthr on the phenomenological threshold space
    print("fitting first order V threshold..")

    params_init = np.ones(10) * 1e-3

    def res_func(params):
        vthresh = eff_thresh(muV_fit, sV_fit, TvN_fit, params)
        res = np.mean((Veff_fit - vthresh)**2)
        return res

    fit = minimize(res_func, params_init, 
               method='SLSQP', options={'ftol': 1e-17, 'disp': True, 'maxiter':3000})
    
    print("P = ", fit['x'])
    
    print("Fitting Transfer Function..")

    params_init2 = fit['x']

    def res2_func(params):
        res2 = np.mean((output_rate(params) - ydata2)**2)
        return res2

    fit2 = minimize(res2_func, params_init2, 
                    method='nelder-mead', options={'disp': True, 'maxiter':10000})

    # def Res(P):
    #     # fitting first order Vthr on the phenomenological threshold space 
    #     return np.mean((Veff_fit - Vthre(np.concatenate([P,[0]*6]), muV_fit, sV_fit, TvN_fit))**2 )

    # res = minimize(Res, [Veff_fit.mean(),1e-3,1e-3,1e-3], method='nelder-mead', tol=1e-15, options={'disp':True,'maxiter':20000})
    # P1 = np.array(res.x)
    # print("P1 = ", P1)

    # if with_square_terms:
    #     print("fitting second order V threshold..")
    #     def Res_2(P): 
    #         # fit the second order parameters on Vthre ( not necessary most of the time!!!! -> SKIP )
    #         return np.mean( (Veff_fit - Vthre(np.concatenate([P1,P]), muV_fit, sV_fit, TvN_fit))**2 )
    #     res = minimize(Res_2, [1e-9]*6, method='nelder-mead', tol=1e-20, options={'disp':True,'maxiter':20000})
    #     # res = minimize(Res, [0]*6, method='SLSQP', options={'ftol':1e-20,'disp':True,'maxiter':20000})
    #     P2 = np.array(res.x)
    #     P = np.concatenate([P1, P2 ])
    # else:
    #     P = np.concatenate([P1, [0]*6])

    # print("P = ", P)
    # print("Fitting Transfer Function..")
    # def Res_TF(P):
    # #     return np.mean( (TC_fit - TF(P, muV_fit, sV_fit, Tv_fit, TvN_fit))**2 )
    #     return np.mean( (FF - TF(P, muV, sV, Tv, TvN))**2 )
    # res = minimize(Res_TF, P, method='nelder-mead', tol=1e-15, options={'disp':True,'maxiter':40000})

    # print(P)

        # then we save it:
    filename = DATA.replace('.npy', f'_{cell_type}_fit.npy')
    print('coefficients saved in ', filename)
    np.save(filename, np.array(P))

    return P
    
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


