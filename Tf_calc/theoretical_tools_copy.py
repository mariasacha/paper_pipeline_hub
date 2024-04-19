import numpy as np
import scipy.special as sp_spec
import scipy.integrate as sp_int
from scipy.optimize import minimize, curve_fit
import sys
from scipy.special import erf, erfc, erfcinv


# Jorin's functions
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


def pseq_params(params, cell_type):
    # Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
    # Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    # Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
    # for key, dval in zip(['Ntot', 'pconnec', 'gei'], [1, 2., 0.5]):
    #     if not key in params.keys():
    #         params[key] = dval

    if 'P' in params.keys():
        P = params['P']
    else: # no correction
        P = [-45e-3]
        for i in range(1,11):
            P.append(0)
    if cell_type=='FS':
        params['El'] = params['EL_i']
    elif cell_type == 'RS':
        params['El'] = params['EL_e']
    
    # params['Q_e'], params['tau_e'], params['E_e'], params['Q_i'], params['tau_i'], params['E_i'], params['El']=\
    

    # params['Gl'] =  
    # # pF to F and pA to A
    # params['Cm']= 

    # return Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10
    return params['Q_e']*1e-9, params['tau_e']*1e-3, params['E_e']*1e-3, params['Q_i']*1e-9, params['tau_i']*1e-3, params['E_i']*1e-3, 1e-9*params['Gl'], 1e-12*params['Cm'], params['El']*1e-3, params['Ntot'], params['p_con'], params['gei'], P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10]

def get_fluct_regime_vars(Fe, Fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    Fe[Fe<1e-9]=1e-9
    Fi[Fi<1e-9]=1e-9
    
    fe = Fe*(1.-gei)*pconnec*Ntot # default is 1 !!
    fi = Fi*gei*pconnec*Ntot

    muGe, muGi = Qe*Te*fe, Qi*Ti*fi
    muG = Gl+muGe+muGi
    muV = (muGe*Ee+muGi*Ei+Gl*El)/muG
    muGn, Tm = muG/Gl, Cm/muG

    Ue, Ui = Qe/muG*(Ee-muV), Qi/muG*(Ei-muV)

    sV = np.sqrt(\
                 fe*(Ue*Te)**2/2./(Te+Tm)+\
                 fi*(Qi*Ui)**2/2./(Ti+Tm))

    fe, fi = fe+1e-9, fi+1e-9 # just to insure a non zero division, 
    Tv = ( fe*(Ue*Te)**2 + fi*(Qi*Ui)**2 ) /( fe*(Ue*Te)**2/(Te+Tm) + fi*(Qi*Ui)**2/(Ti+Tm) )
    TvN = Tv*Gl/Cm

    return muV, sV+1e-12, muGn, TvN

def mean_and_var_conductance(Fe, Fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    fe = Fe*(1.-gei)*pconnec*Ntot # default is 1 !!
    fi = Fi*gei*pconnec*Ntot
    return Qe*Te*fe, Qi*Ti*fi, Qe*np.sqrt(Te*fe/2.), Qi*np.sqrt(Ti*fi/2.)


### FUNCTION, INVERSE FUNCTION
def erfc_func(muV, sV, TvN, Vthre, Gl, Cm):
    return .5/TvN*Gl/Cm*\
      sp_spec.erfc((Vthre-muV)/np.sqrt(2)/sV)

def effective_Vthre(Y, muV, sV, TvN, Gl, Cm):
    Vthre_eff = muV+np.sqrt(2)*sV*sp_spec.erfcinv(\
                    Y*2.*TvN*Cm/Gl) # effective threshold
    return Vthre_eff


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
    Vo2 = P[4]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0) + P[5]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0 + P[6]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0 + P[7]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0) + P[8]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0  + P[9]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0);

    return Vo1 + Vo2


def threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    """
    setting by default to True the square
    because when use by external modules, coeff[5:]=np.zeros(3)
    in the case of a linear threshold
    """
    muV0, DmuV0 = -60e-3,10e-3
    sV0, DsV0 =4e-3, 6e-3
    TvN0, DTvN0 = 0.5, 1.
    return P0+P1*(muV-muV0)/DmuV0+\
        P2*(sV-sV0)/DsV0+P3*(TvN-TvN0)/DTvN0+\
        P4*np.log(muGn)+P5*((muV-muV0)/DmuV0)**2+\
        P6*((sV-sV0)/DsV0)**2+P7*((TvN-TvN0)/DTvN0)**2+\
        P8*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+\
        P9*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+\
        P10*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0
      
# final transfer function template :
def TF_my_template(fe, fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    muV, sV, muGn, TvN = get_fluct_regime_vars(fe, fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
    Vthre = threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
    Fout_th = erfc_func(muV, sV, TvN, Vthre, Gl, Cm)
    return Fout_th

def make_loop(t, nu, vm, nu_aff_exc, nu_aff_inh, BIN,\
              Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    dt = t[1]-t[0]
    # constructing the Euler method for the activity rate
    for i_t in range(len(t)-1): # loop over time
        
        fe = (nu_aff_exc[i_t]+nu[i_t]+Fdrive) # afferent+recurrent excitation
        fi = nu[i_t]+nu_aff_inh[i_t] # recurrent inhibition
        W[i_t+1] = W[i_t] + dt/Tw*(b*nu[i_t]*Tw - W[i_t])

        nu[i_t+1] = nu[i_t] +\
               dt/BIN*(\
                TF_my_template(fe, fi, W[i_t], Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)\
                -nu[i_t])

        vm[i_t], _, _, _ = get_fluct_regime_vars(fe, fi, W[i_t], Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)

    return nu, vm, W


################################################################
##### Now fitting to Transfer Function data
################################################################


def fitting_Vthre_then_Fout(Fout, Fe_eff, fiSim, params,cell_type,\
                            maxiter=10000, xtol=1e-10,
                            verbose=False,
                            with_square_terms=False):

    Fout, Fe_eff, fiSim = Fout.flatten(), Fe_eff.flatten(), fiSim.flatten()
    
    print("Fout: {},Fe_eff: {}, fiSim: {}".format(Fout.shape, Fe_eff.shape, fiSim.shape))
    muV, sV, muGn, TvN = get_fluct_regime_vars(Fe_eff, fiSim, *pseq_params(params, cell_type))

    #try to remove nans and infs
    Veff = effective_Vthre(Fout, muV, sV, TvN, params['Gl']*1e-9, params['Cm']*1e-12)
    nanindex=np.where(np.isnan(Veff))
    infindex=np.where(np.isinf(Veff))

    bigindex = np.concatenate([nanindex,infindex],axis=1)

    ve2=np.delete(Fe_eff,bigindex)
    vi2=np.delete(fiSim,bigindex)
    FF2=np.delete(Fout,bigindex)
    # print(ve2.shape)

    #Keep the good ones
    muV_fit, sV_fit, muGn_fit, TvN_fit = get_fluct_regime_vars(ve2, vi2, *pseq_params(params, cell_type))
    Vthre_eff = effective_Vthre(FF2, muV_fit, sV_fit, TvN_fit, params['Gl']*1e-9, params['Cm']*1e-12)
    
    #check if there are still nans/infs
    print(np.isnan(Vthre_eff).any())
    print(np.isinf(Vthre_eff).any())

    
    # i_non_zeros = np.where(Fout>0)

    # Vthre_eff = effective_Vthre(Fout[i_non_zeros], muV[i_non_zeros],\
    #             sV[i_non_zeros], TvN[i_non_zeros], params['Gl'], params['Cm'])
    
    if with_square_terms:
        P = np.zeros(11)
    else:
        P = np.zeros(5)
    P[:5] = Vthre_eff.mean(), 1e-3, 1e-3, 1e-3, 1e-3

    def Res(p):
        if not with_square_terms:
            pp = np.concatenate([p, np.zeros(6)])
        else:
            pp=p
        # vthre = threshold_func(muV[i_non_zeros], sV[i_non_zeros],\
        #                        TvN[i_non_zeros], muGn[i_non_zeros], *pp)
        vthre = threshold_func(muV_fit, sV_fit, TvN_fit, muGn_fit, *pp)
        return np.mean((Vthre_eff-vthre)**2)
    
    plsq = minimize(Res, P, method='SLSQP',\
                    options={'ftol': 1e-8, 'disp': True, 'maxiter':40000})

    if verbose:
        print(plsq)

    P = plsq.x
    
    def Res(p):
        if not with_square_terms:
            params['P'] = np.concatenate([p, np.zeros(6)])
        else:
            params['P'] = p
        return np.mean((Fout-\
                        TF_my_template(Fe_eff, fiSim, *pseq_params(params, cell_type)))**2)

    plsq = minimize(Res, P, method='nelder-mead',\
            options={'xtol': xtol, 'disp': True, 'maxiter':maxiter})

    if verbose:
        print(plsq)
    
    if with_square_terms:
        return plsq.x
    else:
        return np.concatenate([plsq.x, np.zeros(6)])

def make_fit_from_data(DATA, cell_type, params_file,  with_square_terms=False,
                       verbose=False):
    feSim, fiSim, params = np.load(params_file,allow_pickle=True) 
    Fe_eff = np.zeros((fiSim.size,feSim.size))
    for i in range(fiSim.size):
            Fe_eff[i][:] = feSim 
    MEANfreq = np.load(DATA)
    # MEANfreq, SDfreq, Fe_eff, fiSim, params = np.load(DATA,allow_pickle=True) 

    Fe_eff, Fout = np.array(Fe_eff), np.array(MEANfreq)
    # levels = fiSim # to store for colors
    fiSim = np.meshgrid(np.zeros(Fe_eff.shape[1]), fiSim)[1]

    P = fitting_Vthre_then_Fout(Fout, Fe_eff, fiSim, params,cell_type,\
                                with_square_terms=with_square_terms,
                                verbose=verbose)
                            
    print('==================================================')
    print(1e3*np.array(P), 'mV')

    # then we save it:
    filename = DATA.replace('.npy', '_fit.npy')
    print('coefficients saved in ', filename)
    np.save(filename, np.array(P))

    return P

def make_fit_from_data_2(DATA,cell_type, params_file, with_square_terms=False ):

    FF=np.load(DATA).T
    ve, vi, params = np.load(params_file,allow_pickle=True) 
    vve, vvi = np.meshgrid(ve, vi)

    muV, sV, Tv, TvN = MPF(vve, vvi, FF, params, cell_type)

    # Veff = pheV(FF, muV, sV, Tv)
    muV_fit, sV_fit, Tv_fit, TvN_fit, Veff_fit = get_rid_of_nans(vve, vvi, FF, params, cell_type)

    # fitting first order Vthr on the phenomenological threshold space
    print("fitting first order V threshold..")
    def Res(P):
        # fitting first order Vthr on the phenomenological threshold space 
        return np.mean((Veff_fit - Vthre(np.concatenate([P,[0]*6]), muV_fit, sV_fit, TvN_fit))**2 )

    res = minimize(Res, [Veff_fit.mean(),1e-3,1e-3,1e-3], method='nelder-mead', tol=1e-15, options={'disp':True,'maxiter':20000})
    P1 = np.array(res.x)
    print("P1 = ", P1)

    if with_square_terms:
        print("fitting second order V threshold..")
        def Res_2(P): 
            # fit the second order parameters on Vthre ( not necessary most of the time!!!! -> SKIP )
            return np.mean( (Veff_fit - Vthre(np.concatenate([P1,P]), muV_fit, sV_fit, TvN_fit))**2 )
        res = minimize(Res_2, [1e-9]*6, method='nelder-mead', tol=1e-20, options={'disp':True,'maxiter':20000})
        # res = minimize(Res, [0]*6, method='SLSQP', options={'ftol':1e-20,'disp':True,'maxiter':20000})
        P2 = np.array(res.x)
        P = np.concatenate([P1, P2 ])
    else:
        P = np.concatenate([P1, [0]*6])

    print("P = ", P)
    print("Fitting Transfer Function..")
    def Res_TF(P):
    #     return np.mean( (TC_fit - TF(P, muV_fit, sV_fit, Tv_fit, TvN_fit))**2 )
        return np.mean( (FF - TF(P, muV, sV, Tv, TvN))**2 )
    res = minimize(Res_TF, P, method='nelder-mead', tol=1e-15, options={'disp':True,'maxiter':40000})

    print(P)

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


