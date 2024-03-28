import numpy as np
from mytools import AttrDict


P = {}

def loadparams(scenario):
    return P[scenario]

# load thalamus fitting params
# PTC=np.load('data\\NEW6params_TC.npy')
# PRE=np.load('data\\NEW6params_RE.npy')
PTC=[1,2,3]
PRE = [1,2,3]
# load cortex fitting params
PRS=np.load("./data/RS-cell0_CONFIG1_fit.npy")[[0,1,2,3,5,8,9,6,10,7]]
# PRS=np.load("data\\RS-cell0_CONFIG1_fit.npy")[[0,1,2,3,5,8,9,6,10,7]]
PFS=np.load("./data/FS-cell_CONFIG1_fit.npy")[[0,1,2,3,5,8,9,6,10,7]]


# THALAMUS-CORTEX =================================

params = {}
# spont osc occur for just changing TC_Nexc: 800->400 and p_TC-RS=0.1 !!!
#Nexc/inh : number of excitatory - inhibitory > they are okay

params['RS'] = AttrDict({
    'P' : PRS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -63e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 8e-9,
    'b' : 10e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
params['FS'] = AttrDict({
    'P' : PFS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -67e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0,
    'b' : 10e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})

params['TC'] = AttrDict({
    'P' : PTC,
    'Nexc' : 400,
    'Ninh' : 25,
    'Qe' : 1e-9,
    'Qi' : 6e-9,
    'Cm' : 160e-12,
    'El' : -65e-3,
    'Gl' : 10e-9,
    'Tw' : 200e-3,
    'a' : 0,
    'b' : 10e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
params['RE'] = AttrDict({
    'P' : PRE,
    'Nexc' : 400,
    'Ninh' : 150,
    'Qe' : 4e-9,
    'Qi' : 1e-9,
    'Cm' : 200e-12,
    'El' : -75e-3,
    'Gl' : 10e-9,
    'Tw' : 200e-3,
    'a' : 8e-9,
    'b' : 10e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
P['thaco_spont-osc'] = params


# THALAMUS =============================================


# domenico with ACh ------------------------------------

params = {}

params['TC'] = AttrDict({
    'P' : PTC,
    'Nexc' : 800, #400
    'Ninh' : 25,
    'Qe' : 1e-9,
    'Qi' : 6e-9,
    'Cm' : 160e-12,
    'El' : -65e-3, #-55
    'Gl' : 10e-9,
    'Tw' : 200e-3,
    'a' : 0,
    'b' : 10e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
params['RE'] = AttrDict({
    'P' : PRE,
    'Nexc' : 400,
    'Ninh' : 150,
    'Qe' : 4e-9,
    'Qi' : 1e-9,
    'Cm' : 200e-12,
    'El' : -75e-3,
    'Gl' : 10e-9,
    'Tw' : 200e-3,
    'a' : 8e-9,
    'b' : 10e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
P['thalamus_ACh'] = params


# domenico control ------------------------------------------

params = {}

params['TC'] = AttrDict({
    'P' : PTC,
    'Nexc' : 800, # 400
    'Ninh' : 25,
    'Qe' : 1e-9,
    'Qi' : 6e-9,
    'Cm' : 160e-12,
    'El' : -73e-3, #-63
    'Gl' : 9.5e-9,
    'Tw' : 270e-3,
    'a' : 14e-9,
    'b' : 20e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
params['RE'] = AttrDict({
    'P' : PRE,
    'Nexc' : 400,
    'Ninh' : 150,
    'Qe' : 4e-9,
    'Qi' : 1e-9,
    'Cm' : 200e-12,
    'El' : -85e-3,
    'Gl' : 13e-9,
    'Tw' : 230e-3,
    'a' : 28e-9,
    'b' : 10e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
P['thalamus_control'] = params


# Wolfart inputs ------------------------------------

params = {}

params['TC'] = AttrDict({
    'P' : PTC,
    'Nexc' : 800,
    'Ninh' : 25,
    'Qe' : 1e-9,
    'Qi' : 6e-9,
    'Cm' : 160e-12,
    'El' : -65e-3,
    'Gl' : 10e-9,
    'Tw' : 200e-3,
    'a' : 0,
    'b' : 10e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
params['RE'] = AttrDict({
    'P' : PRE,
    'Nexc' : 400,
    'Ninh' : 150,
    'Qe' : 4e-9,
    'Qi' : 1e-9,
    'Cm' : 200e-12,
    'El' : -75e-3,
    'Gl' : 10e-9,
    'Tw' : 200e-3,
    'a' : 8e-9,
    'b' : 10e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
P['thalamus_Wolfart'] = params


# no adaptation ------------------------------------

params = {}

params['TC'] = AttrDict({
    'P' : PTC,
    'Nexc' : 800,
    'Ninh' : 25,
    'Qe' : 1e-9,
    'Qi' : 6e-9,
    'Cm' : 160e-12,
    'El' : -65e-3,
    'Gl' : 10e-9,
    'Tw' : 200e-3,
    'a' : 0,
    'b' : 0e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
params['RE'] = AttrDict({
    'P' : PRE,
    'Nexc' : 400,
    'Ninh' : 150,
    'Qe' : 4e-9,
    'Qi' : 1e-9,
    'Cm' : 200e-12,
    'El' : -75e-3,
    'Gl' : 10e-9,
    'Tw' : 200e-3,
    'a' : 0e-9,
    'b' : 0e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
P['thalamus_noadapt'] = params


# CORTEX =============================================

# diVolo -----------------------------------------

params = {}

params['RS'] = AttrDict({
    'P' : PRS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1.5e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -65e-3,
    'Gl' : 10e-9,
    'Tw' : 200e-3,
    'a' : 4e-9,
    'b' : 40e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
params['FS'] = AttrDict({
    'P' : PFS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1.5e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -65e-3,
    'Gl' : 10e-9,
    'Tw' : 200e-3,
    'a' : 0,
    'b' : 0,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
P['cortex'] = params

# UD states ---------------------------------------

params = {}

params['RS'] = AttrDict({
    'P' : PRS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1.5e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -64e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0e-9,
    'b' : 30e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
params['FS'] = AttrDict({
    'P' : PFS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1.5e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -65e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0,
    'b' : 0e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
P['cortex_updown'] = params

# UD states MARIA---------------------------------------

params = {}

params['RS'] = AttrDict({
    'P' : PRS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1.5e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -64e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0e-9,
    'b' : 30e-12,
    'Ti' : 5e-3,
    'Te' : 4e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
params['FS'] = AttrDict({
    'P' : PFS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1.5e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -65e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0,
    'b' : 0e-12,
    'Ti' : 5e-3,
    'Te' : 4e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
P['cortex_updown_maria'] = params

# UD states NO ADAPTATION MARIA---------------------------------------

params = {}

params['RS'] = AttrDict({
    'P' : PRS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1.5e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -64e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0e-9,
    'b' : 0e-12,
    'Ti' : 5e-3,
    'Te' : 4e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
params['FS'] = AttrDict({
    'P' : PFS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1.5e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -65e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0,
    'b' : 0e-12,
    'Ti' : 5e-3,
    'Te' : 4e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
P['cortex_updown_maria_no_adapt'] = params

# UD states NO ADAPTATION MARIA---------------------------------------

params = {}

params['RS'] = AttrDict({
    'P' : PRS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1.5e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -64e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0e-9,
    'b' : 0e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
params['FS'] = AttrDict({
    'P' : PFS,
    'Nexc' : 400,
    'Ninh' : 100,
    'Qe' : 1.5e-9,
    'Qi' : 5e-9,
    'Cm' : 200e-12,
    'El' : -65e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0,
    'b' : 0e-12,
    'Ti' : 5e-3,
    'Te' : 5e-3,
    'Ee' : 0,
    'Ei' : -80e-3
})
P['cortex_updown_maria_def'] = params