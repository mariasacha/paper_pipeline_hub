"""
Some configuration of neuronal properties so that we pick up
within this file
"""
from __future__ import print_function

def get_neuron_params_double_cell(NAME, SI_units=False):
    
    if NAME == 'FS-RS': 
        params = {#Cells
            'V_m': -60, 'V_r': -65, 'Cm': 200, 'Gl': 10, 'tau_w': 500, 'V_th': -50, 'V_cut' : -30,
                #Excitatory
                  'a_e': 0,   'b_e': 30,  'delta_e': 2, 'EL_e': -64,
                # Inhibitory
                  'a_i': 0,'b_i': 0,'delta_i': 0.5,'EL_i': -65,
                # Synaptic
                   'tau_e': 5, 'tau_i': 5, 'E_e': 0, 'E_i': -80, 'Q_i': 5.0, 'Q_e': 1.5,
                # Network
                  'p_con': 0.05, 'gei': 0.2, 'Ntot': 10000}
    elif NAME == 'FS-RS_10': 
        params = {#Cells
            'V_m': -60, 'V_r': -65, 'Cm': 200, 'Gl': 10, 'tau_w': 500, 'V_th': -50, 'V_cut' : -30,
                #Excitatory
                  'a_e': 0,   'b_e': 10,  'delta_e': 2, 'EL_e': -64,
                # Inhibitory
                  'a_i': 0,'b_i': 0,'delta_i': 0.5,'EL_i': -65,
                # Synaptic
                   'tau_e': 5, 'tau_i': 5, 'E_e': 0, 'E_i': -80, 'Q_i': 5.0, 'Q_e': 1.5,
                # Network
                  'p_con': 0.05, 'gei': 0.2, 'Ntot': 10000}
    elif NAME == 'FS-RS_faycal': 
        params = {#Cells
            'V_m': -60, 'V_r': -65, 'Cm': 200, 'Gl': 10, 'tau_w': 500, 'V_th': -50, 'V_cut' : -30,
                #Excitatory
                  'a_e': 0,   'b_e': 0,  'delta_e': 2, 'EL_e': -63,
                # Inhibitory
                  'a_i': 0,'b_i': 0,'delta_i': 0.5,'EL_i': -65,
                # Synaptic
                   'tau_e': 5, 'tau_i': 5, 'E_e': 0, 'E_i': -80, 'Q_i': 5.0, 'Q_e': 1.5,
                # Network
                  'p_con': 0.05, 'gei': 0.2, 'Ntot': 10000}
    elif NAME == 'FS-RS_0': 
        params = {#Cells
            'V_m': -60, 'V_r': -65, 'Cm': 200, 'Gl': 10, 'tau_w': 500, 'V_th': -50, 'V_cut' : -30,
                #Excitatory
                  'a_e': 0,   'b_e': 0,  'delta_e': 2, 'EL_e': -64,
                # Inhibitory
                  'a_i': 0,'b_i': 0,'delta_i': 0.5,'EL_i': -65,
                # Synaptic
                   'tau_e': 5, 'tau_i': 5, 'E_e': 0, 'E_i': -80, 'Q_i': 5.0, 'Q_e': 1.5,
                # Network
                  'p_con': 0.05, 'gei': 0.2, 'Ntot': 10000}
    elif NAME == 'FS': 
        params = {#Cells
            'V_m': -60, 'V_r': -65, 'Cm': 200, 'Gl': 10, 'tau_w': 500, 'V_th': -50, 'V_cut' : -30,
                # Inhibitory
                  'a': 0,'b': 0,'delta': 0.5,'EL': -65,
                # Synaptic
                   'tau_e': 5, 'tau_i': 5, 'E_e': 0, 'E_i': -80, 'Q_i': 5.0, 'Q_e': 1.5,
                # Network
                  'p_con': 0.05, 'gei': 0.2, 'Ntot': 10000}
    elif NAME == 'RS': 
        params = {#Cells
            'V_m': -60, 'V_r': -65, 'Cm': 200, 'Gl': 10, 'tau_w': 500, 'V_th': -50, 'V_cut' : -30,
                # Inhibitory
                  'a': 0,'b': 10,'delta': 2,'EL': -64,
                # Synaptic
                   'tau_e': 5, 'tau_i': 5, 'E_e': 0, 'E_i': -80, 'Q_i': 5.0, 'Q_e': 1.5,
                # Network
                  'p_con': 0.05, 'gei': 0.2, 'Ntot': 10000}
    else:
        print('====================================================')
        print('------------ CELL NOT RECOGNIZED !! ---------------')
        print('====================================================')
    
    if SI_units:
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
    else:
        print('cell parameters --NOT-- in SI units')

    return params.copy()
    
def get_neuron_params(NAME, name='', number=1, SI_units=False):

    if NAME=='FS-cell':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'EL':-65., 'V_th':-50., 'V_r':-65., 'delta':0.5,\
                  'a':0., 'b': 0., 'tau_w':500, }
    elif NAME=='RS-cell':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'EL':-64., 'V_th':-50., 'V_r':-65., 'delta':2.,\
                  'a':0., 'b':20., 'tau_w':500.}
    else:
        print('====================================================')
        print('------------ CELL NOT RECOGNIZED !! ---------------')
        print('====================================================')


    if SI_units:
        print('cell parameters in SI units')
        # mV to V
        params['El'], params['Vthre'], params['Vreset'], params['delta_v'] =\
            1e-3*params['El'], 1e-3*params['Vthre'], 1e-3*params['Vreset'], 1e-3*params['delta_v']
        # ms to s
        params['Trefrac'], params['tauw'] = 1e-3*params['Trefrac'], 1e-3*params['tauw']
        # nS to S
        params['a'], params['Gl'] = 1e-9*params['a'], 1e-9*params['Gl']
        # pF to F and pA to A
        params['Cm'], params['b'] = 1e-12*params['Cm'], 1e-12*params['b']
    else:
        print('cell parameters --NOT-- in SI units')
        
    return params.copy()



if __name__=='__main__':

    print(__doc__)
