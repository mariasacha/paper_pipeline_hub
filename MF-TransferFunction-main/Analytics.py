import numpy as np

class MPF(object):
    '''
    calculates the membrane-potential-fluctuation 's statistical moments for TC and RE thalamic cells
        * call class MPF with argument "TC" or "RE" for the cell type
        * then call MPF.run with arguments for the excitatory and inhibatory input frequencies 
    '''

    def __init__(self, cell):
        self.gL = 10 #*1e-9 # nS
        self.tau_w = 200 #*1e-3 # ms
        self.b = 0.01 #*1e-9 # nA
        self.E = np.array([0, -80]) #*1e-3 # mV
        self.tau = np.array([5, 5]) #*1e-3 # ms
        self.K = np.array([2, 0.5]) #*1e-3 # mV
        if cell=='TC':
            self.EL = -55 #*1e-3 # mV
            self.Q = np.array([1, 6]) #*1e-9 # nS
            self.Cm = 160 #*1e-12 # pF
            self.a = 0 # nS
        elif cell=='RE':
            self.EL = -75 #*1e-3 # mV
            self.Q = np.array([4, 1]) #*1e-9 # nS
            self.Cm = 200 #*1e-12 # pF
            self.a = 8 #*1e-12 # nS


    ##################################
    # Dynamics

    def mu_Gei(self, v, ei):
        return v*self.K[ei]*self.tau[ei]*self.Q[ei]
    def sigma_Gei(v, ei):
        return np.sqrt( v*self.K[ei]*self.tau[ei]/2 )*self.Q[ei]

    def mu_G(self, ve, vi):
        return self.mu_Gei(ve,0) + self.mu_Gei(vi,1) + self.gL
    def tau_m(self, ve, vi):
        return self.Cm/self.mu_G(ve, vi)

    def mu_V(self, ve, vi, adapt, vout):
        return (self.E[0]*self.mu_Gei(ve,0) + self.E[1]*self.mu_Gei(vi,1) + self.EL*self.gL - adapt - vout*self.tau_w*self.b + self.a*self.EL)/(self.mu_G(ve, vi) + self.a)

    def U(self, ve, vi, ei, adapt, vout):
        return self.Q[ei]/self.mu_G(ve, vi) * (self.E[ei]-self.mu_V(ve, vi, adapt, vout))

    def sigma_V(self, ve, vi, adapt, vout):
        return np.sqrt((self.K[0]*ve*(self.U(ve, vi, 0, adapt, vout)*self.tau[0])**2/(self.tau_m(ve, vi) + self.tau[0]) + self.K[1]*vi*(self.U(ve, vi, 1, adapt, vout)*self.tau[1])**2/(self.tau_m(ve, vi) + self.tau[1]) )/2)

    def tau_V(self, ve, vi, adapt, vout):
        ve+=1e-9
        vi+=1e-9
        return ( self.K[0]*ve*(self.U(ve, vi, 0, adapt, vout)*self.tau[0])**2
                + self.K[1]*vi*(self.U(ve, vi, 1, adapt, vout)*self.tau[1])**2 )/self.sigma_V(ve, vi, adapt, vout)**2
    
    ##################################
    # Main

    def run(self, ve, vi, adapt=None, vout=None):
        '''
        calculates MPF statistical moments
        args: ve, vi, adaptation w, vout
        returns: membrane potential mean, std, tau (in mV!)
        '''
        if adapt==None: adapt=0
        if vout==None: vout=0
        return self.mu_V(ve, vi, adapt, vout)+1e-9, self.sigma_V(ve, vi, adapt, vout)+1e-9, self.tau_V(ve, vi, adapt, vout)+1e-9
