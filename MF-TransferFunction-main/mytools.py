import numpy as np
from scipy.stats import norm


class AttrDict(dict): # selfmade parameter dictionary class
    def __init__(self, *args, **kwargs):
        super(AttrDict,self).__init__(*args,**kwargs)
        self.__dict__ = self



def progressBar(iterable, prefix='', suffix='', decimals=1, length=50, fill='=', arrow='>', printEnd='\r'):

        total = len(iterable)
        def printProgressBar (iteration): # Progress Bar Printing Function
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * (filledLength-1) + arrow + '-' * (length - filledLength)
            print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
        printProgressBar(0) # Initial Call
        # Update Progress Bar
        for i, item in enumerate(iterable):
            yield item
            printProgressBar(i + 1)
        print() # Print New Line on Complete



def double_gaussian(t, t0, T1, T2, amplitude):
    
    return amplitude*(\
                      np.exp(-(t-t0)**2/2./T1**2)*(t<t0)+\
                      np.exp(-(t-t0)**2/2./T2**2)*(t>t0))
        

def ornstein_uhlenbeck(steps,T,mu,theta,sigma,paths=1,start=0,seed=None,nonzero=True):
    
    if seed!=None:
        np.random.seed(seed=seed)
 
    _, dt = np.linspace(0, T, steps, retstep=True ) 

    X0 = start
    X = np.zeros((paths,steps))
    X[:,0] = X0
    W = norm.rvs( loc=0, scale=1, size=(paths,steps-1) )


    std_dt = np.sqrt( sigma**2 /(2*theta) * (1-np.exp(-2*theta*dt)) )
    for t in range(0,steps-1):
        X[:,t+1] = mu + np.exp(-theta*dt)*(X[:,t]-mu) + std_dt * W[:,t]
        
    # X_T = X[:,-1]    # values of X at time T
    # X_1 = X[1,:]     # a single path
    if paths==1: X=X[0,:]
    # X[X==0]=1e-12
    if nonzero: X[X<0]=-X[X<0]
    return np.array(X)