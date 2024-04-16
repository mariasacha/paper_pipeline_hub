
from brian2 import Equations

def get_model(MODEL):
    
    if MODEL == "adex":
        eqs = Equations("""
        dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)-GsynE*(vm-Ee)-GsynI*(vm-Ei)+I-w)/C : volt (unless refractory)
        dw/dt=(a*(vm-EL)-w)/tauw : amp
        dGsynI/dt = -GsynI/TsynI : siemens
        dGsynE/dt = -GsynE/TsynE : siemens
        TsynI:second
        TsynE:second
        Vr:volt
        b:amp
        DeltaT:volt
        Vcut:volt
        VT:volt
        EL:volt
        """)
    
    else:
        print('Model not recognised')

    return eqs