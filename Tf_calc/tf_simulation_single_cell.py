from brian2 import *
import time
import argparse 
from Tf_calc.cell_library import get_neuron_params_double_cell
from functions import get_np_linspace, bin_array
import os
from IPython.display import clear_output

start_scope()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--model', type=str, default='adex', help='model to run')
parser.add_argument('--cells', type=str, default='RS', help='cell types of the populations')
parser.add_argument('--range_inh', type=get_np_linspace , default='0.1,30,60', help='inhibitory input values')
parser.add_argument('--range_exc', type=get_np_linspace , default='0.1,30,60', help='excitatory input values')

parser.add_argument('--time', type=float, default=10000, help='Total Time of simulation (ms)')
# parser.add_argument('--save_path', type=str, default='./', help='path to save results')
parser.add_argument('--save_name', type=str, default='trial', help='name to save')
args = parser.parse_args()

CELLS = args.cells
# save_path = args.save_path
save_name = args.save_name
# eqs = get_model(MODEL)
params = get_neuron_params_double_cell(CELLS)
# Extract values from params for each key
extracted_values = {}
for key in params.keys():
    extracted_values[key] = params[key]
# Unpack extracted values into variables
locals().update(extracted_values)

range_inh = args.range_inh
range_exc = args.range_exc
TotTime = args.time
start_time = time.time()

DT=0.1 # time step
defaultclock.dt = DT*ms

duration = TotTime*ms

i=0
eqs="""
dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)-GsynE*(vm-Ee)-GsynI*(vm-Ei)+Is-w)/Cm :volt (unless refractory)
dw/dt=(a*(vm-EL)-w)/tauw : amp
dGsynI/dt = -GsynI/TsynI : siemens
dGsynE/dt = -GsynE/TsynE : siemens
TsynI:second
TsynE:second
Is:amp
Cm:farad
gL:siemens
Vr:volt
b:amp
DeltaT:volt
Vcut:volt
VT:volt
EL:volt
a:siemens
tauw:second
Dt:volt
Ee:volt
Ei:volt
"""
Fe_eff = np.zeros((len(range_inh),len(range_exc)))
MAXfout = 30
dFexc = max(range_exc)/len(range_exc)

JUMP = np.linspace(0,MAXfout,len(range_exc)) # constrains the fout jumps

FRout_inh=[]
muvi=[]

Adapt=[]
Npts_e = len(range_exc) 
Npts_i = len(range_inh)

print(range_exc, range_inh)
for rate_exc in range_exc:
	print("rate exc =", rate_exc)
	FRout_inh.append([])
	Adapt.append([])
	muvi.append([])
	for rate_inh in range_inh:
		print("rate inh =", rate_inh)

		# Population 1 - Fast Spiking

		G_inh = NeuronGroup(1, eqs, threshold='vm > Vcut',refractory=5*ms, reset="vm = Vr; w += b", method='heun')
		G_inh.vm = V_m * mV 
		G_inh.w = a*nS * (G_inh.vm - G_inh.EL)
		G_inh.Vr = V_r * mV  
		G_inh.TsynI = tau_i * ms  
		G_inh.TsynE = tau_e * ms  
		G_inh.b = b * pA
		G_inh.DeltaT = delta * mV
		G_inh.VT = V_th * mV
		G_inh.Vcut = V_cut * mV
		G_inh.EL = EL * mV
		G_inh.GsynI=0.0*nS
		G_inh.GsynE=0.0*nS
		G_inh.Ee= E_e*mV
		G_inh.Ei= E_i*mV
		G_inh.Cm = Cm*pF
		G_inh.gL = Gl*nS
		G_inh.tauw = tau_w*ms 
		G_inh.Is = 0.0*nA 

			
		P_inh = int(p_con*gei*Ntot)
		P_exc = int(p_con*(1- gei)*Ntot)

		# external drive--------------------------------------------------------------------------

		P_ed_inh = PoissonGroup(P_inh, rates=rate_inh*Hz) #5% of 2000 inh cells
		P_ed_exc = PoissonGroup(P_exc, rates=rate_exc*Hz) #5% of 8000 exc cells


		# Network-----------------------------------------------------------------------------

		# connections-----------------------------------------------------------------------------
		#seed(0)
		Qi=Q_i*nS
		Qe=Q_e*nS

		prbC=1. #1


		S_edin_in = Synapses(P_ed_inh, G_inh, on_pre='GsynI_post+=Qi')
		S_edin_in.connect(p=prbC)

		S_edex_in = Synapses(P_ed_exc, G_inh, on_pre='GsynE_post+=Qe')
		S_edex_in.connect(p=prbC)



		# Recording tools -------------------------------------------------------------------------------
		M2G2_vi = StateMonitor(G_inh, 'vm', record=0)
		M3G2 = StateMonitor(G_inh, 'w', record=0)
		FRG_inh = PopulationRateMonitor(G_inh)

		# Run simulation -------------------------------------------------------------------------------

		#print('--##Start simulation##--')
		run(duration)
		#print('--##End simulation##--')


		#  -------------------------------------------------------------------------------


		# prepare firing rate

		BIN=5
		time_array = arange(int(TotTime/DT))*DT

		# print(test)
		
		LfrG_inh=array(FRG_inh.rate/Hz)
		TimBinned,popRateG_inh=bin_array(time_array, BIN, time_array),bin_array(LfrG_inh, BIN, time_array)



		uu=M3G2[0].w
		vvi=M2G2_vi[0].vm
		FRout_inh[i].append(mean(popRateG_inh[int(0.8*len(popRateG_inh))::]))
		Adapt[i].append(mean(uu[int(0.8*len(uu)):len(uu)]))
		muvi[i].append(mean(vvi[int(0.8*len(vvi)):len(vvi)]))

	i=i+1
	
	if i % 1 == 0:
		print(i)


try:
	os.listdir('./Tf_calc/data/')
except:
	os.makedirs('./Tf_calc/data/')
#
np.save(f'./Tf_calc/data/ExpTF_{Npts_e}x{Npts_i}_{save_name}_{CELLS}.npy', FRout_inh)
np.save(f'./Tf_calc/data/{Npts_e}x{Npts_i}_{save_name}_{CELLS}_adapt.npy', Adapt)

np.save(f'./Tf_calc/data/{Npts_e}x{Npts_i}_{save_name}_{CELLS}_muv.npy', muvi)

# np.save(f'{save_path}data/params_range_{Npts_e}x{Npts_i}_{save_name}.npy', np.array([range_exc, range_inh, params], dtype='object'), allow_pickle=True)
np.save(f'./Tf_calc/data/{Npts_e}x{Npts_i}_{save_name}_{CELLS}_params.npy', np.array([Fe_eff, range_inh, params], dtype='object'), allow_pickle=True)

end_time = time.time()
execution_time = end_time - start_time

if execution_time > 3600:
	print("Execution time:", execution_time/3600, "hours")
else:
	print("Execution time:", execution_time/60, "minutes")







