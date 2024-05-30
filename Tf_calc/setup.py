# %precision 2
# %load_ext autoreload
# %autoreload 2

from IPython.display import clear_output
import sys
import os
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
from brian2 import *
import ptitprince as pt
import seaborn as sns
import matplotlib.colors as mplcol
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from functions import *
import TVB.tvb_model_reference.src.nuu_tools_simulation_human as tools
from Tf_calc.theoretical_tools import *

from TVB.tvb_model_reference.simulation_file.parameter.parameter_M_Berlin_new import Parameter
parameters = Parameter()


# clear output
clear_output()

print('Everything is now installed. You can proceed. ')