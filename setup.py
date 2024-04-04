# %precision 2
# %load_ext autoreload
# %autoreload 2

from IPython.display import clear_output

import matplotlib.pyplot as plt
from brian2 import *
import ptitprince as pt
import seaborn as sns
import matplotlib.colors as mplcol

from functions import *
from codes_analyses import *
import tvb_model_reference.src.nuu_tools_simulation_human as tools

from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin_new import Parameter
parameters = Parameter()


# clear output
clear_output()

print('Everything is now installed. You can proceed. ')