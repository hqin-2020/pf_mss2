import shutil
import os
from tqdm import tqdm

batch_num = 120

workdir = os.path.dirname(os.getcwd())
source_dir = '/project2/lhansen/particle_filtering/pf_mss2/'
destination_dir = workdir + '/output/'

N = 10_000
T = 283

for i in tqdm(range(1,batch_num + 1)):
    print(i)
    case = 'simulated data, seed = ' + str(i) + ', T = ' + str(T) + ', N = ' + str(N)
    casedir = destination_dir + case  + '/'
    try:
        os.mkdir(casedir)
        shutil.copy(source_dir + case  + '/θ_282.pkl', casedir)
    except:
        pass
