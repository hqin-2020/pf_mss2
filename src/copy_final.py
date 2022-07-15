import shutil
import os
from tqdm import tqdm

workdir = os.path.dirname(os.getcwd())
source_dir = '/project2/lhansen/pf_mss/'
destination_dir = workdir + '/output/'

N = 100_000
T = 283

for i in tqdm(range(1,151)):
    print(i)
    case = 'actual data, seed = ' + str(i) + ', T = ' + str(T) + ', N = ' + str(N)
    casedir = destination_dir + case  + '/'
    try:
        os.mkdir(casedir)
        shutil.copy(source_dir + case  + '/θ_282.pkl', casedir)
    except:
        pass
