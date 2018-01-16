import numpy as np
import os,sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from graph import *
from custom_svc import *
import copy
import NN,time
from custom_gridsearch import GridSearch
import sys

number=sys.argv[1]
alpha=[float(a) for a in sys.argv[2].split(',')]
eps=[float(a) for a in sys.argv[3].split(',')]

path='../data/MUTAG_2/'
dataset=list(build_MUTAG_dataset(path))
X,y=zip(*dataset)

n_splits=10
n_jobs=1

tuned_parameters = [{'epsilon':list(eps),
                    'alpha':list(alpha)
                     ,'method':['shortest_path']
                     ,'normalize_distance':[True]
                     ,'features_metric':['dirac']
                     ,'scale':[False,True]
                     ,'C':[10**k for k in [2.22,2.23,2.24,2.25,2.26,2.28,2.29,2.3,2.32,2.33,2.34,2.35,2.36,2.4]]
                     ,'gamma':[2**k for k in [-0.1,-0.2,-0.3,-0.4,0,0.01,0.02,0.03,0.04,0.05,0.06,0.1,0.2,0.3,0.4,0.5,0.6,1,2]]}]

#tuned_parameters = [{'epsilon':list(np.logspace(-2,2,2)),
#                    'alpha':list(np.linspace(0,1,1))
#                     ,'method':['shortest_path']
#                     ,'normalize_distance':[False]
#                     ,'features_metric':['dirac']
#                     ,'scale':[False]
#                     ,'C':[10**k for k in [-1]]
#                     ,'gamma':[2**k for k in [1]]}]


graph_svc=Graph_WGW_SVC_Classifier()
clf = GridSearch(graph_svc, tuned_parameters=tuned_parameters,nb_splits=n_splits,n_jobs=n_jobs,parallel=False,csv='result_'+str(number)+'.csv')
clf.fit(np.array(X).reshape(-1,1),np.array(y))


