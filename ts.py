import scipy.stats as stat
from matplotlib import colors as mcolors
import matplotlib.pyplot as pl
import matplotlib
import random
import utils
#import seaborn as sns
import numpy as np
import ot



class TS(object):
    def __init__(self):
        self.name='A ts as no name'
        self.times=[]
        self.values=[]
        self.C=None

    def __eq__(self, other) : 
        #print('yo method')
        return (self.values == other.values) and (self.times == other.times)

    def __hash__(self):
        return hash(str(self))

    def characterized(self):
        if self.name!='A ts as no name':
            return self.name
        else:
            return self

    def plot_ts(self,show=False,**kwargs):
        pl.plot(self.values,**kwargs)
        if show==True:
           pl.show()

    def distance_matrix(self,method='sqeuclidean',force_recompute=False):

        if (self.C is None) or force_recompute:

            if method=='dirac':
                f=lambda x,y: x!=y
                C = ot.dist(np.array(self.times).reshape(-1,1),np.array(self.times).reshape(-1,1),f)
            else :
                C  = ot.dist(np.array(self.times).reshape(-1,1),np.array(self.times).reshape(-1,1),method)
            self.C = C/np.max(C)

            self.name_struct_dist=method

            return self.C

        else :
            return self.C

    def plot_sample_ts_dataset(self,dataset,N=10,**kwargs):
	    colors=color_list = pl.cm.Set3(np.linspace(0, 1, 12))
	    #for name, hex in matplotlib.colors.cnames.items():
	        #colors.append(hex)
	    choosenclasses=[]
	    for i in range(N):
	        j=random.randint(0,len(dataset)-1)
	        choosenclasses.append(dataset[j][1])
	        dataset[j][0].plot_ts(color=colors[dataset[j][1]])
	    choosenclasses=['classe'+str(x) for x in set(choosenclasses)]
	    pl.legend(choosenclasses)
	    pl.show()



def create_ts(path,t='TRAIN'):
    data=[]
    file=utils.read_files(path)
    file=[x for x in file if x.split('_')[1]==t][0]
    with open(path+file) as f:
        k=0
        for it in f:
            ts=TS()
            liste=list(it.split(','))
            ts.values=[float(x) for x in liste[1:]]
            ts.times=range(len(ts.values))
            ts.name=str(k)+t
            data.append((ts,int(liste[0])))
            k=k+1
    return data

def create_random_ts(datasetsize=50,N=100,a=0,b=10):
    data=[]
    k=0
    for i in range(datasetsize):      
        ts=TS()
        ts.values=list(np.random.uniform(low=a, high=b, size=(N,)))
        ts.times=range(len(ts.values))
        ts.name=str(k)
        data.append((ts,0))
        k=k+1
    return data







