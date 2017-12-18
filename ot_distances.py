import ot
import WGW_2 as wgw
import numpy as np

# A factoriser !!!!!

class Wasserstein_distance():
    def __init__(self,features_metric='sqeuclidean'): #remplacer method par distance_method  
        self.features_metric=features_metric
        self.transp=None

    def tree_d(self,graph1,graph2):
        import ot
        import numpy as np

        leaves1=graph1.return_leaves(graph1.tree)
        leaves2=graph2.return_leaves(graph2.tree)
        t1masses = np.ones(len(leaves1))/len(leaves1)
        t2masses = np.ones(len(leaves2))/len(leaves2)
        x1=graph1.leaves_matrix_attr().reshape(-1, 1)
        x2=graph2.leaves_matrix_attr().reshape(-1, 1)

        if self.features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=self.features_metric) 
        M= M/np.max(M)

        transp = ot.emd(t1masses,t2masses, M)
        self.transp=transp

        return np.sum(transp*M)

    def graph_d(self,graph1,graph2):

        import ot
        import numpy as np

        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        t1masses = np.ones(len(nodes1))/len(nodes1)
        t2masses = np.ones(len(nodes2))/len(nodes2)
        x1=graph1.all_matrix_attr().reshape(-1, 1)
        x2=graph2.all_matrix_attr().reshape(-1, 1)

        if self.features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=self.features_metric) 
        M= M/np.max(M)

        transp = ot.emd(t1masses,t2masses, M)
        self.transp=transp

        return np.sum(transp*M)

    def ts_d(self,ts1,ts2):

        import ot
        import numpy as np
        N1=len(ts1.values)
        N2=len(ts2.values)
        t1masses = np.ones(N1)/N1
        t2masses = np.ones(N2)/N2

        x1=np.array(ts1.values).reshape(-1,1)
        x2=np.array(ts2.values).reshape(-1,1)

        if self.features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=self.features_metric) 
        M= M/np.max(M)

        transp = ot.emd(t1masses,t2masses, M)
        self.transp=transp

        return np.sum(transp*M)

class Gromov_distance():

    def __init__(self,epsilon=1,method='shortest_path',timedistance='sqeuclidean',max_iter=500): #remplacer method par distance_method  
        self.epsilon=epsilon
        self.method=method
        self.timedistance=timedistance # attention à ça c'est bof
        self.max_iter=max_iter
        self.transp=None
        self.log=None
  
    def tree_d(self,graph1,graph2):

        import WGW_2 as wgw
        import numpy as np


        leaves1=graph1.return_leaves(graph1.tree)
        leaves2=graph2.return_leaves(graph2.tree)
        C1=graph1.distance_matrix(nodeOfInterest=leaves1,method=self.method)
        C2=graph2.distance_matrix(nodeOfInterest=leaves2,method=self.method)
        t1masses = np.ones(len(leaves1))/len(leaves1)
        t2masses = np.ones(len(leaves2))/len(leaves2)

        transpwgw,log= wgw.wgw(np.zeros((C1.shape[0],C2.shape[0])),C1,C2,t1masses,t2masses,'square_loss',self.epsilon,alpha=1,max_iter=self.max_iter,verbose=False,log=True)

        self.transp=transpwgw
        self.log=log

        return log['GW_dist'][::-1][0]

    def graph_d(self,graph1,graph2):
        import WGW_2 as wgw
        import numpy as np

        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        C1=graph1.distance_matrix(method=self.method)
        C2=graph2.distance_matrix(method=self.method)
        t1masses = np.ones(len(nodes1))/len(nodes1)
        t2masses = np.ones(len(nodes2))/len(nodes2)

        transpwgw,log= wgw.wgw(np.zeros((C1.shape[0],C2.shape[0])),C1,C2,t1masses,t2masses,'square_loss',self.epsilon,alpha=1,max_iter=self.max_iter,verbose=False,log=True)

        self.transp=transpwgw
        self.log=log

        return log['GW_dist'][::-1][0]

    def ts_d(self,ts1,ts2):
        import WGW_2 as wgw
        import numpy as np

        C1=ts1.distance_matrix(method=self.method,timedistance=self.timedistance)
        C2=ts2.distance_matrix(method=self.method,timedistance=self.timedistance)

        t1masses = np.ones(len(ts1.values))/len(ts1.values)
        t2masses = np.ones(len(ts2.values))/len(ts2.values)

        transpwgw,log= wgw.wgw(np.zeros((C1.shape[0],C2.shape[0])),C1,C2,t1masses,t2masses,'square_loss',self.epsilon,alpha=1,max_iter=self.max_iter,verbose=False,log=True)

        self.transp=transpwgw
        self.log=log

        return log['GW_dist'][::-1][0]
   

class Gromov_Wasserstein_distance():

    def __init__(self,epsilon=1,alpha=1,ratio=None,method='shortest_path',features_metric='sqeuclidean',timedistance='sqeuclidean',max_iter=500): #remplacer method par distance_method  
        self.epsilon=epsilon
        self.method=method
        self.timedistance=timedistance # attention à ça c'est bof
        self.max_iter=max_iter
        self.alpha=alpha
        self.ratio=ratio
        self.features_metric=features_metric
        self.transp=None
        self.log=None
        
    def tree_d(self,graph1,graph2):

        leaves1=graph1.return_leaves(graph1.tree)
        leaves2=graph2.return_leaves(graph2.tree)
        C1=graph1.distance_matrix(nodeOfInterest=leaves1,method=self.method)
        C2=graph2.distance_matrix(nodeOfInterest=leaves2,method=self.method)
        t1masses = np.ones(len(leaves1))/len(leaves1)
        t2masses = np.ones(len(leaves2))/len(leaves2)
        x1=graph1.leaves_matrix_attr().reshape(-1, 1) # A regarder si c'est pas le contraire !!!!!!
        x2=graph2.leaves_matrix_attr().reshape(-1, 1)

        if self.features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=self.features_metric) 
        M= M/np.max(M)

        if self.ratio is None :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',self.epsilon,self.alpha,max_iter=self.max_iter,verbose=False,log=True)
        else :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',self.epsilon,self.epsilon*self.ratio,max_iter=self.max_iter,verbose=False,log=True)

        self.transp=transpwgw
        self.log=log

        return log['GW_dist'][::-1][0]

    def graph_d(self,graph1,graph2):

        import ot
        import WGW_2 as wgw
        import numpy as np

        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        C1=graph1.distance_matrix(method=self.method)
        C2=graph2.distance_matrix(method=self.method)
        t1masses = np.ones(len(nodes1))/len(nodes1)
        t2masses = np.ones(len(nodes2))/len(nodes2)
        x1=graph1.all_matrix_attr().reshape(-1, 1)
        x2=graph2.all_matrix_attr().reshape(-1, 1)

        if self.features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=self.features_metric) 
        M= M/np.max(M)

        if self.ratio is None :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',self.epsilon,self.alpha,max_iter=self.max_iter,verbose=False,log=True)
        else :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',self.epsilon,self.epsilon*self.ratio,max_iter=self.max_iter,verbose=False,log=True)

        self.transp=transpwgw
        self.log=log

        return log['GW_dist'][::-1][0]
    
    def ts_d(self,ts1,ts2):

        import WGW_2 as wgw
        import numpy as np

        C1=ts1.distance_matrix(method=self.method,timedistance=self.timedistance)
        C2=ts2.distance_matrix(method=self.method,timedistance=self.timedistance)

        t1masses = np.ones(len(ts1.values))/len(ts1.values)
        t2masses = np.ones(len(ts2.values))/len(ts2.values)

        x1=np.array(ts1.values).reshape(-1,1)
        x2=np.array(ts2.values).reshape(-1,1)

        if self.features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=self.features_metric) 
        M= M/np.max(M)

        if self.ratio is None :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',self.epsilon,self.alpha,max_iter=self.max_iter,verbose=False,log=True)
        else :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',self.epsilon,self.epsilon*self.ratio,max_iter=self.max_iter,verbose=False,log=True)

        self.transp=transpwgw
        self.log=log

        return log['GW_dist'][::-1][0]
    
""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

def wgw_tree_distance(alpha=1,epsilon=1,method='shortest_path',features_metric='sqeuclidean',ratio=None):

    def dist(graph1,graph2):


        leaves1=graph1.return_leaves(graph1.tree)
        leaves2=graph2.return_leaves(graph2.tree)
        C1=graph1.distance_matrix(nodeOfInterest=leaves1,method=method)
        C2=graph2.distance_matrix(nodeOfInterest=leaves2,method=method)
        t1masses = np.ones(len(leaves1))/len(leaves1)
        t2masses = np.ones(len(leaves2))/len(leaves2)
        x1=graph1.leaves_matrix_attr().reshape(-1, 1) # A regarder si c'est pas le contraire !!!!!!
        x2=graph2.leaves_matrix_attr().reshape(-1, 1)

        if features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=features_metric) 
        M= M/np.max(M)

        if ratio is None :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',epsilon,alpha,max_iter=500,verbose=False,log=True)
        else :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',epsilon,epsilon*ratio,max_iter=500,verbose=False,log=True)

        return log['GW_dist'][::-1][0]
    
    return dist

def wgw_graph_distance(alpha=1,epsilon=1,method='shortest_path',features_metric='sqeuclidean',ratio=None): # il faut que les features aient la même dimension


    def dist(graph1,graph2):

        import ot
        import WGW_2 as wgw
        import numpy as np

        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        C1=graph1.distance_matrix(method=method)
        C2=graph2.distance_matrix(method=method)
        t1masses = np.ones(len(nodes1))/len(nodes1)
        t2masses = np.ones(len(nodes2))/len(nodes2)
        x1=graph1.all_matrix_attr().reshape(-1, 1)
        x2=graph2.all_matrix_attr().reshape(-1, 1)

        if features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=features_metric) 
        M= M/np.max(M)

        if ratio is None :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',epsilon,alpha,max_iter=500,verbose=False,log=True)
        else :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',epsilon,epsilon*ratio,max_iter=500,verbose=False,log=True)

        return log['GW_dist'][::-1][0]
    
    return dist

def wgw_ts_distance(alpha=1,epsilon=1,method='time',timedistance='sqeuclidean',features_metric='sqeuclidean',ratio=None): # il faut que les features aient la même dimension


    def dist(ts1,ts2):

        import WGW_2 as wgw
        import numpy as np

        C1=ts1.distance_matrix(method=method,timedistance=timedistance)
        C2=ts2.distance_matrix(method=method,timedistance=timedistance)

        t1masses = np.ones(len(ts1.values))/len(ts1.values)
        t2masses = np.ones(len(ts2.values))/len(ts2.values)

        x1=np.array(ts1.values).reshape(-1,1)
        x2=np.array(ts2.values).reshape(-1,1)

        if features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=features_metric) 
        M= M/np.max(M)

        if ratio is None :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',epsilon,alpha,max_iter=500,verbose=False,log=True)
        else :
            transpwgw,log= wgw.wgw(M,C1,C2,t1masses,t2masses,'square_loss',epsilon,epsilon*ratio,max_iter=500,verbose=False,log=True)

        return log['GW_dist'][::-1][0]
    
    return dist


""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

def gw_tree_distance(epsilon=1,method='shortest_path'): # il faut que les features aient la même dimension


    def dist(graph1,graph2):

        import WGW_2 as wgw
        import numpy as np

        leaves1=graph1.return_leaves(graph1.tree)
        leaves2=graph2.return_leaves(graph2.tree)
        C1=graph1.distance_matrix(nodeOfInterest=leaves1,method=method)
        C2=graph2.distance_matrix(nodeOfInterest=leaves2,method=method)
        t1masses = np.ones(len(leaves1))/len(leaves1)
        t2masses = np.ones(len(leaves2))/len(leaves2)

        transpwgw,log= wgw.wgw(np.zeros((C1.shape[0],C2.shape[0])),C1,C2,t1masses,t2masses,'square_loss',epsilon,alpha=1,max_iter=500,verbose=False,log=True)

        return log['GW_dist'][::-1][0]
    
    return dist

def gw_graph_distance(epsilon=1,method='shortest_path'): # il faut que les features aient la même dimension


    def dist(graph1,graph2):

        import WGW_2 as wgw
        import numpy as np

        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        C1=graph1.distance_matrix(method=method)
        C2=graph2.distance_matrix(method=method)
        t1masses = np.ones(len(nodes1))/len(nodes1)
        t2masses = np.ones(len(nodes2))/len(nodes2)

        transpwgw,log= wgw.wgw(np.zeros((C1.shape[0],C2.shape[0])),C1,C2,t1masses,t2masses,'square_loss',epsilon,alpha=1,max_iter=500,verbose=False,log=True)

        return log['GW_dist'][::-1][0]
    
    return dist

def gw_ts_distance(epsilon=1,method='time',timedistance='sqeuclidean'): # il faut que les features aient la même dimension


    def dist(ts1,ts2):

        import WGW_2 as wgw
        import numpy as np

        C1=ts1.distance_matrix(method=method,timedistance=timedistance)
        C2=ts2.distance_matrix(method=method,timedistance=timedistance)

        t1masses = np.ones(len(ts1.values))/len(ts1.values)
        t2masses = np.ones(len(ts2.values))/len(ts2.values)

        transpwgw,log= wgw.wgw(np.zeros((C1.shape[0],C2.shape[0])),C1,C2,t1masses,t2masses,'square_loss',epsilon,alpha=1,max_iter=500,verbose=False,log=True)

        return log['GW_dist'][::-1][0]
    
    return dist


""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

def emd_tree_distance(features_metric='sqeuclidean'):

    def dist(graph1,graph2):
        import ot
        import numpy as np

        leaves1=graph1.return_leaves(graph1.tree)
        leaves2=graph2.return_leaves(graph2.tree)
        t1masses = np.ones(len(leaves1))/len(leaves1)
        t2masses = np.ones(len(leaves2))/len(leaves2)
        x1=graph1.leaves_matrix_attr().reshape(-1, 1)
        x2=graph2.leaves_matrix_attr().reshape(-1, 1)

        if features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=features_metric) 
        M= M/np.max(M)

        transp = ot.emd(t1masses,t2masses, M)

        return np.sum(transp*M)

    return dist

def emd_graph_distance(features_metric='sqeuclidean'):

    def dist(graph1,graph2):

        import ot
        import numpy as np

        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        t1masses = np.ones(len(nodes1))/len(nodes1)
        t2masses = np.ones(len(nodes2))/len(nodes2)
        x1=graph1.all_matrix_attr().reshape(-1, 1)
        x2=graph2.all_matrix_attr().reshape(-1, 1)

        if features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=features_metric) 
        M= M/np.max(M)

        transp = ot.emd(t1masses,t2masses, M)

        return np.sum(transp*M)

    return dist

def emd_ts_distance(features_metric='sqeuclidean'):

    def dist(ts1,ts2):

        import ot
        import numpy as np
        N1=len(ts1.values)
        N2=len(ts2.values)
        t1masses = np.ones(N1)/N1
        t2masses = np.ones(N2)/N2

        x1=np.array(ts1.values).reshape(-1,1)
        x2=np.array(ts2.values).reshape(-1,1)

        if features_metric=='dirac':
            f=lambda x,y: x==y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=features_metric) 
        M= M/np.max(M)

        transp = ot.emd(t1masses,t2masses, M)

        return np.sum(transp*M)

    return dist




