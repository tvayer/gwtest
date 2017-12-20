from sklearn.svm import SVC
import numpy as np
from sklearn.base import TransformerMixin
import itertools
from ot_distances import Gromov_Wasserstein_distance
from sklearn.preprocessing import StandardScaler



class GenericSVCClassifier(TransformerMixin):
    def __init__(self,similarity_measure,C=1,gamma=1,parallel=False,verbose=False,scale=False):
        self.similarity_measure = similarity_measure
        self.gamma=gamma
        self.C=C
        self.parallel=parallel
        self.verbose=verbose
        self.similarities_dict=dict() #là pour na pas calculer deux fois dist(x,y) et dist(y,x)
        self.D=None
        self.scale=scale

    def compute_similarity(self,x,y):
        if (x,y) in self.similarities_dict: # ou suppose que les similarités sont symétriques
            similarity=self.similarities_dict[(x,y)]
        elif (y,x) in self.similarities_dict:
            similarity=self.similarities_dict[(y,x)]
        else:
            similarity=self.similarity_measure(x,y)
            self.similarities_dict[(x,y)]=similarity
        return similarity

    def fit(self,X,y=None):
        self.classes_ =np.array(y)
        print('Compute all distances')
        self._fit_X = self.gram_matrix(X)
        print('Done')
        self.svc=SVC(C=self.C,kernel="precomputed",verbose=self.verbose)
        print('fit svc')
        self.svc.fit(self._fit_X,self.classes_)
        print('done')
        return self

    def predict(self,X):
        print('Compute all distances')
        self.gram_matrix(X)
        print('Done')
        return self.svc.predict(self.gram_matrix(X))

    def gram_matrix(self,X):
        self.compute_all_distance(X)
        Z=np.exp(-self.gamma*self.D)
        if self.scale==True:
            std=StandardScaler()
            Z=std.fit_transform(Z)

        return Z

    def compute_all_distance(self,X): # Il faut stocker ce kernel en dessous

        X=list(X.reshape(X.shape[0],)) #idem
        v=X
        pairs = list(itertools.combinations(X,2))
        D=np.zeros((len(v),len(v)))
        self.map_node=dict([i for i in enumerate(v)]) # à créer ailleurs 
        self.inv_map_node = {v: k for k, v in self.map_node.items()} # à créer ailleurs 

        for (s,e) in pairs: # cette boucle est longue : OUI
            distance=self.compute_similarity(s,e)
            D[self.inv_map_node[s]-1,self.inv_map_node[e]-1]=distance

        self.D=D
       

    def get_params(self, deep=True):
        return {"similarity_measure":self.similarity_measure,"parallel":self.parallel,"gamma":self.gamma,"C":self.C}

    def set_params(self, **parameters):
        self.similarity_measure=parameters["similarity_measure"]
        self.parallel=parameters["parallel"]
        self.C=parameters["C"]
        self.gamma=parameters["gamma"]
        self.similarities_dict=dict()

        return self

""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

class Normalized_SVC_Classifier(GenericSVCClassifier):

    def __init__(self,similarity_measure,C=1,gamma=1,parallel=False,verbose=False,normalize_distance=False,scale=False):
        self.normalize_distance=normalize_distance
        GenericSVCClassifier.__init__(self,similarity_measure,C,gamma,parallel,verbose,scale)

    def compute_similarity(self,x,y):
        if self.normalize_distance==True:
            if x!=y:
                #print('Ils sont diff ',x,y,x.name,y.name)
                if (x,y) in self.similarities_dict:
                    #print('Je prends ',x.name,y.name)
                    similarity=self.similarities_dict[(x,y)]
                elif (y,x) in self.similarities_dict:
                    #print('Je prends ',x.name,y.name)
                    similarity=self.similarities_dict[(y,x)]
                else:
                    #print('sheisse_xy entre ',x.name,y.name)
                    similaritydiff=self.similarity_measure(x,y)
                    if (x,x) in self.similarities_dict:
                        similarityself1=self.similarities_dict[(x,x)]
                    else:
                        #print('sheisse_xx entre ',x.name,x.name)
                        similarityself1=self.similarity_measure(x,x)
                        self.similarities_dict[(x,x)]=similarityself1
                    if (y,y) in self.similarities_dict:
                        similarityself2=self.similarities_dict[(y,y)]
                    else:
                        #print('sheisse_yy entre ',y.name,y.name)
                        similarityself2=self.similarity_measure(y,y)
                        self.similarities_dict[(y,y)]=similarityself2
                    similarity=2*similaritydiff-similarityself1-similarityself2
                    self.similarities_dict[(x,y)]=similarity                
            else:
                #print('Go_self')              
                similarity=0
                if (x,x) not in self.similarities_dict:
                    #print('Je rentre ',x.name,x.name)
                    self.similarities_dict[(x,x)]=self.similarity_measure(x,x)
        else:
            similarity=super(Normalized_SVC_Classifier,self).compute_similarity(x,y) # Attention

        return similarity

    def get_params(self, deep=True):
        a=super(Normalized_SVC_Classifier,self).get_params(deep)
        b={"normalize_distance":self.normalize_distance}
        return dict(a, **b)

    def set_params(self, **parameters):
        super(Normalized_SVC_Classifier,self).set_params(**parameters)
        self.normalize_distance = parameters["normalize_distance"]
        return self

""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

class Graph_WGW_SVC_Classifier(Normalized_SVC_Classifier):

    def __init__(self,C=1,gamma=1,alpha=1,epsilon=1,method='shortest_path',features_metric='sqeuclidean',
    	parallel=False,verbose=False,normalize_distance=False,scale=False):
        self.gw=Gromov_Wasserstein_distance(alpha=alpha,epsilon=epsilon,method=method,features_metric=features_metric)
        similarity_measure=self.gw.graph_d
        self.alpha=alpha
        self.epsilon=epsilon
        self.features_metric=features_metric
        self.method=method
        Normalized_SVC_Classifier.__init__(self,C=C,gamma=gamma,similarity_measure=similarity_measure,parallel=parallel,verbose=verbose,normalize_distance=normalize_distance,scale=scale)

    def fit(self,X,y=None):
        self.classes_ = y
        self._fit_X = list(X.reshape(X.shape[0],)) 
        print('Construction des matrices de structures')
        for x in self._fit_X :
            if x.C is None or x.name_struct_dist!=self.method:
                C=x.distance_matrix(method=self.method,force_recompute=True)
        print('Done')

        super(Graph_WGW_SVC_Classifier,self).fit(X,y)

    def get_params(self, deep=True):
        return {"alpha":self.alpha,"epsilon":self.epsilon,"normalize_distance":self.normalize_distance
        ,"features_metric":self.features_metric,"method":self.method}

    def set_params(self, **parameters):
        print('---------------------')
        print(self.get_params())
        print('---------------------')
        self.alpha = parameters["alpha"]
        self.features_metric = parameters["features_metric"]
        self.method = parameters["method"]
        self.epsilon=parameters["epsilon"]
        self.normalize_distance=parameters["normalize_distance"]      
        gw2=Gromov_Wasserstein_distance(alpha=self.alpha,epsilon=self.epsilon,method=self.method,features_metric=self.features_metric)
        if self.gw.get_tuning_params()!=gw2.get_tuning_params():
            self.gw=gw2
            self.similarity_measure=gw2.graph_d
            self.similarities_dict=dict()

        return self
