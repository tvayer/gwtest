from sklearn.neighbors import KNeighborsClassifier
import numpy,os,sys
module_path = os.path.abspath(os.path.join('/Users/vayer/Documents/OT/Python/GW_tests/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pathos.multiprocessing import ProcessingPool as Pool
from ot_distances import emd_tree_distance,emd_graph_distance
from ot_distances import gw_tree_distance,gw_graph_distance
from ot_distances import wgw_graph_distance,wgw_tree_distance
from ot_distances import Gromov_distance,Wasserstein_distance,Gromov_Wasserstein_distance
from sklearn.base import TransformerMixin

""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""
class Generic1NNClassifier(KNeighborsClassifier,TransformerMixin):
    def __init__(self, similarity_measure, the_lower_the_better=True,parallel=False,verbose=False):
        KNeighborsClassifier.__init__(self, n_neighbors=1)
        self.similarity_measure = similarity_measure
        self.the_lower_the_better = the_lower_the_better
        self.parallel=parallel
        self.verbose=verbose
        self.similarities_dict=dict()

    def compute_similarity(self,x,y):
        if (x,y) in self.similarities_dict: # ou suppose que les similarités sont symétriques
            similarity=self.similarities_dict[(x,y)]
        elif (y,x) in self.similarities_dict:
            similarity=self.similarities_dict[(y,x)]
        else:
            similarity=self.similarity_measure(x,y)
            self.similarities_dict[(x,y)]=similarity
        return similarity

    def fit(self, X, y=None):
        self.classes_ = y
        self._fit_X = list(X.reshape(X.shape[0],)) #Ouais faudra gérer ça proprement
        return self

    def predict(self, X):

        pred = []
        S=[]
        X=list(X.reshape(X.shape[0],)) #idem
        for Xi in X:

            if self.parallel==False:
                if self.verbose==True:
                    similarities=[]
                    k=0
                    for  X_train in self._fit_X:
                        similarities.append(self.compute_similarity(Xi,X_train))
                        if k % 20 == 0:
                            print(k)
                            #print(similarities)
                        k=k+1
                else:
                    similarities = [self.compute_similarity(X_train,Xi) for X_train in self._fit_X]
            else :
                pool=Pool(3)
                g=lambda X_train:self.compute_similarity(X_train,Xi) 
                similarities=pool.map(g,self._fit_X)

            if self.the_lower_the_better:
                try :
                    best_match_idx = numpy.nanargmin(similarities)
                except ValueError:
                    best_match_idx =numpy.nan
            else:
                try :
                    best_match_idx = numpy.nanargmax(similarities)
                except ValueError:
                    best_match_idx =numpy.nan
            if self.verbose==True:
                print(best_match_idx)
            try:
                pred.append(self.classes_[best_match_idx])
            except IndexError:
                if self.verbose==True:
                    print('-----')
                    print(best_match_idx)
                    print('-----')
                    print(self.get_params())
                #pred.append(numpy.nan)
                pred.append(-10) #très dangeureux à mettre au propre
            S.append(similarities)

        self.similarities=S
        return numpy.array(pred)

    def transform(self,X):     
        return self.predict(X)[0]
    
    def get_params(self, deep=True):
        return {"the_lower_the_better": self.the_lower_the_better,"similarity_measure":self.similarity_measure,"parallel":self.parallel}

    def set_params(self, **parameters):
        self.the_lower_the_better = parameters["the_lower_the_better"]
        self.similarity_measure=parameters["similarity_measure"]
        self.parallel=parameters["parallel"]
        return self

""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

class Graph_EMD_1NN_Classifier(Generic1NNClassifier):
    def __init__(self,features_metric='sqeuclidean',the_lower_the_better=True,parallel=False,verbose=False):
        wd=Wasserstein_distance(features_metric)
        similarity_measure=wd.graph_d
        self.features_metric=features_metric
        Generic1NNClassifier.__init__(self,similarity_measure,the_lower_the_better,parallel,verbose)

    def get_params(self, deep=True):
        return {"features_metric":self.features_metric}

    def set_params(self, **parameters):
        self.features_metric = parameters["features_metric"]
        wd=Wasserstein_distance(features_metric)
        self.similarity_measure=wd.graph_d
        return self

class Tree_EMD_1NN_Classifier(Generic1NNClassifier):
    def __init__(self,features_metric='sqeuclidean',the_lower_the_better=True,parallel=False,verbose=False):
        wd=Wasserstein_distance(features_metric)
        self.similarity_measure=wd.tree_d
        self.features_metric=features_metric
        Generic1NNClassifier.__init__(self,similarity_measure,the_lower_the_better,parallel,verbose)

    def get_params(self, deep=True):
        return {"features_metric":self.features_metric}

    def set_params(self, **parameters):
        self.features_metric = parameters["features_metric"]
        wd=Wasserstein_distance(self.features_metric)
        self.similarity_measure=wd.tree_d

        return self

class Ts_EMD_1NN_Classifier(Generic1NNClassifier):
    def __init__(self,features_metric='sqeuclidean',the_lower_the_better=True,parallel=False,verbose=False):
        wd=Wasserstein_distance(features_metric)
        similarity_measure=wd.ts_d
        self.features_metric=features_metric
        Generic1NNClassifier.__init__(self,similarity_measure,the_lower_the_better,parallel,verbose)

    def get_params(self, deep=True):
        return {"features_metric":self.features_metric}

    def set_params(self, **parameters):
        self.features_metric = parameters["features_metric"]
        wd=Wasserstein_distance(self.features_metric)
        self.similarity_measure=wd.ts_d

        return self

""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

class Normalized_1NN_Classifier(Generic1NNClassifier):

    def __init__(self,similarity_measure,the_lower_the_better=True
        ,parallel=False,verbose=False,normalize_distance=False):
        self.normalize_distance=normalize_distance
        Generic1NNClassifier.__init__(self,similarity_measure,the_lower_the_better,parallel,verbose)

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
            similarity=super(Normalized_1NN_Classifier,self).compute_similarity(x,y) # Attention

        return similarity

    def fit(self, X, y=None):
        #print('JE FIT')
        self.classes_ = y
        self._fit_X = list(X.reshape(X.shape[0],)) #Ouais faudra gérer ça proprement
        if self.normalize_distance==True:
            print("Les distances entre pour chaque point et eux mêmes sont calculées")
            if self.parallel==False:
                for X_train in self._fit_X: #A gérer proprement
                    a=self.compute_similarity(X_train,X_train) #on s'en fout ça fait 0 donc a=
            else:
                pool=Pool(3)
                def g(Xi):
                    a=self.compute_similarity(Xi,self._fit_X) 
                self.auto_dist=pool.map(g,X)

        return self

    def get_params(self, deep=True):
        a=super(Normalized_1NN_Classifier,self).get_params(deep)
        b={"normalize_distance":self.normalize_distance}
        return dict(a, **b)

    def set_params(self, **parameters):
        super(Normalized_1NN_Classifier,self).set_params(**parameters)
        self.normalize_distance = parameters["normalize_distance"]
        return self

""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

class Tree_GW_1NN_Classifier(Normalized_1NN_Classifier):
    def __init__(self,epsilon=1,the_lower_the_better=True
        ,parallel=False,verbose=False,method='shortest_path',normalize_distance=False):
        gd=Gromov_distance(epsilon=epsilon,method=method)
        #similarity_measure=gw_tree_distance(epsilon,method)
        similarity_measure=gd.tree_d
        self.epsilon=epsilon
        self.method=method
        Normalized_1NN_Classifier.__init__(self,similarity_measure=similarity_measure
            ,the_lower_the_better=the_lower_the_better,parallel=parallel,verbose=verbose,normalize_distance=normalize_distance)

    def get_params(self, deep=True):
        return {"epsilon":self.epsilon,"method":self.method,"normalize_distance":self.normalize_distance}

    def set_params(self, **parameters):
        self.epsilon=parameters["epsilon"]
        self.method=parameters["method"]
        self.normalize_distance=parameters["normalize_distance"]
        gd=Gromov_distance(epsilon=self.epsilon,method=self.method)
        self.similarity_measure=gd.tree_d
        return self

class Graph_GW_1NN_Classifier(Normalized_1NN_Classifier):
    def __init__(self,epsilon=1,the_lower_the_better=True
        ,parallel=False,verbose=False,method='shortest_path',normalize_distance=False):
        gd=Gromov_distance(epsilon=epsilon,method=method)
        #similarity_measure=gw_tree_distance(epsilon,method)
        similarity_measure=gd.graph_d
        self.epsilon=epsilon
        self.method=method
        Normalized_1NN_Classifier.__init__(self,similarity_measure=similarity_measure
            ,the_lower_the_better=the_lower_the_better,parallel=parallel,verbose=verbose,normalize_distance=normalize_distance)

    def get_params(self, deep=True):
        return {"epsilon":self.epsilon,"method":self.method,"normalize_distance":self.normalize_distance}

    def set_params(self, **parameters):
        self.epsilon=parameters["epsilon"]
        self.method=parameters["method"]
        self.normalize_distance=parameters["normalize_distance"]
        gd=Gromov_distance(epsilon=self.epsilon,method=self.method)
        self.similarity_measure=gd.graph_d

        return self

class Ts_GW_1NN_Classifier(Normalized_1NN_Classifier):
    def __init__(self,epsilon=1,the_lower_the_better=True
        ,parallel=False,verbose=False,method='time',timedistance='sqeuclidean',normalize_distance=False):
        gd=Gromov_distance(epsilon=epsilon,method=method,timedistance=timedistance)
        similarity_measure=gd.ts_d
        self.epsilon=epsilon
        self.method=method
        self.timedistance=timedistance
        Normalized_1NN_Classifier.__init__(self,similarity_measure=similarity_measure
            ,the_lower_the_better=the_lower_the_better,parallel=parallel,verbose=verbose,normalize_distance=normalize_distance)

    def get_params(self, deep=True):
        return {"epsilon":self.epsilon,"method":self.method,"normalize_distance":self.normalize_distance,"timedistance":self.timedistance}

    def set_params(self, **parameters):
        self.epsilon=parameters["epsilon"]
        self.method=parameters["method"]
        self.timedistance=parameters["timedistance"]
        self.normalize_distance=parameters["normalize_distance"]
        gd=Gromov_distance(epsilon=self.epsilon,method=self.method,timedistance=self.timedistance)
        self.similarity_measure=gd.ts_d

        return self

""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

class Tree_WGW_1NN_Classifier(Normalized_1NN_Classifier):

    def __init__(self,ratio=1,epsilon=1,method='shortest_path',features_metric='sqeuclidean',the_lower_the_better=True
        ,parallel=False,verbose=False,normalize_distance=False):
        gw=Gromov_Wasserstein_distance(ratio=ratio,epsilon=epsilon,method=method,features_metric=features_metric)
        similarity_measure=gw.tree_d
        self.ratio=ratio
        self.epsilon=epsilon
        self.features_metric=features_metric
        self.method=method
        Normalized_1NN_Classifier.__init__(self,similarity_measure=similarity_measure
            ,the_lower_the_better=the_lower_the_better,parallel=parallel,verbose=verbose,normalize_distance=normalize_distance)

    def get_params(self, deep=True):
        return {"ratio":self.ratio,"epsilon":self.epsilon,"normalize_distance":self.normalize_distance
        ,"features_metric":self.features_metric,"method":self.method}

    def set_params(self, **parameters):
        self.ratio = parameters["ratio"]
        self.features_metric = parameters["features_metric"]
        self.method = parameters["method"]
        self.epsilon=parameters["epsilon"]
        self.normalize_distance=parameters["normalize_distance"]
        gw=Gromov_Wasserstein_distance(ratio=self.ratio,epsilon=self.epsilon,method=self.method,features_metric=self.features_metric)
        self.similarity_measure=gw.tree_d

        return self


class Graph_WGW_1NN_Classifier(Normalized_1NN_Classifier):

    def __init__(self,ratio=1,epsilon=1,method='shortest_path',features_metric='sqeuclidean',the_lower_the_better=True
        ,parallel=False,verbose=False,normalize_distance=False):
        gw=Gromov_Wasserstein_distance(ratio=ratio,epsilon=epsilon,method=method,features_metric=features_metric)
        similarity_measure=gw.graph_d
        self.ratio=ratio
        self.epsilon=epsilon
        self.features_metric=features_metric
        self.method=method
        Normalized_1NN_Classifier.__init__(self,similarity_measure=similarity_measure
            ,the_lower_the_better=the_lower_the_better,parallel=parallel,verbose=verbose,normalize_distance=normalize_distance)

    def get_params(self, deep=True):
        return {"ratio":self.ratio,"epsilon":self.epsilon,"normalize_distance":self.normalize_distance
        ,"features_metric":self.features_metric,"method":self.method}

    def set_params(self, **parameters):
        self.ratio = parameters["ratio"]
        self.features_metric = parameters["features_metric"]
        self.method = parameters["method"]
        self.epsilon=parameters["epsilon"]
        self.normalize_distance=parameters["normalize_distance"]
        gw=Gromov_Wasserstein_distance(ratio=self.ratio,epsilon=self.epsilon,method=self.method,features_metric=self.features_metric)
        self.similarity_measure=gw.graph_d

        return self

class Ts_WGW_1NN_Classifier(Normalized_1NN_Classifier):

    def __init__(self,ratio=1,epsilon=1,method='time',features_metric='sqeuclidean',timedistance='sqeuclidean',the_lower_the_better=True
        ,parallel=False,verbose=False,normalize_distance=False):
        gw=Gromov_Wasserstein_distance(ratio=ratio,epsilon=epsilon,method=method,features_metric=features_metric,timedistance=timedistance)
        similarity_measure=gw.ts_d
        self.ratio=ratio
        self.epsilon=epsilon
        self.features_metric=features_metric
        self.method=method
        self.timedistance=timedistance
        Normalized_1NN_Classifier.__init__(self,similarity_measure=similarity_measure
            ,the_lower_the_better=the_lower_the_better,parallel=parallel,verbose=verbose,normalize_distance=normalize_distance)

    def get_params(self, deep=True):
        return {"ratio":self.ratio,"epsilon":self.epsilon,"normalize_distance":self.normalize_distance
        ,"features_metric":self.features_metric,"method":self.method,"timedistance":self.timedistance}

    def set_params(self, **parameters):
        self.ratio = parameters["ratio"]
        self.features_metric = parameters["features_metric"]
        self.method = parameters["method"]
        self.epsilon=parameters["epsilon"]
        self.normalize_distance=parameters["normalize_distance"]
        self.timedistance=parameters["timedistance"]
        gw=Gromov_Wasserstein_distance(ratio=self.ratio,epsilon=self.epsilon,method=self.method,features_metric=self.features_metric,timedistance=self.timedistance)
        self.similarity_measure=gw.ts_d

        return self









