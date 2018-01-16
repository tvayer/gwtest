from sklearn.svm import SVC
import numpy as np
from sklearn.base import TransformerMixin
import itertools
from ot_distances import Gromov_Wasserstein_distance
from sklearn.preprocessing import StandardScaler
import time

class InfiniteException(Exception):
    pass


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
        self.similarity_measure_time=[]

    def reshaper(self,x):
        try:
            a=x.shape[1]
            return x
        except IndexError:
            return x.reshape(-1,1)

    def compute_similarity(self,x,y):
        k=0
        if (x.characterized(),y.characterized()) in self.similarities_dict: # ou suppose que les similarités sont symétriques
            #print('yes')
            similarity=self.similarities_dict[(x.characterized(),y.characterized())]
            k=1
        elif (y.characterized(),x.characterized()) in self.similarities_dict:
            #print('yes')
            similarity=self.similarities_dict[(y.characterized(),x.characterized())]
            k=1
        else:
            start=time.time()
            similarity=self.similarity_measure(x,y)
            end=time.time()
            self.similarity_measure_time.append(end-start)
            self.similarities_dict[(x.characterized(),y.characterized())]=similarity
        return similarity,k

    def fit(self,X,y=None,matrix=None):
        #print('FIT')
        self.classes_ =np.array(y)
        self._fit_X=np.array(X)
        Gtrain = np.zeros((X.shape[0],X.shape[0]))

        self.svc=SVC(C=self.C,kernel="precomputed",verbose=self.verbose)
        
        try :
            #print('Compute all distances')
            Gtrain = self.gram_matrix(X,X,matrix)
            #print('Done')
            #print('fit svc')
            self.svc.fit(Gtrain,self.classes_)
        except InfiniteException:
            print('Value error')
            self.svc.fit(np.zeros((Gtrain.shape[0],Gtrain.shape[1])),self.classes_)
        return self

    def predict(self,X,matrix=None):
        #print('PREDICT')

        try :
            #print('Compute all distances')
            G=self.gram_matrix(X,self._fit_X,matrix)
            #print('Done')
            preds=self.svc.predict(G)
        except InfiniteException:
            #print('Preds error')
            preds=np.repeat(-10,len(X)) # Ca c'est moche
        return preds

    def gram_matrix(self,X,Y,matrix=None):
        #print('gram_matrix')
        start=time.time()

        self.compute_all_distance(X,Y,matrix)
        end=time.time()
        #print('Compute all distance time :',(end-start))
        start=time.time()
        Z=np.exp(-self.gamma*self.D)

        if not self.assert_all_finite(Z):
            raise InfiniteException('Il y a des Nan')
        else :    
            if self.scale==True:
                std=StandardScaler()
                Z=std.fit_transform(Z)
        end=time.time()
        #print('SVM fit time :',(end-start))              
        return Z


    def assert_all_finite(self,X):
        """Like assert_all_finite, but only for ndarray."""

        X = np.asanyarray(X)
        a=X.dtype.char in np.typecodes['AllFloat']
        b=np.isfinite(X.sum())
        c=np.isfinite(X).all()

        if (a and not b and not c):
            return False 
        else :
            return True
            #raise InfiniteException("Input contains NaN, infinity"
                             #" or a value too large for %r." % X.dtype)

    def compute_all_distance(self,X,Y,matrix=None): # Il faut stocker ce kernel en dessous

        if matrix is not None :
            self.D=matrix[np.ix_(self.idx_X,self.idx_Y)]

        else:

            X=X.reshape(X.shape[0],) #idem
            Y=Y.reshape(Y.shape[0],) #idem

            if np.all(X==Y):
                D= np.zeros((X.shape[0], Y.shape[0]))
                H=np.zeros((X.shape[0], Y.shape[0]))
                R = np.zeros((X.shape[0], Y.shape[0]))

                for i, x1 in enumerate(X):
                    for j,x2 in enumerate(Y):
                        if j>=i:
                            dist,k=self.compute_similarity(x1, x2)
                            D[i, j] = dist
                            R[i, j] = k
                np.fill_diagonal(H,np.diagonal(D))
                D=D+D.T-H
                np.fill_diagonal(H,np.diagonal(R))
                R=R+R.T-H
            else:
                D = np.zeros((X.shape[0], Y.shape[0]))
                R = np.zeros((X.shape[0], Y.shape[0]))-1
                for i, x1 in enumerate(X):
                    row=[self.compute_similarity(x1, x2)[0] for j,x2 in enumerate(Y)]
                    D[i,:]=row

            if np.all(R):
                print('TOUTES LES DISTANCES ONT ETE CHERCHEES ')
            if np.all(R-1):
                print('TOUTES LES DISTANCES ONT ETE RECALCULEES ')


            self.D=D

       
    def set_one_param(self,dicto,key):
        if key in dicto:
            setattr(self, key, dicto[key])

    def get_params(self, deep=True):
        return {"similarity_measure":self.similarity_measure,"parallel":self.parallel,"gamma":self.gamma,"C":self.C}

    def set_params(self, **parameters):
        self.set_one_param(parameters,"similarity_measure")
        self.set_one_param(parameters,"parallel")
        self.set_one_param(parameters,"C")
        self.set_one_param(parameters,"gamma")
        self.similarities_dict=dict()

        return self

""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""

class Normalized_SVC_Classifier(GenericSVCClassifier):

    def __init__(self,similarity_measure,C=1,gamma=1,parallel=False,verbose=False,normalize_distance=False,scale=False):
        self.normalize_distance=normalize_distance
        GenericSVCClassifier.__init__(self,similarity_measure,C,gamma,parallel,verbose,scale)
        self.similarity_time=[]

    def compute_similarity(self,x,y):
        start=time.time()

        if self.normalize_distance==True:

            if x.characterized()!=y.characterized():
                similaritydiff,k1=super(Normalized_SVC_Classifier,self).compute_similarity(x,y) # Attention
                similarityself1,k2=super(Normalized_SVC_Classifier,self).compute_similarity(x,x) # Attention
                similarityself2,k3=super(Normalized_SVC_Classifier,self).compute_similarity(y,y) # Attention

                similarity=2*similaritydiff-similarityself1-similarityself2
                k=k1 and k2 and k3

            else:
                # Pour éviter de calculer 0 on fixe
                similarity=0
                # On l'ajoute quand même dans le dictionnaire
                similarity_useless,k=super(Normalized_SVC_Classifier,self).compute_similarity(x,x)
        else:
            similarity,k=super(Normalized_SVC_Classifier,self).compute_similarity(x,y) # Attention

        end=time.time()
        self.similarity_time.append(end-start)
        #print('time to compute one similarity : ',1000*(end-start))
        return similarity,k

        

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

    def fit(self,X,y=None,matrix=None):
        self.classes_ = y
        self._fit_X = list(X.reshape(X.shape[0],)) 
        start=time.time()
        for x in self._fit_X :
            if x.C is None or x.name_struct_dist!=self.method:
                #print('Construction des matrices de structures')
                C=x.distance_matrix(method=self.method,force_recompute=True)
                #print('Done')
        end=time.time()
        #print('Temps pour calculer toutes les structures :',(end-start))

        super(Graph_WGW_SVC_Classifier,self).fit(X,y,matrix)

    def get_params(self, deep=True):
        return {"alpha":self.alpha,"epsilon":self.epsilon,"normalize_distance":self.normalize_distance
        ,"features_metric":self.features_metric,"method":self.method,"C":self.C,"gamma":self.gamma,"scale":self.scale}

    def set_params(self, **parameters):
        #print('---------------------')
        #print(self.get_params())
        #print('---------------------')
        self.set_one_param(parameters,"alpha")
        self.set_one_param(parameters,"features_metric")
        self.set_one_param(parameters,"method")
        self.set_one_param(parameters,"epsilon")
        self.set_one_param(parameters,"normalize_distance")
        self.set_one_param(parameters,"C")
        self.set_one_param(parameters,"gamma")
        self.set_one_param(parameters,"scale")

        gw2=Gromov_Wasserstein_distance(alpha=self.alpha,epsilon=self.epsilon,method=self.method,features_metric=self.features_metric)
        if self.gw.get_tuning_params()!=gw2.get_tuning_params():
            print('CHANGE PARAMETERS : RAZ similarities_dict')
            self.gw=Gromov_Wasserstein_distance(alpha=self.alpha,epsilon=self.epsilon,method=self.method,features_metric=self.features_metric)
            self.similarity_measure=self.gw.graph_d
            self.similarities_dict=dict()
        else :
            a=1+1
            #print('KEEP THE similarities_dict')

        return self


class TS_WGW_SVC_Classifier(Normalized_SVC_Classifier):

    def __init__(self,C=1,gamma=1,alpha=1,epsilon=1,method='sqeuclidean',features_metric='sqeuclidean',
        parallel=False,verbose=False,normalize_distance=False,scale=False):
        self.gw=Gromov_Wasserstein_distance(alpha=alpha,epsilon=epsilon,method=method,features_metric=features_metric)
        similarity_measure=self.gw.ts_d
        self.alpha=alpha
        self.epsilon=epsilon
        self.features_metric=features_metric
        self.method=method
        Normalized_SVC_Classifier.__init__(self,C=C,gamma=gamma,similarity_measure=similarity_measure,parallel=parallel,verbose=verbose,normalize_distance=normalize_distance,scale=scale)

    def fit(self,X,y=None,matrix=None):
        self.classes_ = y
        self._fit_X = list(X.reshape(X.shape[0],)) 
        start=time.time()
        for x in self._fit_X :
            if x.C is None or x.name_struct_dist!=self.method:
                #print('Construction des matrices de structures')
                C=x.distance_matrix(method=self.method,force_recompute=True)
                #print('Done')
        end=time.time()
        #print('Temps pour calculer toutes les structures :',(end-start))

        super(TS_WGW_SVC_Classifier,self).fit(X,y,matrix)

    def get_params(self, deep=True):
        return {"alpha":self.alpha,"epsilon":self.epsilon,"normalize_distance":self.normalize_distance
        ,"features_metric":self.features_metric,"method":self.method,"C":self.C,"gamma":self.gamma,"scale":self.scale}

    def set_params(self, **parameters):
        #print('---------------------')
        #print(self.get_params())
        #print('---------------------')
        self.set_one_param(parameters,"alpha")
        self.set_one_param(parameters,"features_metric")
        self.set_one_param(parameters,"method")
        self.set_one_param(parameters,"epsilon")
        self.set_one_param(parameters,"normalize_distance")
        self.set_one_param(parameters,"C")
        self.set_one_param(parameters,"gamma")
        self.set_one_param(parameters,"scale")

        gw2=Gromov_Wasserstein_distance(alpha=self.alpha,epsilon=self.epsilon,method=self.method,features_metric=self.features_metric)
        if self.gw.get_tuning_params()!=gw2.get_tuning_params():
            print('CHANGE PARAMETERS : RAZ similarities_dict')
            self.gw=Gromov_Wasserstein_distance(alpha=self.alpha,epsilon=self.epsilon,method=self.method,features_metric=self.features_metric)
            self.similarity_measure=self.gw.ts_d
            self.similarities_dict=dict()
        else :
            a=1+1
            #print('KEEP THE similarities_dict')

        return self
