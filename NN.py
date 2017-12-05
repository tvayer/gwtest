from sklearn.neighbors import KNeighborsClassifier
import numpy,os,sys
module_path = os.path.abspath(os.path.join('/Users/vayer/Documents/OT/Python/GW_tests/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pathos.multiprocessing import ProcessingPool as Pool
import ot_distances 

class Generic1NNClassifier(KNeighborsClassifier):
    def __init__(self, similarity_measure, the_lower_the_better=True,parallel=False,verbose=False):
        KNeighborsClassifier.__init__(self, n_neighbors=1)
        self.local_metric = similarity_measure
        self.the_lower_the_better = the_lower_the_better
        self.parallel=parallel
        self.verbose=verbose

    def fit(self, X, y):
        self.classes_ = y
        self._fit_X = X
        return self

    def predict(self, X):

        pred = []
        S=[]

        for Xi in X:

            if self.parallel==False:
                if self.verbose==True:
                    similarities=[]
                    k=0
                    for  X_train in self._fit_X:
                        similarities.append(self.local_metric(X_train, Xi))
                        if k % 20 == 0:
                            print(k)
                        k=k+1
                else:
                    similarities = [self.local_metric(X_train, Xi) for X_train in self._fit_X]

            else :
                pool=Pool(3)
                g=lambda X_train:self.local_metric(X_train,Xi) 
                similarities=pool.map(g,self._fit_X)

            if self.the_lower_the_better:
                best_match_idx = numpy.argmin(similarities)
            else:
                best_match_idx = numpy.argmax(similarities)


            pred.append(self.classes_[best_match_idx])
            S.append(similarities)

        return numpy.array(pred),S

class Tree_OT_1NN_Classifier(Generic1NNClassifier):

    def __init__(self,alpha,epsilon,the_lower_the_better=True,parallel=False,verbose=False,method='shortest_path',features_metric='sqeuclidean'):
        similarity_measure=ot_distances.wgw_tree_distance(alpha,epsilon,method,features_metric)
        Generic1NNClassifier.__init__(self,similarity_measure,the_lower_the_better,parallel,verbose)


class Graph_OT_1NN_Classifier(Generic1NNClassifier):

    def __init__(self,alpha,epsilon,the_lower_the_better=True,parallel=False,verbose=False,method='shortest_path',features_metric='sqeuclidean'):
        similarity_measure=ot_distances.wgw_graph_distance(alpha,epsilon,method,features_metric)
        Generic1NNClassifier.__init__(self,similarity_measure,the_lower_the_better,parallel,verbose)






