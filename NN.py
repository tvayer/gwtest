from sklearn.neighbors import KNeighborsClassifier
import numpy
from pathos.multiprocessing import ProcessingPool as Pool

class Generic1NNClassifier(KNeighborsClassifier):
    def __init__(self, similarity_measure, the_lower_the_better=False,parallel=False):
        KNeighborsClassifier.__init__(self, n_neighbors=1)
        self.local_metric = similarity_measure
        self.the_lower_the_better = the_lower_the_better
        self.parallel=parallel

    def fit(self, X, y):
        self.classes_ = y
        self._fit_X = X
        return self

    def predict(self, X):

        pred = []

        for Xi in X:

            if self.parallel==False:

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

        return numpy.array(pred)
