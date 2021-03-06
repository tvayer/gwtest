import numpy as np
import os,sys
module_path = os.path.abspath(os.path.join('/Users/vayer/Documents/OT/Python/GW_tests/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from graph import *
from custom_svc import *
import copy
import NN,time
from sklearn.model_selection import GridSearchCV

path='./data/MUTAG_2/'
dataset=list(build_NCI1_dataset(path))
X,y=zip(*dataset)

rationtraintest=0.9

A,B=split_train_test(dataset,rationtraintest)
x_train,y_train=zip(*A)
x_test,y_test=zip(*B)

dir_path='./'
result_file='result_scv_mutag.csv'
text_file = open(os.path.join(dir_path, result_file), 'w')

n_splits=5
n_jobs=-1
start_time = time.time()

print('CV Nb_splits : ', n_splits, file=text_file)
print('Data size : ',len(X),file=text_file)
print('Train/test : ',rationtraintest)


tuned_parameters = [{'epsilon':list(np.linspace(0.005,1,8)),
                    'alpha':list(np.linspace(0,0.99,10))
                     ,'method':['shortest_path']
                     ,'normalize_distance':[True,False]
                     ,'features_metric':['dirac']
                     ,'scale':[False,True]
                     ,'C':[10**k for k in [-1,0,1,2]]
                     ,'gamma':[2**k for k in [-2,-1,0,1,2]]}]

print('Tuned tuned_parameters : ',tuned_parameters,file=text_file) 

graph_svc=Graph_WGW_SVC_Classifier()
clf = GridSearchCV(graph_svc, tuned_parameters, cv=n_splits,scoring='accuracy',verbose=1,n_jobs=n_jobs)
clf.fit(np.array(x_train).reshape(-1,1),np.array(y_train))

print('--------------------------', file=text_file)
print('--------------------------', file=text_file)
print('', file=text_file)
print("Best parameters set found on development set:", file=text_file)
print('', file=text_file)
print(clf.best_params_, file=text_file)
print('', file=text_file)
print('--------------------------', file=text_file)
print('--------------------------', file=text_file)
print("Grid scores on development set:", file=text_file)
print('', file=text_file)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
        % (mean, std * 2, params), file=text_file)
print('', file=text_file)


end_time = time.time()
print('--------------------------', file=text_file)
print('--------------------------', file=text_file)

preds=clf.predict(np.array(x_test))
nested_scores=np.sum(preds==np.array(y_test))/len(y_test)

print('Score on test set with best_params_ : ',nested_scores,file=text_file)

print('All Time :', end_time-start_time, file=text_file)

