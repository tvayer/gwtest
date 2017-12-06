import numpy as np
import os,sys
module_path = os.path.abspath(os.path.join('/Users/vayer/Documents/OT/Python/GW_tests/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from graph import *
import copy
import NN,time
from sklearn.model_selection import GridSearchCV

path='./mutag/'
dataset=list(build_mutag_dataset(path))
X,y=zip(*dataset)
dir_path='./'
result_file='result_grid_mutag_GW.csv'
text_file = open(os.path.join(dir_path, result_file), 'w')

n_splits=10
start_time = time.time()

print('CV Nb_splits : ', n_splits, file=text_file)
print('Data size : ',len(X),file=text_file)

tuned_parameters = [{'features_metric':['dirac']}]

print('Tuned tuned_parameters : ',tuned_parameters,file=text_file) 

emd_1NN=NN.Graph_EMD_1NN_Classifier()
clf = GridSearchCV(emd_1NN, tuned_parameters, cv=n_splits,verbose=1)
y2=np.array(y)
y2[y2==-1]=0 #je sais pas si ça sert à quelque chose
clf.fit(np.array(X).reshape(-1,1),y2)


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
print('All Time :', end_time-start_time, file=text_file)

