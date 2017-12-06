import numpy as np
import os,sys
module_path = os.path.abspath(os.path.join('/Users/vayer/Documents/OT/Python/GW_tests/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from graph import *
import copy
import NN,time
from sklearn.model_selection import StratifiedKFold

path='./mutag/'
dataset=list(build_mutag_dataset(path))
A,B=split_train_test(dataset,0.8)
x_train,y_train=zip(*A)
x_test,y_test=zip(*B)
dir_path='./'
result_file='result_grid_mutag.csv'
text_file = open(os.path.join(dir_path, result_file), 'w')
method_C1C2='shortest_path'

print('Train dataset length : ', len(x_train), file=text_file)
print('Test dataset length : ', len(x_test), file=text_file)
print('Methode pour C1 et C2: ',method_C1C2,file=text_file)

n_splits=2
start_time = time.time()
k_fold = StratifiedKFold(n_splits=n_splits)
print('CV Nb_splits : ', n_splits, file=text_file)

cv_scores = {}
for alpha in list(np.linspace(0.001,0.1,10)):  
    for epsilon in list(np.linspace(0.01,5,10)): 
        start_time = time.time()
        clf = NN.Graph_OT_1NN_Classifier(alpha,epsilon
            ,the_lower_the_better=True
            ,method=method_C1C2,features_metric='dirac')
        l_scores = []
        for idx_subtrain, idx_valid in k_fold.split(x_train,y_train):
            x_subtrain = [x_train[i] for i in idx_subtrain]
            y_subtrain = [y_train[i] for i in idx_subtrain]
            x_valid=[x_train[i] for i in idx_valid]
            y_valid=[y_train[i] for i in idx_valid]
            pred = clf.fit(x_subtrain,y_subtrain).predict(x_valid)
            l_scores.append(np.sum(pred == y_valid) / len(y_valid))
        cv_scores[alpha, epsilon] = np.mean(l_scores)
        print("--- %s seconds ---" % (time.time() - start_time), file=text_file)
        print ("params :", alpha, epsilon, " accuracy :", np.mean(l_scores), file=text_file)

best_param1, best_param2 = utils.dict_argmax(cv_scores)

end_time = time.time()
print('--------------------------', file=text_file)
print('--------------------------', file=text_file)
print('All Time :', end_time-start_time, file=text_file)
print('Alpha :',best_param1, file=text_file)
print('Epsilon :',best_param2, file=text_file)
print('Best score: ',max(cv_scores.values()), file=text_file)
