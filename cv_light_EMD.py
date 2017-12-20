import numpy as np
import os,sys
from graph import *
from ts import *
import copy
import NN,time
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

path='./data/UCR_TS_Archive_2015/Lighting2/'
train=create_ts(path,t='TRAIN')
test=create_ts(path,t='TEST')
x_train,y_train=zip(*train)
x_test,y_test=zip(*test)

dir_path='./'
result_file='result_grid_lightning_WGW_test.csv'
text_file = open(os.path.join(dir_path, result_file), 'w')

n_splits=5
n_jobs=1
start_time = time.time()

print('CV Nb_splits : ', n_splits, file=text_file)
print('Data train size : ',len(x_train),file=text_file)
print('Data test size : ',len(x_train),file=text_file)
print('N_jobs : ',n_jobs,file=text_file)


#tuned_parameters = [{'epsilon':list(np.linspace(0.01,50,5)),
#					 'ratio':list(np.linspace(0.01,2.5,5))
#                    ,'method':['time']
#                     ,'normalize_distance':[False,True]
#                     ,'features_metric':['sqeuclidean']
#                     ,'timedistance':['sqeuclidean']}]

tuned_parameters = [{'features_metric':['sqeuclidean']}]

print('Tuned tuned_parameters : ',tuned_parameters,file=text_file) 

wgw_1NN=NN.Ts_EMD_1NN_Classifier()
clf = GridSearchCV(wgw_1NN, tuned_parameters, cv=n_splits,scoring='accuracy',verbose=1,n_jobs=n_jobs)
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

#nested_score = cross_val_score(clf, X=np.array(x_test), y=np.array(y_test)
#	,cv=n_splits,n_jobs=n_jobs,scoring='accuracy')
#nested_scores = nested_score.mean()
preds=clf.predict(np.array(x_test))
nested_scores=np.sum(preds==np.array(y_test))/len(y_test)

print('Score on test set with best_params_ : ',nested_scores,file=text_file)



print('All Time :', end_time-start_time, file=text_file)

