
from sklearn.model_selection import StratifiedKFold
from pathos.multiprocessing import ProcessingPool as Pool
import itertools
import utils
import os,time
import numpy as np
from sklearn.base import clone
from copy import deepcopy
import logging
import ast

# IL FAUT CREER AUTANT D'ESTIMATEUR QUE DE PARAMETRES ????
# CHANGER LA MANIERE DE LOG CA FAIT NIMP : avec une classe extérieure.
class GridSearch():
    def __init__(self,estimator,tuned_parameters,csv,nb_splits=5,n_jobs=1,parallel=False,verbose=False):
        self.estimator=clone(estimator)
        self.tuned_parameters=tuned_parameters
        self.nb_splits=nb_splits
        self.n_jobs=n_jobs
        self.dir_path='./'
        self.parallel=parallel
        self.verbose=verbose

        # Set up a specific logger with our desired output level
        logging.basicConfig(filename=csv.split('.')[0]+'.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
        logging.info('ALL PARAMS GRID: '+str(self.tuned_parameters))



        self.csv = csv
        self.cv_scores = {}
        self.l_scores= {}



    def explode_tuned_parameters(self):
        tuned_parameters2=self.tuned_parameters[0]
        varNames = sorted(tuned_parameters2)
        combinations = [dict(zip(varNames, prod)) for prod in itertools.product(*(tuned_parameters2[varName] for varName in varNames))]

        return combinations

    def filter_dict(self,old_dict,your_keys):
        return { your_key: old_dict[your_key] for your_key in your_keys }


    def parameters_for_distance(self,combinations):
        filtered_combinations=[]
        filtered_combinations.append(combinations[0])
        filtre=set(self.estimator.gw.get_tuning_params().keys()).intersection(set(combinations[0].keys()))
        filter_base=self.filter_dict(combinations[0],filtre)
        allreadydone=[]
        for elt in combinations:
            filter_elt=self.filter_dict(elt,filtre)
            if filter_elt!=filter_base and filter_elt not in allreadydone:
                allreadydone.append(filter_elt)
                filtered_combinations.append(elt)

        return filtered_combinations

    def fit_one_params(self,params):

        startinit=time.time()


        filtre=set(self.estimator.gw.get_tuning_params().keys()).intersection(set(self.combinations[0].keys()))

        #Quel estimateur va t'on cloner ? celui qui a les m^mes filtered params 
        start=time.time()
        filtered_params=self.filter_dict(params,filtre)
        end=time.time()
        if self.verbose==True:
            logging.info('Filter in fit_one_params : '+str(end-start))
        #print('go')
        #estimator=clone(self.dict_base_estimator[repr(filtered_params)])
        #estimator.similarities_dict=self.dict_base_estimator[repr(filtered_params)]
        #estimator=deepcopy(self.dict_base_estimator[repr(filtered_params)])

        ######### ICI QUE C'EST TRICKY ###############
        start=time.time()
        estimator=clone(self.dict_base_estimator[repr(filtered_params)])
        #estimator=deepcopy(self.dict_base_estimator[repr(filtered_params)])
        #estimator=self.dict_base_estimator[repr(filtered_params)] # IL FAUT QUE l'ESTIMATEUR AIT TOUTES LES DONNEES
        #estimator.similarities_dict=self.dict_base_estimator[repr(filtered_params)].similarities_dict # on a sans doute pas besoin de ça
        #estimator.alldistance_matrix=self.dict_base_estimator[repr(filtered_params)].alldistance_matrix # on a sans doute pas besoin de ça
        #estimator.usematrix=self.dict_base_estimator[repr(filtered_params)].usematrix

        end=time.time()
        if self.verbose==True:
            logging.info('Get esti + egalite dict in fit_one_params : '+str(end-start))

        
        #print(len(estimator.similarities_dict))

        #print('ok')
        #print('estimator.set_params(**params)')
        start=time.time()
        estimator.set_params(**params)
        end=time.time()
        if self.verbose==True:
            logging.info('set params in fit_one_params : '+str(end-start))

        estimator.idx_X=self.idx_subtrain
        estimator.idx_Y=self.idx_subtrain
        end=time.time()
        if self.verbose==True:
            logging.info('Time initialisation in fit_one_params : '+str(end-startinit))
        #estimator.idx_valid=self.idx_valid
        #print('estimator.set_params(**params) : OK')

        #print('estimator.fit')
        start=time.time()
        estimator.fit(np.array(self.x_subtrain),y=np.array(self.y_subtrain),matrix=self.dict_base_matrix[repr(filtered_params)])
        end=time.time()
        if self.verbose==True:
            logging.info('Time fit in fit_one_params : '+str(end-start))
        #print('estimator.fit : OK')

        #print('estimator.predict')

        estimator.idx_X=self.idx_valid # Un peu chiant c'est inversé
        estimator.idx_Y=self.idx_subtrain

        start=time.time()
        preds=estimator.predict(np.array(self.x_valid),matrix=self.dict_base_matrix[repr(filtered_params)])
        end=time.time()
        if self.verbose==True:
            logging.info('Time predict in fit_one_params : '+str(end-start)+' for '+repr(params))
        #print('estimator.predict: OK')

        endend=time.time()

        if np.all(preds==-10):
            logging.info('Predictions equal -10 for parameters : '+repr(params))
            estimator.error=np.nan

        else: 
            estimator.error=np.sum(preds == self.y_valid) / len(self.y_valid)

        logging.info('Time fit_one_params : '+str(endend-startinit))

        return estimator

    def calc_distance(self,X,params):

        #print('self.estimator.get_params() : ',self.estimator.get_params())
        #est=deepcopy(self.estimator) # JE PENSE QUE C EST LA LE SOUCIS
        est=clone(self.estimator)
        est.set_params(**params)

        #print('est.get_params() : ',est.get_params())
        start=time.time()
        est.compute_all_distance(X,X)
        end=time.time()
        logging.info('-------------------------')
        logging.info('Done for parameters  : '+repr(params))
        logging.info('Time calc_distance  : '+str(end-start))
        #est.alldistance_matrix=est.D
        #print(repr(params))

        filtre=set(est.gw.get_tuning_params().keys()).intersection(set(self.combinations[0].keys()))
        filtered_params=self.filter_dict(params,filtre)

        #print(repr(filtered_params))

        self.dict_base_matrix[repr(filtered_params)]=est.D
        #print('X : ',len(X))
        #print('est.similarities_dict : ',len(est.similarities_dict))
        #print('est.D : ',est.D.shape)


        return est 


    def fit(self,X,y):
        logging.info("-----------Begin fit-----------")


        self.list_base_estimator=[]
        self.dict_base_estimator={}
        self.dict_base_matrix={}

        self.dict_estimator={}

        self.start_time=time.time()

        self.combinations=self.explode_tuned_parameters()
        logging.info('All combinations : '+str(len(self.combinations)))

        filtre=set(self.estimator.gw.get_tuning_params().keys()).intersection(set(self.combinations[0].keys()))
        #print('filtre : ',filtre )
        #print(' ')

        logging.info('----------CALCULS DES MATRICES DE DISTANCE---------------')

        #if self.parallel==True:
        #    pool=Pool(self.n_jobs)
        #    self.list_base_estimator=pool.map(f,self.parameters_for_distance(self.combinations)) # je l'applique pas au bon truc ?
        #else:
        self.list_base_estimator=[self.calc_distance(X,params) for params in self.parameters_for_distance(self.combinations)] # je l'applique pas au bon truc ?


        logging.info('---------- FIN CALCULS DES MATRICES DE DISTANCE---------------')

        #print("parameters_for_distance : ",self.parameters_for_distance(self.combinations))
        #print(" ")
        #print("self.list_base_estimator : ", len(self.list_base_estimator))

        for estimator in self.list_base_estimator:
            #print('---')
            #print('estimator.get_params : ',estimator.get_params())
            #print('estimator.get_params_filtered : ',self.filter_dict(estimator.get_params(),filtre))
            self.dict_base_estimator[repr(self.filter_dict(estimator.get_params(),filtre))]=estimator
        #print(' ')
        #print('self.dict_base_estimator : ',self.dict_base_estimator)

        logging.info('----------DEBUT VALIDATION CROISEE---------------')

        start_cv=time.time()
        k_fold = StratifiedKFold(n_splits=self.nb_splits)


        i=0
        for idx_subtrain, idx_valid in k_fold.split(X,y):
            self.idx_subtrain=idx_subtrain
            self.idx_valid=idx_valid
            x_subtrain = [X[i] for i in idx_subtrain]
            y_subtrain = [y[i] for i in idx_subtrain]
            x_valid=[X[i] for i in idx_valid]
            y_valid=[y[i] for i in idx_valid]
            self.x_subtrain=x_subtrain
            self.y_subtrain=y_subtrain
            self.x_valid=x_valid
            self.y_valid=y_valid
                

            if self.parallel==True:
                pool=Pool(self.n_jobs)
                self.dict_estimator[i]=pool.map(g,self.combinations)
            else:
                start=time.time()
                self.dict_estimator[i]=[self.fit_one_params(params) for params in self.combinations]
                end=time.time()
                logging.info('---------')
                logging.info('TIME FOR ONE FOLD FOR ALL PARAMS: '+str(end-start))
                logging.info('---------')

            i=i+1



            logging.info('########## ONE FOLD DONE ###############')

        self.endtime=time.time()
        logging.info('----------FIN VALIDATION CROISEE---------------')
        logging.info('VALIDATION CROISEE TIME : '+str(self.endtime-start_cv))

        logging.info("-----------End fit-----------")
        logging.info("ALL TIME : --- %s seconds ---" % (self.endtime - self.start_time))

        self.write_results()


    def write_results(self):

        self.cv_results={}
        for k,v in self.dict_estimator.items():
            for elt in v:
                if repr(elt.get_params()) not in self.cv_results:
                    self.cv_results[repr(elt.get_params())]=[]
                self.cv_results[repr(elt.get_params())].append(elt.error)

        for k,v in self.cv_results.items():
            self.cv_results[k] = (np.mean(v),np.std(v))

        self.best_param1 = utils.dict_argmax(self.cv_results)

        results = open(os.path.join(self.dir_path, 'train_'+self.csv), 'w')

        print("GRID SEARCH RESULTS  ",file=results)
        print("---------------------------- ",file=results)

        print('Tuned tuned_parameters : ',self.tuned_parameters,file=results)
        print("  ",file=results)

        print('Number of parameters combinations : ',len(self.combinations),file=results)
        print("  ",file=results)

        print('Number of parameters fit : ',self.nb_splits*len(self.combinations),file=results)
        print("  ",file=results)

        print('CV splits : ',self.nb_splits,file=results)
        print("  ",file=results)

        print("ALL TIME : --- %s seconds ---" % (self.endtime - self.start_time),file=results)
        print("Best params: ",self.best_param1,file=results)
        print("  ",file=results)
        print("------------GRID---------------- ",file=results)

        for k,v in self.cv_results.items():
            print("%0.3f (+/-%0.03f) for %r" % (self.cv_results[k][0], self.cv_results[k][1] * 2, k), file=results)

    def test_best_estimator(self,x_train,y_train,x_test,y_test):

        results = open(os.path.join(self.dir_path, 'test_'+self.csv), 'w')
        est=clone(self.estimator)
        est.set_params(**ast.literal_eval(self.best_param1))
        

        startall=time.time()
        logging.info("---------------------------- ")
        logging.info("Fit best estimator on train ")
        est.fit(x_train,y_train)
        end=time.time()
        logging.info('TIME FIT '+str(end-startall))

        start=time.time()
        print("---------------------------- ",file=results)
        print("Predict best estimator on test   ",file=results)
        preds=est.predict(x_test)
        end=time.time()

        logging.info('TIME PREDICT '+str(end-start))

        error=np.sum(preds == y_test) / len(y_test)

        print("---------------------------- ",file=results)
        print("PRECISION ON TEST",error,file=results)









