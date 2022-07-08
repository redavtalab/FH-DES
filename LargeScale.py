"""
====================================================================
Dynamic selection with linear classifiers: Statistical Experiment
====================================================================
"""

import pickle
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from deslib.des import DESKNN, FHDES_JFB, DESFHMW_JFB, DESFHMW_allboxes, FHDES_prior, FHDES_Allboxes_vector, \
    FHDES_JFB_vector, DESFHMW_JFB_vector, FHDES_prior_vector, DESFHMW_prior_vector, DESFHMW_allboxes_vector, METADES

from deslib.static.oracle import Oracle
from deslib.static.single_best import SingleBest
from deslib.util.datasets import make_P2

import sklearn.preprocessing as preprocessing
import scipy.io as sio
import time
import os
import warnings
import xlsxwriter
import math
from myfunctions import *

warnings.filterwarnings("ignore")


#+##############################################################################


# Prepare the DS techniques. Changing k value to 7.
def initialize_ds(pool_classifiers, uncalibratedpool, X_DSEL, y_DSEL, k=7):
    FH_1v = FHDES_JFB_vector(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                             doContraction=False, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    FH_2v = FHDES_JFB_vector(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                             doContraction=True, thetaCheck=False, multiCore_process=True, shuffle_dataOrder=False)

    FH_3v = FHDES_Allboxes_vector(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                                  doContraction=False, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    FH_4v = FHDES_Allboxes_vector(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                                  doContraction=True, thetaCheck=False, multiCore_process=True, shuffle_dataOrder=False)

    FH_5v = FHDES_JFB_vector(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                             doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    FH_6v = DESFHMW_JFB_vector(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                               doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)

    FH_7v = FHDES_Allboxes_vector(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                                  doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    FH_8v = DESFHMW_allboxes_vector(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold,
                                    mis_sample_based=True,
                                    doContraction=True, thetaCheck=True, multiCore_process=True,
                                    shuffle_dataOrder=False)

    FH_9v = FHDES_prior_vector(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                               doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    FH_10v = DESFHMW_prior_vector(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                                  doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    meta = METADES(pool_classifiers, k=k)

    oracle = Oracle(pool_classifiers)
    single_best = SingleBest(pool_classifiers, n_jobs=-1)
    majority_voting = pool_classifiers

    list_ds = [meta, oracle , FH_1v, FH_2v, FH_3v, FH_4v, FH_5v, FH_6v, FH_7v, FH_8v, FH_9v, FH_10v]

    methods_names = ['META-DES', 'Oracle', 'FH_1v', 'FH_2v', 'FH_3v', 'FH_4v', 'FH_5v', 'FH_6v', 'FH_7v', 'FH_8v', 'FH_9v', 'FH_10v']
    # # methods_names = ['MV', 'SB', 'Oracle', 'KNORA-U', 'KNORA-E', 'DESKNN', 'OLA', 'RANK', 'KNOP', 'META-DES', 'MCB', 'FH_4']
    #                 # 'FHm_cN_tY','FHm_cY_tN','FHm_cY_tY' ]

    # fit the ds techniques
    for ds in list_ds:
        if ds != majority_voting:
            ds.fit(X_DSEL, y_DSEL)

    return list_ds, methods_names


def write_NO_Hbox(Dataset,NO_HBox,M_Names):
    wpath = ExperimentPath + '/0' + Dataset + '.xlsx'

    workbook = xlsxwriter.Workbook(wpath)

    # Write Accuracy Sheet
    worksheet = workbook.add_worksheet('NO_HBox')
    rez = np.concatenate((NO_HBox, np.average(NO_HBox, 0).reshape(-1, len(M_Names))), 0)

    for i in range(len(M_Names)):
        worksheet.write(0,i,M_Names[i])
        worksheet.write_column(1, i, rez[:,i])

    worksheet.write(0, 12, Dataset)

    workbook.close()
def write_results_to_file(EPath, accuracy,labels,yhat, methods, datasetName):
    path =  EPath + "/" + datasetName + "Final Results.p"
    rfile = open(path, mode="wb")
    pickle.dump(methods,rfile)
    pickle.dump(accuracy,rfile)
    pickle.dump(labels,rfile)
    pickle.dump(yhat,rfile)
    rfile.close()
def run_process(X_train, X_DSEL, X_test, y_train, y_DSEL, y_test,n):

    state = 0
    rng = np.random.RandomState(state)
    result_one_dataset = np.zeros((NO_techniques, no_itr))
    predicted_labels = np.zeros((NO_techniques, no_itr,len(y_test)))
    yhat = np.zeros((no_itr, len(y_test)))
    NO_Box = np.zeros((no_itr,NO_techniques-2))

    for itr in range(0, no_itr):
        rng = np.random.RandomState(state)
        if do_train:

            yhat[itr, :] = y_test

            ###########################################################################
            #                               Training                                  #
            ###########################################################################
            learner = Perceptron(max_iter=100, tol=10e-3, alpha=0.001, penalty=None, random_state=rng)
            calibratedmodel = CalibratedClassifierCV(learner, cv=5,method='isotonic')
            # learner = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=rng)
            # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, ), random_state=rng)
            uncalibratedpool = BaggingClassifier(learner,n_estimators=NO_classifiers,bootstrap=True,
                                                 max_samples=1.0,
                                                 random_state=rng)
            # uncalibratedpool.fit(X_train, y_train)

            pool_classifiers = BaggingClassifier(calibratedmodel, n_estimators=NO_classifiers, bootstrap=True,
                                                 max_samples=1.0,
                                                 random_state=rng)
            pool_classifiers.fit(X_train,y_train)

            list_ds, methods_names = initialize_ds(pool_classifiers,uncalibratedpool, X_DSEL, y_DSEL, k=7)
            # for i in range(2,12):
            #     print("NO_samples: ", len(y_DSEL) ,"Methode: ", methods_names[i], "NO-HBox: ", len(list_ds[i].HBoxes),)
            for method_ind in range(2, NO_techniques):
                NO_Box[itr,method_ind - 2] = list_ds[method_ind].NO_hypeboxes

            if(save_all_results):
                save_elements(ExperimentPath+"/Pools" ,datasetName + np.str(n),itr,state,pool_classifiers,X_train,y_train,X_test,y_test,X_DSEL,y_DSEL,list_ds,methods_names)
        else: # do_not_train
            pool_classifiers,X_train,y_train,X_test,y_test,X_DSEL,y_DSEL,list_ds,methods_names = load_elements(ExperimentPath , datasetName + np.str(n),itr,state)
        ###########################################################################
        #                               Generalization                            #
        ###########################################################################

        for ind in range(0, len(list_ds)):
            result_one_dataset[ind, itr] = list_ds[ind].score(X_test, y_test) * 100
            if ind==1: # Oracle results --> y should be passed too.
                predicted_labels[ind, itr, :] = list_ds[ind].predict(X_test,y_test)
                continue
            predicted_labels[ind, itr,:] = list_ds[ind].predict(X_test)
        state += 1
    write_NO_Hbox(datasetName+np.str(n),NO_Box, methods_names[2:])
    write_results_to_file(ExperimentPath, result_one_dataset, predicted_labels, yhat, methods_names, datasetName+np.str(n))
    return result_one_dataset,methods_names,list_ds

theta = .27
NO_Hyperbox_Thereshold = 0.99
ExperimentPath = "Experiment_LargeScale"
NO_classifiers =100
no_itr = 5
save_all_results = False
do_train = True
NO_techniques = 12

list_ds = []
methods_names = []
# n_samples_ = [ 1000, 10000, 100000, 300000, 500000, 700000,900000]
n_samples_ = [ 100000 ]
NO_datasets = len(n_samples_)
whole_results = np.zeros([NO_datasets,NO_techniques,no_itr])


dataset_count = 0
done_list = []


####################################################################################
#                               First Dataset (Data)
####################################################################################
X, y = make_classification(n_samples=n_samples_[-1] + 2000,
                           n_features=5,
                           random_state=1)

scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)

X_DSE, X_tt, y_DSE , y_tt = train_test_split(X, y, test_size=2000, stratify=y, random_state=1)  # stratify=y
datasetName = "Data"
####################################################################################################
# Spliting the Tarin and Test data
X_train, X_test, y_train, y_test = train_test_split(X_tt, y_tt, test_size=0.5, stratify=y_tt,
                                                                random_state=1)  # stratify=y

for n in n_samples_:
    X_DSEL = X_DSE[:n,:]
    y_DSEL = y_DSE[:n]
    # print("X_DSEL size is:",X_DSEL.shape)
    result,methods_names,list_ds = run_process(X_train, X_DSEL, X_test, y_train, y_DSEL,  y_test,n)
    whole_results[dataset_count,:,:] = result
    dataset_count +=1
    done_list.append(datasetName+np.str(n))

write_whole_results_into_excel(ExperimentPath,whole_results, done_list.copy(), methods_names)
path = ExperimentPath + "/WholeResults.p"
rfile = open(path, mode="wb")
pickle.dump(whole_results,rfile)
datasets = done_list
pickle.dump(datasets,rfile)
pickle.dump(methods_names,rfile)
rfile.close()

write_in_latex_table(whole_results,done_list,methods_names,rows="datasets")



####################################################################################
#                               Second Dataset (Sensor)
####################################################################################

# whole_results = np.zeros([NO_datasets,NO_techniques,no_itr])
#
#
# dataset_count = 0
# done_list = []
#
# # Sensor Dataset ############################################################################################
# redata = sio.loadmat("LargDatasets/Sensor_900.mat")
# data = redata['dataset']
# X_DSE = data[:, 0:-1]
# y_DSE = data[:, -1]
#
# redata = sio.loadmat("LargDatasets/Sensor_tt.mat")
# ttdata = redata['dataset']
# X_tt = ttdata[:, 0:-1] # Tarin and Test data
# y_tt = ttdata[:, -1]
# datasetName = "Sensor"
#
# # Spliting the Tarin and Test data
# X_train, X_test, y_train, y_test = train_test_split(X_tt, y_tt, test_size=0.5, stratify=y_tt,
#                                                                 random_state=1)  # stratify=y
#
# for n in n_samples_:
#     X_DSEL = X_DSE[:n,:]
#     y_DSEL = y_DSE[:n]
#     # print("X_DSEL size is:",X_DSEL.shape)
#     result,methods_names,list_ds = run_process(X_train, X_DSEL, X_test, y_train, y_DSEL,  y_test,n)
#     whole_results[dataset_count,:,:] = result
#     dataset_count +=1
#     done_list.append(datasetName+np.str(n))
#
# write_whole_results_into_excel(ExperimentPath,whole_results, done_list.copy(), methods_names)
# path = ExperimentPath + "/WholeResults.p"
# rfile = open(path, mode="wb")
# pickle.dump(whole_results,rfile)
# datasets = done_list
# pickle.dump(datasets,rfile)
# pickle.dump(methods_names,rfile)
# rfile.close()
#
# write_in_latex_table(whole_results,done_list,methods_names,rows="datasets")