"""
====================================================================
Calculate number of hyperboxes for each FH-DES method
(using the results of ST_16techniques.py code)
====================================================================
"""
import os
import warnings
import myfunctions as mf
import numpy as np
import pandas as pd
import pickle

def load_model(tec_name,datasetName):
    path = ExperimentPath + "/Models/" + tec_name +"_"+ datasetName + "_model.p"
    poolspec = open(path, mode="rb")
    return pickle.load(poolspec)

def calculate_NO_hyperboxes(datasetName):

    state = 0
    NO_Box = np.zeros((NO_techniques,))
    #
    # for itr in range(0, no_itr):
    #     pool_classifiers, X_train, y_train, X_test, y_test, X_DSEL, y_DSEL, list_ds, methods_names = mf.load_elements(ExperimentPath, datasetName, itr, state)
    #     for method_ind in range(10,13):
    #         NO_Box[method_ind-10] += len(list_ds[method_ind].HBoxes)
    #     state += 1
    [X_train, X_test, X_DSEL, y_train, y_test, y_DSEL] = np.load('Datasets3/' + datasetName + str(3) + '.npy',
                                                                 allow_pickle=True)
    NO_samples = len(y_DSEL)
    # NO_Box = NO_Box / no_itr
    counter = -1
    for tec in range(NO_techniques):
        counter += 1
        results = []
        labels = []
        yhat = []
        ds_tec = load_model(methods_names[tec],datasetName)
        for itr in range(no_itr):
            if methods_names[tec] == 'Oracle':
                NO_Box[counter] += 555
            else:
                NO_Box[counter] += ds_tec[itr].NO_hypeboxes

    NO_Box = NO_Box / no_itr
    return NO_samples,NO_Box

ExperimentPath = "Experiment1"
theta = .1
NO_Hyperbox_Thereshold = 0.85
NO_classifiers =100
no_itr = 20
do_train = False


datasets = {
    #     Data set of DGA1033 report
    "Audit",
    "Banana",
    "Banknote",
    "Blood",
    "Breast",
    "Car",
    "Datausermodeling",
    "Faults",
    "German",
    "Haberman",
    "Heart",
    "ILPD",
    "Ionosphere",
    "Laryngeal1",
    "Laryngeal3",
    "Lithuanian",
    "Liver",
    "Mammographic",
    "Monk2",
    "Phoneme",
    "Pima",
    "Sonar",
    "Statlog",
    "Steel",
    "Thyroid",
    "Vehicle",
    "Vertebral",
    "Voice3",
    "Weaning",
    "Wine"
}

datasets = sorted(datasets)
methods_names = ['FH_1v', 'FH_2v', 'FH_3v', 'FH_4v', 'FH_5v', 'FH_6v','FH_7v', 'FH_8v', 'FH_9v', 'FH_10v']
# methods_names = ['FH_1p', 'FH_2p', 'FH_3p', 'FH_4p', 'FH_5p', 'FH_6p', 'FH_7p', 'FH_8p', 'FH_9p', 'FH_10p']
NO_techniques = len(methods_names)
NO_datasets = len(datasets)
whole_results = np.zeros([NO_datasets,NO_techniques,no_itr])


dataset_count = 0
no_samples = np.zeros((len(datasets)))
NO_box = np.zeros((len(datasets),NO_techniques))

for datasetName in datasets:
    no_samples[dataset_count],NO_box[dataset_count,:] = calculate_NO_hyperboxes(datasetName)
    dataset_count +=1

NO_box = np.round(NO_box,2)
# no_Hyperbox_M = np.round(np.append(no_Hyperbox_M,np.average(no_Hyperbox_M))/100,2)
names = methods_names.copy()
names = np.insert(names,0,"No_samples")
results = np.concatenate((no_samples.reshape(NO_datasets,1),NO_box),axis=1)
# mf.write_in_latex_table(results,datasets,names)

dic = {}
dic['DataSets'] = list(datasets)

dic.update({na: list(val) for na, val in zip(names, results.T)})
df = pd.DataFrame(dic)
print(df.to_latex(index=False))
print("Average NO_Boxes: ", np.round(np.average(NO_box,0),2))
print("NO_Samples: " ,np.round(np.average(no_samples),2))