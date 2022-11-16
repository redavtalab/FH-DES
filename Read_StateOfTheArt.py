
import pickle

import pandas as pd

import myfunctions as mf
from scipy.stats import rankdata
#import pandas as pd
#import openml
#from openml.tasks import TaskType
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import FuncFormatter
import numpy as np
import math
from itertools import chain
#methods_names = ['KNORA-U', 'OLA', 'DES-KNN','META-DES', 'FH-DES_W', 'FH-DES_M']
#datasets = {"Circle", "P2"}
#datasets = { "Thyroid","Breast","Wine","Heart"}
def read_results(tec_name,datasetName):
    path = ExperimentPath + "/Results/" + tec_name +"_"+datasetName + "_result.p"
    poolspec = open(path, mode="rb")
    accuracy = pickle.load(poolspec)
    labels = pickle.load(poolspec)
    yhat = pickle.load(poolspec)
    return accuracy,labels,yhat

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


#
# "Adult",
# "Audit2",
# "Cardiotocography",
# "Chess",
# "Credit-screening",
# "CTG",
# "Ecoli",
# "Glass",
# "Pima",
# "Transfusion"

 # "Wholesale",
    #  "Transfusion", low oracle accuracy

    # 30 Dataset
    # Has problem: "Adult", "Glass",  "Ecoli",    "Seeds",         "Voice9"
    # Large: "Magic", "CTG",  "Segmentation", "WDVG1",
}


datasets = sorted(datasets)
# mf.print_dataset_table(datasets)

no_itr = 20

ExperimentPath = "Experiment1"
methods_names = ['KNORA-U', 'KNORA-E', 'MCB', 'DESKNN', 'OLA', 'RANK', 'KNOP', 'META-DES','FH_2v']
# methods_names = ['FH_1v', 'FH_2v', 'FH_3v', 'FH_4v', 'FH_5v', 'FH_6v', 'FH_7v', 'FH_8v', 'FH_9v', 'FH_10v']
# methods_names = ['FH_1p', 'FH_2p', 'FH_3p', 'FH_4p', 'FH_5p', 'FH_6p', 'FH_7p', 'FH_8p', 'FH_9p', 'FH_10p']
# methods_names = ['MV', 'SB', 'FMM', 'FH_2v', 'FH_4v','FH_9v']
No_methods = len(methods_names)


##### Reading Pickle File
dataset_counter = 0
overall_results = np.zeros((len(datasets), No_methods, no_itr))
for dataInd, datasetName in enumerate(datasets):
    result = []
    for  tecInd, tecName in enumerate(methods_names):
        accuracy, labels, yhat = read_results(tecName,datasetName)
        overall_results[dataInd,tecInd,:] = accuracy


 ##### Adding IJCNN Results
if(methods_names[0] == "FH_1v" or methods_names[0] == "FH_1"):

        overall_results[:, 0, 0] = [96.87, 89.5, 99.34, 76.55, 96.61, 74.11, 91.29, 70.38, 74.86, 71.56, 83.09, 70.65,
                                    88.13, 82.5, 71.8, 90.5, 69.14, 78.87, 87.64, 78.1, 76.28, 79.62, 75.08, 71.37, 95.98,
                                    75.05, 84.04, 76.58, 82.43, 98]
        for i in range(20):
            overall_results[:, 0, i] = overall_results[:, 0, 0]
        sd1 = [0.92, 1.68, 0.50, 2.64, 1.61, 1.25, 3.63, 2.03, 2.27, 3.98, 4.42, 2.59, 1.37, 3.49, 4.08, 2.32, 4.34, 2.73,
               3.24, 0.91, 2.72, 5.42, 2.00, 1.33, 1.37, 2.41, 3.23, 3.09, 4.39, 1.71]

if (methods_names[0] == "FH_1p" or methods_names[0] == "FH_1"):

    overall_results[:, 0, 0] = [96.87, 89.12, 99.13, 77.46, 96.28, 74.63, 87.77, 70.12, 74.8, 71.36, 82.79, 71.3,
                                  88.35, 81.94, 71.01, 89.63, 67.82, 78.03, 86.2, 77.81, 74.9, 81.35, 75.26, 70.72,
                                  95.87, 74.53, 82.44, 77.17, 81.25, 97.11]
    for i in range(20):
        overall_results[:, 0, i] = overall_results[:, 0, 0]
    sd1 = [.74, 2.37, .66, 1.98, 1.49, 1.08, 3.93, 1.84, 2.0, 3.88, 3.89, 3.1, 2.21, 3.25, 4.15, 2.08, 4.77, 3.2, 2.85,
           .97, 2.46, 6.41, 1.91, 1.67, 1.32, 2.1, 3.54, 3.42, 4.94, 2.23]



####################################    Write in latex      ###################################################
mf.write_in_latex_table_IJCNN(overall_results,datasets,methods_names)

##############################################         Win_Tie_Loss           #####################################
compared_index = -1
# ind_list = list(chain (range(0,3), range(3,len(methods_names)-1)))
ind_list = range(len(methods_names))

win = np.zeros((len(ind_list)-1,))
tie = np.zeros((len(ind_list)-1,))
loss = np.zeros((len(ind_list)-1,))
no_exp = overall_results[:, 0 , :].shape[0]
dataset_methods_acc = np.average(overall_results,2)
## win-tie-loss FH-DES-M
kk = 0
for ind in ind_list[:-1]:
    no_win = np.sum(dataset_methods_acc[:, ind]  < (dataset_methods_acc[:, compared_index] ))
    no_loss = np.sum(dataset_methods_acc[:, ind] > (dataset_methods_acc[:, compared_index] ))
    win[kk] = no_win
    loss[kk] = no_loss
    tie[kk] = no_exp - no_win - no_loss
    kk +=1

nc = no_exp/2 + 1.645*np.sqrt(no_exp)/2
met_list = [methods_names[i] for i in ind_list[:-1]]
mf.plot_winloss(met_list ,win,tie,loss,nc, without_tie = False)


############################ Ranking - CD diagram  ############################

d_ranks = np.zeros_like(overall_results[:,0:-1,0])
# errors_mat = 100 - np.delete(dataset_methods_acc,2,1)
errors_mat = 100 - dataset_methods_acc

d_ranks = rankdata(errors_mat,axis=1)

ranks = np.average(d_ranks,axis=0) - 1
# m_list = [methods_names[i] for i in chain(range(2),range(3,No_methods))]
m_list = [methods_names[i] for i in range(No_methods)]
mf.plot_CD(m_list,ranks,len(datasets))


np.average(rankdata(np.average(overall_results, axis=2), axis=1), axis=0)
np.average(rankdata(np.average(overall_results, axis=2), axis=1), axis=0)
print("Overal Accuracy:")
scores = np.average(np.average(overall_results, axis=2), axis=0)
# sc = np.concatenate((scores[:2],scores[3:]))
# scores[0] = 81.89
mf.plot_overallresult(scores, methods_names)
print(np.round(ranks,2))
