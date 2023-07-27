
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
# methods_names = ['KNORA-U', 'KNORA-E', 'MCB', 'DESKNN', 'OLA', 'RANK', 'KNOP', 'META-DES','FH_4-M']
methods_names = ['FH_1-M', 'FH_2-M', 'FH_3-M', 'FH_4-M', 'FH_5-M', 'FH_6-M', 'FH_7-M', 'FH_8-M', 'FH_9-M', 'FH_10-M']
# methods_names = ['FH_1-C', 'FH_2-C', 'FH_3-C', 'FH_4-C', 'FH_5-C', 'FH_6-C', 'FH_7-C', 'FH_8-C', 'FH_9-C', 'FH_10-C']
# methods_names = ['MV', 'SB', 'GFMM', 'FH_4-M']
No_methods = len(methods_names)


##### Reading Pickle File
dataset_counter = 0
overall_results = np.zeros((len(datasets), No_methods, no_itr))
for dataInd, datasetName in enumerate(datasets):
    result = []
    for  tecInd, tecName in enumerate(methods_names):
        accuracy, labels, yhat = read_results(tecName,datasetName)
        overall_results[dataInd,tecInd,:] = accuracy



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

# d_ranks = np.zeros_like(overall_results[:,0:-1,0])  # When the list includes the Oracle
# errors_mat = 100 - np.delete(dataset_methods_acc,2,1)

d_ranks = np.zeros_like(overall_results[:,:,0]) # When the list does not include the Oracle

errors_mat = 100 - dataset_methods_acc
d_ranks = rankdata(errors_mat,axis=1)
ranks = np.average(d_ranks,axis=0)
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
