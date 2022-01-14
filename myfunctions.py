import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import *
import Orange
import xlsxwriter
from deslib.util import *
import scipy.io as sio

import math
def print_dataset_table(list_datasets):
    NO_features = np.zeros((len(list_datasets)))
    NO_samples = np.zeros((len(list_datasets)))
    i =0

    for datasetName in list_datasets:
        redata = sio.loadmat("DataSets/" + datasetName + ".mat")
        data = redata['dataset']
        X = data[:, 0:-1]
        y = data[:, -1]
        NO_samples[i] = len(y)
        NO_features[i] = np.size(X,1)

        i+=1
    datasets = {}

    datasets['Group'] = ['*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*','*']
    datasets['Name'] = list(list_datasets)
    datasets['#Samples'] = NO_samples
    datasets['#Features'] = NO_features
    df = pd.DataFrame(datasets)
    print(df.to_latex())




def statistical_differences(y,yc,ym):
    qtest_score = Q_statistic(y,yc,ym)
    agrement_measure = agreement_measure(y,yc,ym)
    correlation_coef = correlation_coefficient(y,yc,ym)
    disagree_measure = disagreement_measure(y,yc,ym)
    doub_fault = double_fault(y,yc,ym)
    # ne_double_fault = negative_double_fault(y,yc,ym)
    ratio_err = ratio_errors(y,yc,ym)
    return np.round([qtest_score, agrement_measure, correlation_coef, disagree_measure, doub_fault, ratio_err],2)
def write_whole_results_into_excel(results,datasets,methods,path = None):
    total_average = np.round(np.average(np.average(results, 2), 0), 2).reshape(1, len(methods))
    total_std = np.round(np.average(np.std(results, 2), 0), 2).reshape(1, len(methods))

    ave = np.round(np.average(results, 2), 2)
    std = np.round(np.std(results, 2), 2)

    # Appending the Average row
    ave = np.concatenate((ave, total_average), axis=0)
    std = np.concatenate((std, total_std), axis=0)
    datasets.append('Average')
    wpath = 'Results/excelFile.xlsx'
    if path:
        wpath = path
    workbook = xlsxwriter.Workbook(wpath)

    # Write Accuracy Sheet
    worksheet = workbook.add_worksheet('Accuracy')
    worksheet.write_row(0,1,methods)
    worksheet.write_column(1,0,datasets)
    col = 0
    for row, data in enumerate(ave):
        worksheet.write_row(row+1, col+1, data)

    # Write STD Sheet
    worksheet = workbook.add_worksheet('STD')
    worksheet.write_row(0,1,methods)
    worksheet.write_column(1,0,datasets)
    col = 0
    for row, data in enumerate(std):
        worksheet.write_row(row+1, col+1, data)


    workbook.close()



def save_elements(foldername,datasetName,itr,state,pool_classifiers,X_train,y_train,X_test,y_test,X_DSEL,y_DSEL,list_ds,methods_names):

    rng = np.random.RandomState(state)
    path =  foldername + "/" + datasetName + str(itr) + "-RState-" + np.str(state) + ".p"
    poolspec = open(path, mode="wb")
    pickle.dump(state, poolspec)
    pickle.dump(pool_classifiers, poolspec)
    pickle.dump(X_train, poolspec)
    pickle.dump(y_train, poolspec)
    pickle.dump(X_test, poolspec)
    pickle.dump(y_test, poolspec)
    pickle.dump(X_DSEL, poolspec)
    pickle.dump(y_DSEL, poolspec)
    pickle.dump(list_ds, poolspec)
    pickle.dump(methods_names, poolspec)
def load_elements(foldername,datasetName,itr,state):
    filepath = foldername + "/" + datasetName + str(itr) + "-RState-" + np.str(state) + ".p"
    poolspec = open(filepath, "rb")
    state = pickle.load(poolspec)
    pool_classifiers = pickle.load(poolspec)
    X_train = pickle.load(poolspec)
    y_train = pickle.load(poolspec)
    X_test = pickle.load(poolspec)
    y_test = pickle.load(poolspec)
    X_DSEL = pickle.load(poolspec)
    y_DSEL = pickle.load(poolspec)
    list_ds = pickle.load(poolspec)
    methods_names = pickle.load(poolspec)
    return pool_classifiers,X_train,y_train,X_test,y_test,X_DSEL,y_DSEL,list_ds,methods_names


def plot_multi_curve(results,methods_name,y_legend,x_range):
    [no_pc , no_methods] = results.shape

    for method_counter in range(no_methods):
        plt.plot(x_range, results[:,method_counter], marker='o',label = methods_name[method_counter])
    # Labeling the X-axis
    plt.xlabel('Pool Size')
    # Labeling the Y-axis
    plt.ylabel(y_legend)
    # Give a title to the graph
    #plt.title('Influence of pool size on performance')
    plt.legend()
    plt.show()

def plot_CD(techniques, avranks,no_datasets):

    cd = Orange.evaluation.compute_CD(avranks, no_datasets)
    Orange.evaluation.graph_ranks(avranks, techniques, cd=cd, width=6, textspace=1.5)
    plt.show()

def plot_winloss(techniques, win,tie,loss,nc,without_tie=True):
    if without_tie:
        win += tie / 2
        loss += tie / 2
        tie = 0
    ind = np.arange(len(techniques))  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    p1 = ax.bar(ind, win, width, label='Win')
    p2 = ax.bar(ind, tie, width, bottom=win, label='Tie')
    p3 = ax.bar(ind, loss, width, bottom=win + tie, label='Loss')

    ax.axhline(nc, color='blue', linewidth=1.8)
    ax.set_ylabel('# Datasets')
    ax.set_title('Win-Tie-Loss')
    ax.set_xticks(ind)
    ax.set_xticklabels((techniques))
    ax.legend()

    # Label with label_type 'center' instead of the default 'edge'
    ax.bar_label(p1, label_type='center')
    ax.bar_label(p3, label_type='center')

    plt.show()

def print_Acc_STD_onedataset(datasetName,methods_name,result):
    print("Results for Dataset " + datasetName)
    ave_acc = np.average(result, 1)
    std_acc = np.std(result, 1)
    for na, ac in zip(methods_name, ave_acc):
        print("Accuracy {} = {}"
              .format(na, ac))
    for na, std in zip(methods_name, std_acc):
        print("STD {} = {}"
              .format(na, std))

def plot_MinMaxAve_onedataset(datasetName,methods_name,result):
    """ Plot Accuracy + STD for one dataset"""
    ave_acc = np.average(result, 1)
    min_acc = np.min(result, 1)
    max_acc = np.max(result, 1)
    std_acc = np.std(result, 1)
    fig, ax = plt.subplots()
    ax.plot(methods_name, ave_acc, label="Average")
    ax.plot(methods_name, min_acc, label="Min")
    ax.plot(methods_name, max_acc, label="Max")

    plt.xticks(methods_name)
    print("Results for "+datasetName)

    minlim = np.min(min_acc) - 1
    maxlim = np.max(max_acc) + 1
    ax.set_ylim(minlim, maxlim)
    ax.set_xlabel('DS Method', fontsize=13)
    ax.set_ylabel('Accuracy on the test set (%)', fontsize=13)
    ax.legend(loc='lower right')
    plt.show()

def plot_Acc_STD(results, dataset_names, hyperparameter_range):
    """ result [dataset, parameter_value, itr] """
    NO_datasets, NO_ranges, no_itr = results.shape
    ave = np.round(np.average(results,2),2)
    std = np.round(np.std(results,2),1)

    for dataset in dataset_names:
        accuracy = []
        STD = []
        mu_list = []
        for i in hyperparameter_range:
            mu_list.append(i/100)
            acc,std = results(X,y,i/100)
            print("Mu=" ,i/100, "\t accuracy=",np.round(acc,2) ,"\t std=",np.round(std,1))
            STD.append(std)
            accuracy.append(acc)
        ax.errorbar(mu_list, accuracy, STD, fmt='-o',uplims=True, lolims=True, label=dsname)

def write_in_latex_table(results, dataset_names, method_names, rows="datasets"):

    if rows == "datasets":
        NO_datasets, NO_methods, no_itr = results.shape
        datasets = dataset_names.copy()
        datasets.append("Average")
        dic = {}
        dic['DataSets'] = list(datasets)
        total_average = np.round( np.average(np.average(results,2),0),2).reshape(1,len(method_names))
        total_std = np.round( np.average(np.std(results,2),0),2).reshape(1,len(method_names))

        ave = np.round(np.average(results, 2), 2)
        std = np.round(np.std(results, 2), 2)
        ave = np.concatenate((ave, total_average), axis=0)
        std = np.concatenate((std, total_std), axis=0)


        sha = ave.shape
        ave = ave.flatten().astype(str)
        std = std.flatten().astype(str)
        Tval = [ac + '(' + st + ')' for ac, st in zip(ave, std)]
        Tval = np.array(Tval)
        Tval = Tval.reshape(sha)

        dic.update({na: list(val) for na, val in zip(method_names, Tval.transpose())})
        df = pd.DataFrame(dic)
        print(df.to_latex(index=False))

        av_acc = np.round(np.average(results, (0,2)),2)
        print(av_acc)

        ## Average STD:
        #av_std = np.round(np.average(np.std(results, 2),0),2)
        #for (aav, ast) in zip (av_acc,av_std):
        #    print(aav,"(",ast,"),")

    else:
        NO_datasets ,NO_methods, no_itr =results.shape
        dic={}
        dic['Methods'] = list(method_names)

        ave = np.round(np.average(results,2),2)
        std = np.round(np.std(results,2),2)

        sha=ave.shape
        ave= ave.flatten().astype(str)
        std=std.flatten().astype(str)

        Tval = [ac+'('+st+')' for ac,st in zip(ave,std)]
        Tval = np.array(Tval)
        Tval = Tval.reshape(sha)

        dic.update({na:list(val) for na, val in zip(dataset_names,Tval)})
        df = pd.DataFrame(dic)
        print(df.to_latex(index=False))

def plot_overallresult(average_accuracy,methods_name):
    cmap = get_cmap('Dark2')
    colors = [cmap(i) for i in np.linspace(0, 1, 12)]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    pct_formatter = FuncFormatter(lambda x, pos: '{:.1f}'.format(x * 100))
    ax.bar(np.arange(len(methods_name)),average_accuracy,color=colors,tick_label=methods_name, edgecolor='k')
    minlim = np.min(average_accuracy) - 2
    maxlim = np.max(average_accuracy) + 2

    ax.set_ylim(minlim, maxlim)
    ax.set_ylabel('Accuracy on the test set (%)', fontsize=13)
    ax.yaxis.set_major_formatter(pct_formatter)
    for tick in ax.get_xticklabels():
      tick.set_rotation(60)
    plt.subplots_adjust(bottom=0.18)

    for item in ax.get_xticklabels(): item.set_rotation(90)
    i = 0
    for v in np.round(average_accuracy,2):
        ax.text(i, v, "{:,}".format(v), va='bottom',ha='center')
        i += 1
    plt.tight_layout()
    plt.show()
    for na,ac in zip(methods_name,average_accuracy):
        print("Overall Accuracy {} = {}".format(na, ac))

def read_whole_results(path):
    # path = "Results-DGA1033/With and without Callibration.p"
    rfile = open(path, mode="rb")
    results=pickle.load(rfile)
    datasets = pickle.load(rfile)
    methods_names = pickle.load(rfile)
    rfile.close()
    datasets_names = datasets[0:-1]
    return results,datasets_names,methods_names




#[results,datasets,methods_names] = read_whole_results()
#write_in_latex_table(results,datasets,methods_names,rows="datasets")
#write_whole_results_into_excel(results,datasets,methods_names)