import sys, os, traceback, copy
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics/machineLearning')
sys.path.append('/Users/morton/Documents/GitHub/prostate_ece_semantic/python')

import numpy as np
import pandas as pd
from pyirr import intraclass_correlation
from itertools import compress
import csv
import uuid
import shutil
from time import strftime, localtime
from joblib import dump
import collections
from warnings import warn
import pickle
# import dill
import time
from joblib import Memory

from featureSelection import *
from boruta_wrapper import BorutaPy4CV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, RepeatedStratifiedKFold, learning_curve, permutation_test_score, LeaveOneOut
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr, ranksums
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, safe_mask
import joblib
import matplotlib.pyplot as plt

start_time = time.time()

if len(sys.argv)==3:
    n_jobs = int(sys.argv[2])
else:
    n_jobs = -1

# use all the processors unless we are in debug mode
if getattr(sys, 'gettrace', None)():
    n_jobs = 1

n_jobs = 1

# this score is (weirdly) not available in sklearn
def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

# some precision calculations give divide by zero warning, so hard code zero_division=0 into this function
def precision_score_noWarn(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def appendRow(row):
    file = open(os.path.join(outputPath, 'results.csv'), 'a') #, newline='')
    cw = csv.writer(file, delimiter=',')
    cw.writerow(row)
    file.close()

def auc_score(X, y):
    out = np.zeros(X.shape[1])
    for col in range(X.shape[1]):
        out[col] = roc_auc_score(y, X[:, col])
    return out

# quick and dirty way to mirror any print statements to a log file
class Logger(object):
    def __init__(self, file):
        self.terminal = sys.stdout
        self.log = open(file, "a")
#
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

#df = pd.read_csv("/data/users/morton/prostateECE/data/discovery.csv")
#zradata = '/data/users/morton/prostateECE/data/radiomicFeatures.csv'
df = pd.read_csv('/Users/morton/Documents/GitHub/prostate_ece_semantic/python/discovery.csv')
zradata = '/Users/morton/Documents/GitHub/prostate_ece_semantic/python/radiomicFeatures.csv'
zrad = pd.read_csv(zradata)

pt = zrad['StudyPatientName']
zrad = zrad.loc[:,zrad.columns.str.startswith('original_')]

df1 = pd.concat([pt,zrad], axis=1)

dfR = df1

dfR = dfR[:168]

dfR = dfR.loc[:,~dfR.columns.str.contains('histogram|original_firstorder_Mean|original_firstorder_Variance|original_shape_Maximum2D',regex=True)]

featureNames = dfR.columns.drop('StudyPatientName')

# Reproducibility
dfRrepM = copy.deepcopy(dfR.loc[dfR["StudyPatientName"].str.endswith('_repro'), :])
dfRrepM["StudyPatientName"] = dfRrepM["StudyPatientName"].str.replace('_repro', '')

dfR = dfR.loc[~dfR["StudyPatientName"].str.endswith('_repro'), :]

dfRrepG = pd.merge(dfR, dfRrepM["StudyPatientName"], on="StudyPatientName")
dfRrepM = pd.merge(dfRrepM, dfRrepG["StudyPatientName"], on="StudyPatientName")

dfRrepM = dfRrepM.sort_values(by="StudyPatientName")
dfRrepG = dfRrepG.sort_values(by="StudyPatientName")

y_rep = pd.merge(df[['PID','ECE_Pathology']], dfRrepM, left_on="PID", right_on="StudyPatientName")['ECE_Pathology']
dfRrepM = pd.merge(df['PID'], dfRrepM, left_on="PID", right_on="StudyPatientName")
dfRrepG = pd.merge(df['PID'], dfRrepG, left_on="PID", right_on="StudyPatientName")

dfRrepM.drop(["StudyPatientName",'PID'], axis=1, inplace=True)
dfRrepG.drop(["StudyPatientName",'PID'], axis=1, inplace=True)

# validation dataset
dfV = copy.deepcopy(df.loc[df["PID"].str.endswith('_v'), :])
dfV["PID"] = dfV["PID"].str.replace('_v', '')
dfRV = pd.merge(dfV[['PID','ECE_Pathology']], dfR, left_on="PID", right_on="StudyPatientName")
dfRV = dfRV.set_index('PID')
yV = dfRV['ECE_Pathology']

# training dateset
df = df.loc[~df["PID"].str.endswith('_v'), :]
dfR = pd.merge(df[['PID','ECE_Pathology']], dfR, left_on="PID", right_on="StudyPatientName")
y = dfR['ECE_Pathology']
dfR = dfR.set_index('PID')

# Correlation threshold
featureGroupHierarchy = ['original_shape_MeshVolume', 'shape', 'original_firstorder', 'glcm_Correlation']
dfR = dfR[featureNames]
#dfR = featureSelection_correlation(featureGroupHierarchy=featureGroupHierarchy, threshold=0.9).fit_transform(dfR)
featureNames = dfR.columns

dfRrepM = dfRrepM[featureNames]
dfRrepG = dfRrepG[featureNames]
dfRV = dfRV[featureNames]

# get ICCs
iccValues = {}
for col in dfRrepM.columns:
    data = np.stack((dfRrepM[col], dfRrepG[col]), axis=1)
    data = data[np.all(np.isfinite(data), axis=1),:]
    iccValues[col] = intraclass_correlation(data, "twoway", "agreement").value
print(iccValues)

iccThreshold = 0.75
print('ICC > ' + str(iccThreshold))

# remove very non-reproducible features
iuse = np.ones(len(dfR.columns)).astype(bool)
for n, feat in enumerate(dfR.columns):
    if feat in iccValues.keys() and iccValues[feat] < iccThreshold:
        iuse[n] = False

repdfR = dfR.loc[:, iuse]
featureNames_rep = repdfR.columns

# use reproducible features only
X = dfR.loc[:, featureNames_rep]
# X = pd.DataFrame(StandardScaler().fit_transform(X))
# X.columns = featureNames_rep
X.columns = [x.replace('original_','') for x in X.columns]

# version of the StandardScaler that outputs a DataFrame if the input is a DataFrame
# class StandardScalerDf(StandardScaler):
#
#     def transform(self, X, copy=None):
#
#         # keep column names if input is DataFrame
#         if isinstance(X, pd.DataFrame):
#             columnNames = X.columns
#             inputIsDataFrame = True
#         else:
#             inputIsDataFrame = False
#
#         X = super(StandardScalerDf, self).transform(X, copy=copy)
#
#         # add column names back if input is DataFrame
#         if inputIsDataFrame:
#             X = pd.DataFrame(X, columns=columnNames)
#
#         return X
#
#     def inverse_transform(self, X, copy=None):
#
#         # keep column names if input is DataFrame
#         if isinstance(X, pd.DataFrame):
#             columnNames = X.columns
#             inputIsDataFrame = True
#         else:
#             inputIsDataFrame = False
#
#         X = super(StandardScalerDf, self).inverse_transform(X, copy=copy)
#
#         # add column names back if input is DataFrame
#         if inputIsDataFrame:
#             X = pd.DataFrame(X, columns=columnNames)
#
#         return X


def makeFitAndCV(X, y, n_splits_inner=5, n_splits_outer=5, n_repeats=10, verbose=0):

    # Dictionaries with various bits and pieces for each classification model type
    # Use a pipeline for the model even though the pipeline only has one element - this is so we get the model name out more easily

    nTune = [[0.1], 4, 4, [4], 2]
    #nTune = [np.logspace(-2, 0, 20), 6, 6, [4, 6, 8, 12], 12]

    LR_model = Pipeline(steps=[('dummy', StandardScaler(with_mean=False, with_std=False)),
                               ('LR', LogisticRegression(solver='liblinear', max_iter=10000, penalty='l1'))])
    LR_param = {'classifier__LR__C': np.logspace(-2, 0, 10),
                'classifier': [LR_model]}

    borutaSeed = 123456
    borutaStep = BorutaPy4CV(n_estimators=10, passThrough=True, perc=100, alpha=0.05, two_step=True, max_iter=100, random_state=borutaSeed, verbose=0, weak=True)
    #rfeStep = RFE(SVC(
    SVM_model = Pipeline(steps=[('Selector', borutaStep),
                                ('SVC', SVC(kernel='rbf'))])# , probability=True)
    SVM_param = {'classifier__SVC__C': np.logspace(-2, 3, 4),
                 'classifier__SVC__gamma': np.logspace(-4, 1, 4),
                 'classifier': [SVM_model]}

    RF_model = Pipeline(steps=[('dummy', StandardScaler(with_mean=False, with_std=False)),
                               ('RF', RandomForestClassifier())]) #max_depth=10, min_samples_leaf=4)
    RF_param = {'classifier__RF__max_features': [4, 6, 8, 10, 12],
                'classifier': [RF_model]}

    # modelSelector fits all models over each models tuning parameter grid and chooses the best model/tuning parameter combination
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True)
    featureGroupHierarchy = ['shape_MeshVolume', 'shape', 'firstorder', 'glcm_Correlation']

    modelSelector_pipeline = Pipeline([('classifier', LR_model)])  # !! note that LR_model is a dummy placeholder
    modelSelector_params = [LR_param, RF_param, SVM_param]
    modelSelector = GridSearchCV(modelSelector_pipeline, modelSelector_params, cv=inner_cv, refit=True, verbose=verbose, scoring='roc_auc', n_jobs=n_jobs)

    allSteps = [('scaler', StandardScalerDf()),
                ('correlationSelector', featureSelection_correlation(threshold=0.9, exact=True, featureGroupHierarchy=featureGroupHierarchy, outputNumpy=True)),
                ('modelSelector', modelSelector),
               ]
    pipeline = Pipeline(steps=allSteps)
    # fit pipeline to whole data set
    pipeline.fit(X, y)

    modelDict = {LR_model.steps[1][1].__class__:'LR-LASSO',
                 SVM_model.steps[1][1].__class__:'SVM-RBF + Boruta',
                 RF_model.steps[1][1].__class__:'Random Forest'}

    # print some results
    bestModel = pipeline.steps[2][1].best_estimator_.steps[0][1].steps[1][1]
    print(modelDict[bestModel.__class__])

    # bestModel_Boruta = not pipeline.steps[2][1].best_params_
    #
    # if bestModel_Boruta is True:
    #     print('\nBest model overall: ' + modelDict[bestModel.__class__] + ' with Boruta')
    # else:
    #     print('\nBest model overall: ' + modelDict[bestModel.__class__] + ' without Boruta')

    if hasattr(pipeline.steps[2][1], "decision_function"):
        y_pred_score = pipeline.steps[2][1].decision_function(pipeline.steps[1][1].fit_transform(pipeline.steps[0][1].fit_transform(X)))
    else:
        y_pred_score = pipeline.steps[2][1].predict_proba(pipeline.steps[1][1].fit_transform(pipeline.steps[0][1].fit_transform(X))[:, 1])

    y_pred_class = pipeline.steps[2][1].predict(pipeline.steps[1][1].fit_transform(pipeline.steps[0][1].fit_transform(X)))

    resubAUROC = roc_auc_score(y, y_pred_score)
    resubAccuracy = accuracy_score(y, y_pred_class)
    resubF1 = f1_score(y, y_pred_class)

    fpr, tpr, thresh = roc_curve(y,y_pred_score)
    plt.plot(fpr,tpr,label="Resub, AUC = "+str(resubAUROC))
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('ROC for Resub')
    plt.legend(loc="lower right")
    plt.show()

    # Cross validate the whole pipeline
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits_outer, n_repeats=n_repeats)
    cv_result = cross_validate(pipeline, X=X, y=y, cv=outer_cv, scoring=scoringList, return_estimator=True, verbose=verbose)

    return pipeline, cv_result, modelDict



scoringList = {'roc_auc': 'roc_auc',
               'accuracy': 'accuracy',
               'precision': make_scorer(precision_score_noWarn),
               'f1': 'f1',
               'specificity': make_scorer(specificity_score),
               'recall': 'recall'}


for borutaPassThrough in [True]:

    np.random.seed(0)

    # fit and Cross validate the pipeline
    pipeline, cv_result, modelDict = makeFitAndCV(X, y, n_splits_inner=10, n_splits_outer=10, n_repeats=10)

    # print some results
    # bestModel = pipeline.steps[3][1].best_params_['classifier']
    bestModel = pipeline.steps[2][1].best_estimator_.steps[0][1].steps[1][1]
    print('\nBest model overall: ' + modelDict[bestModel.__class__])
    # bestModel = pipeline.steps[2][1].best_estimator_.steps[1][1].best_params_['classifier']


    #
    # if bestModel_Boruta is True:
    #     print('\nBest model overall: ' + modelDict[bestModel.__class__] + ' with Boruta')
    # else:
    #     print('\nBest model overall: ' + modelDict[bestModel.__class__] + ' without Boruta')

    # Cross validated scores
    print('\nCross validated scores:\n')
    for score in scoringList:
        print(score.ljust(12) + ' = ' + str(round(np.mean(cv_result['test_' + score]),4)))

    # Examine the models selected in each outer CV split
    modelCount = {}
    # modelCountBT = {}
    # for est in cv_result['estimator']:
    #     thisBoruta = est.steps[2][1].best_params_['borutaSelector__passThrough']
    #     if thisBoruta is True:
    #         thisModelBT = modelDict[est.steps[2][1].best_estimator_.steps[1][1].best_params_['classifier'].__class__]
    #         if thisModelBT in modelCount:
    #             modelCountBT[thisModelBT] += 1
    #         else:
    #             modelCountBT[thisModelBT] = 1
    #     else:
    #         thisModel = modelDict[est.steps[2][1].best_estimator_.steps[1][1].best_params_['classifier'].__class__]
    #         if thisModel in modelCount:
    #             modelCount[thisModel] += 1
    #         else:
    #             modelCount[thisModel] = 1
    # print('\nModel selection frequency')
    # print('Without Boruta')
    # print(modelCountBT)
    # print('With Boruta')
    # print(modelCount)

    for est in cv_result['estimator']:
        thisModel = modelDict[est.steps[2][1].best_estimator_.steps[0][1].steps[1][1].__class__]
        if thisModel in modelCount:
            modelCount[thisModel] += 1
        else:
            modelCount[thisModel] = 1
    print('\nModel selection frequency')
    print(modelCount)

print("--- %s seconds ---" % (time.time() - start_time))
