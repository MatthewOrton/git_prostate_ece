import sys, os, traceback, copy
sys.path.append('/data/users/morton/git/icrpythonradiomics/machineLearning')

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
import dill

from featureSelection import *
from LeavePairOutCrossValidation import *
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, RepeatedStratifiedKFold, permutation_test_score, LeaveOneOut
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr, ranksums
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, safe_mask
import joblib


if len(sys.argv)==3:
    n_jobs = int(sys.argv[2])
else:
    n_jobs = -1

# use all the processors unless we are in debug mode
if getattr(sys, 'gettrace', None)():
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

def loadClinicalData(clinicalFile):

    # read clinical data file
    dfC1 = pd.read_excel(clinicalFile, sheet_name='S1B_update', usecols='A:AB', header=2)
    dfC2 = pd.read_excel(clinicalFile, sheet_name='S1A', usecols='A:AJ', header=4)
    dfC = pd.merge(dfC1, dfC2, on="Subject")

    # clinical features to explicit binary
    dfC["Sarcomatoid_change"] = dfC["Sarcomatoid_change"] == 'Y'
    dfC["Necrosis"] = dfC["Necrosis"] == 'Yes'
    dfC["MVI"] = dfC["MVI"] == 'Yes'
    dfC["Renal_Vein_invasion"] = dfC["Renal_Vein_invasion"] == 'Yes'
    dfC["IVC_invasion"] = dfC["IVC_invasion"] == 'Yes'
    dfC["Overall_Stage_12_vs_34"] = dfC["Overall_Stage"] <= 2
    dfC["Overall_Stage_123_vs_4"] = dfC["Overall_Stage"] <= 3
    dfC['Loss9p_OR_Loss14q'] = np.logical_or(dfC["Loss_9p21_3"], dfC["Loss_14q31_1"])

    return dfC

def loadRadiomicsData(fileName):

    dfR = pd.read_csv(fileName)

    # remove unwanted columns
    dfR = dfR.drop(dfR.filter(regex='diagnostics|source|histogram|glcmSeparateDirections').columns, axis=1)

    # remove repro rows
    dfR = dfR.loc[~dfR["StudyPatientName"].str.endswith('_rep'), :]

    # filter the required feaureGroup, and always include the shape features, which are only computed for standard_original
    cols = dfR.columns.str.contains('whole_')
    cols = np.logical_or(cols, dfR.columns.str.startswith('StudyPatientName'))

    dfR.columns = [x.replace('whole_original_', '') for x in dfR.columns]

    return dfR.loc[:, cols]

def mergeClinicalAndRadiomics(dfC, dfR, target):

    df = pd.merge(dfC[["Subject", target]], dfR, left_on="Subject", right_on="StudyPatientName")
    df.drop("Subject", 1, inplace=True)
    df.dropna(inplace=True)
    df.drop("StudyPatientName", 1, inplace=True)

    return df.drop(target, axis=1), df[target]


# here is the main modelling pipeline setup
def makeFitAndCV(X, y, n_splits_inner=3, n_splits_outer=5, n_repeats=1, verbose=0):

    # Dictionaries with various bits and pieces for each classification model type
    # Use a pipeline for the model even though the pipeline only has one element - this is so we get the model name out more easily
    LR_model = LogisticRegression(solver='liblinear', max_iter=10000, penalty='l1')
    LR_param = {'classifier__C': np.logspace(-2, 0, 10),
                'classifier': [LR_model]}

    SVM_model = SVC(kernel='rbf', probability=True)
    SVM_param = {'classifier__C': np.logspace(-2, 3, 6),
                 'classifier__gamma': np.logspace(-4, 1, 6),
                 'classifier': [SVM_model]}

    RF_model = RandomForestClassifier()
    RF_param = {'classifier__min_samples_leaf': [4, 6, 8, 12],
                'classifier__max_depth':np.array(range(1,12)),
                'classifier': [RF_model]}

    # modelSelector fits all models over each models tuning parameter grid and chooses the best model/tuning parameter combination
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True)
    # This GridSearchCV selects the  best model type after each model has been tuned
    modelSelector_pipeline = Pipeline([('classifier', LR_model)])  # !! note that LR_model is a dummy placeholder
    modelSelector_params = [LR_param, SVM_param, RF_param]
    modelSelector = GridSearchCV(modelSelector_pipeline, modelSelector_params, cv=inner_cv, refit=True, verbose=verbose, scoring='roc_auc', n_jobs=1)

    pipeline = Pipeline(steps=[
        ('correlationSelector',
         featureSelection_correlation(threshold=0.9, exact=True, featureGroupHierarchy=featureGroupHierarchy)),
        ('scaler', StandardScaler()),
        # Boruta goes here
        ('modelSelector', modelSelector)
    ]
    )
    featureGroupHierarchy = ['shape_MeshVolume', 'shape', 'firstorder', 'glcm_Correlation']

    # overall pipeline
    pipeline = Pipeline(steps=[
                               ('correlationSelector', featureSelection_correlation(threshold=0.9, exact=True, featureGroupHierarchy=featureGroupHierarchy)),
                               ('scaler', StandardScaler()),
                               # Boruta goes here
                               ('modelSelector', modelSelector)
                               ]
                       )

    # fit pipeline to whole data set
    pipeline.fit(X, y)
    pipeline.predict(X)

    modelDict = {LR_model.__class__:'LR-LASSO', SVM_model.__class__:'SVM-RBF', RF_model.__class__:'Random Forest'}

    # print some results
    bestModel = pipeline.steps[2][1].best_params_['classifier']
    print('\nBest model overall: ' + modelDict[bestModel.__class__])

    # Cross validate the whole pipeline
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits_outer, n_repeats=n_repeats)
    # outer_cv = LeavePairOut()
    cv_result = cross_validate(pipeline, X=X, y=y, cv=outer_cv, scoring=scoringList, return_estimator=True, verbose=verbose, n_jobs=-1)

    return pipeline, cv_result, modelDict



scoringList = {'roc_auc': 'roc_auc',
               'accuracy': 'accuracy',
               'precision': make_scorer(precision_score_noWarn),
               'f1': 'f1',
               'specificity': make_scorer(specificity_score),
               'recall': 'recall'}

dfC = loadClinicalData('/Users/morton/Dicom Files/TracerX/Analysis/SS_200221_Cohort_summary_MOcurated.xlsx')
dfR = loadRadiomicsData('/Users/morton/Dicom Files/TracerX/XNAT_Collaborations_Local/extractions/pythonOutputs/20220127105552__radiomicsFeatures_hiLowEnhancement_glcm0_inf/binWidth_20/radiomicFeatures/radiomics_2p5Dmerged.csv')

X, y = mergeClinicalAndRadiomics(dfC, dfR, target='Loss_9p21_3')

np.random.seed(0)

# fit and Cross validate the pipeline
pipeline, cv_result, modelDict = makeFitAndCV(X, y, n_splits_inner=2, n_splits_outer=2, n_repeats=1)

# print some results
bestModel = pipeline.steps[2][1].best_params_['classifier']
print('\nBest model overall: ' + modelDict[bestModel.__class__])

# Cross validated scores
print('\nCross validated scores:\n')
for score in scoringList:
    print(score.ljust(12) + ' = ' + str(round(np.mean(cv_result['test_' + score]),4)))

# Examine the models selected in each outer CV split
modelCount = {}
for est in cv_result['estimator']:
    thisModel = modelDict[est.steps[2][1].best_params_['classifier'].__class__]
    if thisModel in modelCount:
        modelCount[thisModel] += 1
    else:
        modelCount[thisModel] = 1
print('\nModel selection frequency')
print(modelCount)