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

    return df


def printResultsForFit(pipeline, X, y):

    pipelineFitToAllData = copy.deepcopy(pipeline)
    # get scores for fit to whole data set
    resubScores = {}
    for key in scoringList:
        if key == 'roc_auc':
            resubScores[key] = roc_auc_score(y, pipelineFitToAllData.predict_proba(X)[:, 1])
        else:
            resubScores[key] = eval(key + '_score(y, pipelineFitToAllData.predict(X))')

    # print some outputs for the fit to the whole data set

    for key in resubScores:
        print(key + ' (Resub) = ' + str(np.round(resubScores[key], 3)))
    print('\n')

    print('Feature group = ' + pipelineFitToAllData.steps[2][1].best_estimator_.steps[0][1].groupFilter)
    featureNames = X.columns[pipelineFitToAllData.steps[1][1].mask_]
    featureNames = featureNames[pipelineFitToAllData.steps[2][1].best_estimator_.steps[0][1].colMask_]
    classifier = pipelineFitToAllData.steps[2][1].best_estimator_.steps[1][1]
    if hasattr(classifier, 'coef_'):
        LR_coeffs = np.ndarray.flatten(classifier.coef_)
        featureNames = featureNames[LR_coeffs != 0]
        for name, value in zip(featureNames, LR_coeffs[LR_coeffs != 0]):
            print(name + ' = ' + str(value.round(4)))
    else:
        for featureName in featureNames:
            print(featureName)

    print('\n')

if len(sys.argv)==3:
    n_jobs = int(sys.argv[2])
else:
    n_jobs = -1

# use all the processors unless we are in debug mode
if getattr(sys, 'gettrace', None)():
    n_jobs = 1

print(' ')

# # clinical features
# targets.append("Necrosis")
# targets.append("MVI")
# targets.append("Renal_Vein_invasion")
# targets.append("EvoST_Branched")
# targets.append("EvoST_Linear")
# targets.append("Overall_Stage_12_vs_34")
# targets.append("Overall_Stage_123_vs_4")
# targets.append("ITH_Index_Binarised")
# targets.append("WGII_Max_Binarised")
# targets.append("Loss_9p21_3")
# targets.append("Sarcomatoid_change")
# targets.append("EvoST_Punctuated")
# targets.append("EvoST_Unclassifiable")
# targets.append("EvoST_Branched_vs_Punctuated")
# targets.append("WGII_Median_Binarised")
# targets.append("Loss_9p21_3_isClonal")
# targets.append("IVC_invasion")
# targets.append("Loss_14q31_1")
# targets.append("Loss_14q31_1_isClonal")
# targets.append("Loss9p_OR_Loss14q")
# targets.append("BAP1")
# targets.append("BAP1_isClonal")
# targets.append("PBRM1")
# targets.append("PBRM1_isClonal")

scoringList = {'accuracy': 'accuracy',
               'precision': make_scorer(precision_score_noWarn),
               'f1': 'f1',
               'specificity': make_scorer(specificity_score),
               'recall': 'recall',
               'roc_auc': 'roc_auc'}


# os.mkdir(outputPath)
# os.mkdir(os.path.join(outputPath, 'models'))
#
# # copy this file and the data reading function
# shutil.copyfile(__file__, os.path.join(outputPath, os.path.split(__file__)[1]))
# sys.stdout = Logger(os.path.join(outputPath, 'stdout.log'))


dfC = loadClinicalData('/Users/morton/Dicom Files/TracerX/Analysis/SS_200221_Cohort_summary_MOcurated.xlsx')
dfR = loadRadiomicsData('/Users/morton/Dicom Files/TracerX/XNAT_Collaborations_Local/extractions/pythonOutputs/20220127105552__radiomicsFeatures_hiLowEnhancement_glcm0_inf/binWidth_20/radiomicFeatures/radiomics_2p5Dmerged.csv')

target = 'Loss_9p21_3'
df = mergeClinicalAndRadiomics(dfC, dfR, target)

textureStr = 'glcm|gldm|glszm|glrlm|ngtdm' #'^((?!shape).)*$'
# groupSelectorList = ['MeshVolume', 'shape', 'firstorder|histogram', textureStr, 'shape|firstorder|histogram', 'shape|'+textureStr, 'firstorder|histogram|'+textureStr, '']
groupSelectorList = ['']
featureGroupHierarchy = ['shape_MeshVolume', 'shape', 'firstorder', 'glcm_Correlation']

n_repeats = 10
n_permutations = n_repeats
n_splits_outer = 5
n_splits_inner = 3
verbose = 0

inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True)

# Steps in pipeline that need parameter tuning
groupSelect_LR = {'name':'Logistic regression',
                  'pipeline':Pipeline(steps=[
                                             ('groupSelector', featureSelection_groupName()),
                                             ('logisticLASSO', LogisticRegression(solver='liblinear', max_iter=10000, penalty='l1'))
                                           ]),
                  'param_grid':{'groupSelector__groupFilter': groupSelectorList,
                                'logisticLASSO__C': np.logspace(-2, 0, 10)},
                  'scoring':'neg_log_loss'}

groupSelect_SVM = {'name':'SVM',
                   'pipeline':Pipeline(steps=[
                                              ('groupSelector', featureSelection_groupName()),
                                              ('SVM-RBF', SVC(kernel='rbf', probability=True))
                                            ]),
                   'param_grid':{'groupSelector__groupFilter': groupSelectorList,
                                 'SVM-RBF__C': np.logspace(-2,3,6),
                                 'SVM-RBF__gamma': np.logspace(-4,1,6)},
                   'scoring':'accuracy'}

groupSelect_RF = {'name':'RF',
                   'pipeline':Pipeline(steps=[
                                              ('groupSelector', featureSelection_groupName()),
                                              ('RF', RandomForestClassifier())
                                            ]),
                   'param_grid':{'groupSelector__groupFilter': groupSelectorList,
                                 'RF__min_samples_leaf': [4, 6, 8, 12],
                                 'RF__max_depth':np.array(range(1,12))},
                   'scoring':'accuracy'}


# grid search for parameter tuning
for estimator in [groupSelect_LR]: #, groupSelect_RF, groupSelect_SVM]:

    print('__________________')
    print(estimator['name'] + '\n\n')


    CV_estimator = GridSearchCV(estimator=estimator['pipeline'], param_grid=estimator['param_grid'], cv=inner_cv, refit=True, verbose=0, scoring=estimator['scoring'], n_jobs=n_jobs)

    # full pipeline including unsupervised steps followed by steps with parameter tuning
    pipeline = Pipeline(steps=[
                               ('scaler', StandardScalerDf()),
                               ('correlationSelector', featureSelection_correlation(threshold=0.9, exact=True, featureGroupHierarchy=featureGroupHierarchy)),
                               ('CV_estimator', CV_estimator)
                              ])

    np.random.seed(0)

    X = df.drop(target, axis=1)
    y = df[target]

    # fit pipeline to whole data set to get selected feature group and features
    pipeline.fit(X, y)

    printResultsForFit(pipeline, X, y)

    # cross validate the pipeline
    # outer_cv = RepeatedStratifiedKFold(n_splits=n_splits_outer, n_repeats=n_repeats)
    # outer_cv = LeaveOneOut()
    outer_cv = LeavePairOut()
    cv_result = cross_validate(pipeline, X=X, y=y, cv=outer_cv, scoring=scoringList, return_estimator=True, verbose=verbose)

    scores = {}
    for item in scoringList:
        res = cv_result['test_' + item]
        scores[item] = np.mean(np.reshape(res, (res.size, -1)), axis=1)

    for key in scores:
        print(key + ' (CV) = ' + str(np.mean(scores[key]).round(3)))


