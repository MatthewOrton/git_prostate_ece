import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics/machineLearning')
from loadAndPreProcessRadiomicsFile import loadAndPreProcessRadiomicsFile
from classificationNestedCVpermutationTest import classificationNestedCVpermutationTest

import numpy as np
import pandas as pd
from itertools import compress

from featureSelect_correlation import featureSelect_correlation

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, RepeatedStratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# use all the processors unless we are in debug mode
n_jobs = -1
if getattr(sys, 'gettrace', None)():
    n_jobs = 1

dataFile = '/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/discovery.csv'
df = pd.read_csv(dataFile)

df['Gleason biopsy 1'] = [int(x[2]) for x in df['Gleason biopsy']]
df['Gleason biopsy 2'] = [int(x[4]) for x in df['Gleason biopsy']]
df.drop('GleasonBinary', inplace=True, axis=1)

df.drop(['PID', 'Gleason biopsy','TumorGradeMRI'], inplace=True, axis=1)
df.SmoothCapsularBulging = df.SmoothCapsularBulging.map(dict(YES=1, NO=0))
df.CapsularDisruption = df.CapsularDisruption.map(dict(YES=1, NO=0))
df.UnsharpMargin = df.UnsharpMargin.map(dict(YES=1, NO=0))
df.IrregularContour = df.IrregularContour.map(dict(YES=1, NO=0))
df.BlackEstritionPeripFat = df.BlackEstritionPeripFat.map(dict(YES=1, NO=0))
df.MeasurableECE = df.MeasurableECE.map(dict(YES=1, NO=0))
df.RetroprostaticAngleOblit = df.RetroprostaticAngleOblit.map(dict(YES=1, NO=0))
df.highsignalT1FS = df.highsignalT1FS.map(dict(YES=1, NO=0))

# one missing value, replace with median
psa = np.array(df.PSA)
psa[np.isnan(psa)] = np.nanmedian(psa)
df.PSA = pd.Series(psa)

# MeasurableECE has good accuracy for positive cases, so just use MeasurableECE to predict positive cases and train a classifier to correct the
# radiologist on the negative cases
# nPositiveMeasurableECE = np.sum(df.MeasurableECE==1)
# accuracyPositiveMeasurableECE = np.sum(df.ECE_Pathology[df.MeasurableECE==1])/nPositiveMeasurableECE
# df = df.loc[df.MeasurableECE==0,:]
# df.drop('MeasurableECE', inplace=True, axis=1)

y = np.array(df.ECE_Pathology)
X = df.drop('ECE_Pathology', axis=1)
featureNames = list(X.columns)

# X = featureSelect_correlation(threshold=0.9).fit_transform(X)
X = StandardScaler().fit_transform(X)

scoring = "roc_auc"
logisticModel = {"model": LogisticRegression(solver="liblinear", max_iter=10000, penalty='l1'),
                 "name": "Logistic",
                 "p_grid": {"C": np.logspace(-4,0,20)},
                 "scoring": "neg_log_loss", "result": {}}

randomForestModel = {"model": RandomForestClassifier(criterion='gini', n_estimators=100, bootstrap=True),
                     "name": "Random Forest",
                     "p_grid": {"max_depth": [4, 8, 10]},
                     "scoring": scoring}

svmModel = {"model": SVC(kernel="rbf"),
            "name": "SVM(rbf)",
            "p_grid": {"C": np.logspace(-2, 3, 5),
                       "gamma": np.logspace(-4, 1, 5)},
            "scoring": scoring}

xgbModel = {"model": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "name": "XGBoost",
            "p_grid": {},
            "scoring": scoring}

np.random.seed(0)

for estimator in [logisticModel, randomForestModel]:

    print('________________________________________')
    print('Model = ' + estimator["name"])

    # fit to all data using CV for lasso parameter optimisation
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
    clfAll = GridSearchCV(estimator=estimator["model"], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0, scoring=estimator["scoring"])
    clfAll.fit(X,y)

    if estimator is logisticModel:
        values = np.squeeze(clfAll.best_estimator_.coef_)
    elif estimator is randomForestModel:
        values = clfAll.best_estimator_.feature_importances_

    idxSort = np.argsort(np.abs(values))[::-1]

    print(' ')
    for n in idxSort:
        print(f"{values[n]:.3f}" + ' ' + featureNames[n])
    print(' ')

    if hasattr(clfAll, "decision_function"):
        y_pred_score = clfAll.decision_function(X)
    else:
        y_pred_score = clfAll.predict_proba(X)[:, 1]

    y_pred_class = clfAll.predict(X)

    resubAUROC = roc_auc_score(y, y_pred_score)
    resubAccuracy = accuracy_score(y, y_pred_class)
    resubF1 = f1_score(y, y_pred_class)


    print('AUCROC  (resub) = ' + str(np.round(resubAUROC,3)))
    print('Accuracy (resub) = ' + str(np.round(resubAccuracy,3)))
    print('F1 (resub) = ' + str(np.round(resubF1,3)))

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True)
    clf = GridSearchCV(estimator=estimator["model"], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0, scoring=estimator["scoring"])

    # Permutation Tests for Studying Classifier Performance Markus Ojala

    n_repeats = 10
    n_permutations = n_repeats

    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats)
    cv_result = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring=["accuracy", "roc_auc", "f1"], return_estimator=True, verbose=0, n_jobs=n_jobs)
    # get scores for each repeat, averaged over the CV-folds
    scores_roc_auc = np.mean(np.reshape(cv_result['test_roc_auc'], (n_repeats, -1)), axis=1)
    scores_accuracy = np.mean(np.reshape(cv_result['test_accuracy'], (n_repeats, -1)), axis=1)
    scores_f1 = np.mean(np.reshape(cv_result['test_f1'], (n_repeats, -1)), axis=1)
    scores_accuracy_cv = np.mean(scores_accuracy)

    print('AUCROC   (CV)    = \033[1m' + str(np.mean(scores_roc_auc).round(3)) + '\033[0m') # + ' \u00B1 ' + str(np.std(scores).round(3)))
    print('Accuracy (CV)    = \033[1m' + str(scores_accuracy_cv.round(3)) + '\033[0m') # + ' \u00B1 ' + str(np.std(scores).round(3)))
    print('F1       (CV)    = \033[1m' + str(np.mean(scores_f1).round(3)) + '\033[0m') # + ' \u00B1 ' + str(np.std(scores).round(3)))

    if 'MeasurableECE' not in df:
        netAccuracy = (len(y)*scores_accuracy_cv + nPositiveMeasurableECE*accuracyPositiveMeasurableECE) / (len(y) + nPositiveMeasurableECE)
        print('Net accuracy = ' + str(netAccuracy.round(3)))

    # permutation test needs to use the same type of splitter as for outer_cv, but only needs to use one repeat
    outer_cv.n_repeats = 1
    _, perm_scores, _ = permutation_test_score(clf, X, y, scoring="roc_auc", cv=outer_cv, n_permutations=n_permutations, verbose=0, n_jobs=n_jobs)

    p_values = []
    for score in scores_roc_auc:
        p_values.append((np.count_nonzero(perm_scores >= score) + 1) / (n_permutations + 1))
    print('p-value        = \033[1m' + str(np.mean(p_values).round(4)) + '\033[0m') # + ' (' + str(np.quantile(p_values, 0.025).round(4)) + ', ' + str(np.quantile(p_values, 0.975).round(4)) + ')')
