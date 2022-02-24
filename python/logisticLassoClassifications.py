import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics/machineLearning')
from loadAndPreProcessRadiomicsFile import loadAndPreProcessRadiomicsFile
from classificationNestedCVpermutationTest import classificationNestedCVpermutationTest

import numpy as np
import pandas as pd
from itertools import compress
import copy

from featureSelect_correlation import featureSelect_correlation

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, RepeatedStratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine

def auc_score(X, y):
    out = np.zeros(X.shape[1])
    for col in range(X.shape[1]):
        out[col] = roc_auc_score(y, X[:, col])
    return out

# use all the processors unless we are in debug mode
n_jobs = -1
if getattr(sys, 'gettrace', None)():
    n_jobs = 1

dataFile = '/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/discovery.csv'
df = pd.read_csv(dataFile)

# separate out the Gleason info, or use GleasonBinary
if False:
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

df.drop('MeasurableECE', inplace=True, axis=1)

# MeasurableECE has good accuracy for positive cases, so just use MeasurableECE to predict positive cases and train a classifier to correct the
# radiologist on the negative cases
# nPositiveMeasurableECE = np.sum(df.MeasurableECE==1)
# accuracyPositiveMeasurableECE = np.sum(df.ECE_Pathology[df.MeasurableECE==1])/nPositiveMeasurableECE
#df = df.loc[df.MeasurableECE==0,:]

#df['MeasurableECE'] = 2*df['MeasurableECE'] - 1

#df.drop('MeasurableECE', inplace=True, axis=1)
# df.drop(['ProstateVolume','PSA','MajorLengthIndex', 'CapsularContactLength'], inplace=True, axis=1)
#df = df[['GleasonBinary','PSA', 'ECE_Pathology']] #'ProstateVolume', 'MajorLengthIndex', 'CapsularContactLength',
df['PSA'] = np.log(df['PSA'])
df['ProstateVolume'] = np.log(df['ProstateVolume'])
df['GleasonBinary'] = 2*df['GleasonBinary'] - 1

df['GleasonBinary_PSA'] = df['GleasonBinary']*df['PSA']
df['GleasonBinary_ProstateVolume'] = df['GleasonBinary']*df['ProstateVolume']
df['GleasonBinary_CapsularContactLength'] = df['GleasonBinary']*df['CapsularContactLength']
df['IrregularContour_CapsularContactLength'] = df['IrregularContour']*df['CapsularContactLength']

y = np.array(df.ECE_Pathology)
X = df.drop('ECE_Pathology', axis=1)
featureNames = list(X.columns)
X = np.array(X)

np.random.seed(1234)

# nSamples = 100
# rho = 0.6
# X0 = np.random.multivariate_normal(np.array([0, 0]), np.array([[1, rho],[rho, 1]]), size=nSamples)
# offset = 1.5
# X1 = np.random.multivariate_normal(np.array([offset, -offset]), np.array([[1, rho],[rho, 1]]), size=nSamples)
#
# X = np.hstack((np.vstack((X0, X1)), np.random.normal(size=(200,15))))
# y = np.concatenate((np.zeros(nSamples), np.ones(nSamples)))
#
# plt.plot(X[y==0,0], X[y==0,1], marker='.', linestyle='none')
# plt.plot(X[y==1,0], X[y==1,1], marker='.', linestyle='none')
# plt.show()

# X = featureSelect_correlation(threshold=0.9).fit_transform(X)
X = StandardScaler().fit_transform(X)

scoring = "roc_auc"
logisticModel = {"model": LogisticRegression(solver="liblinear", max_iter=10000, penalty='l1'),
                 "name": "Logistic",
                 "p_grid": {"C": np.logspace(-2,1,20)},
                 "scoring": "neg_log_loss", "result": {}}

randomForestModel = {"model": RandomForestClassifier(criterion='gini', n_estimators=100, bootstrap=True),
                     "name": "Random Forest",
                     "p_grid": {"max_depth": [2, 4, 6]}, #, 8, 10]},
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

univariateModel = {"model": Pipeline(steps=[('fs_auc', SelectKBest(auc_score, k=1)),
                                            ('dummy', LogisticRegression(solver='lbfgs', C=1000000))]),
                   "name": "UnivariateAUC",
                   "p_grid": {},
                   "scoring": "roc_auc"}


np.random.seed(0)

for estimator in [logisticModel]: #svmModel]: #randomForestModel]: # # # #]: #univariateModel]: #]: #,]: #]: #

    print('________________________________________')
    print('Model = ' + estimator["name"])

    # fit to all data using CV for lasso parameter optimisation
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
    clfAll = GridSearchCV(estimator=estimator["model"], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0, scoring=estimator["scoring"])
    clfAll.fit(X,y)

    if estimator is not univariateModel and estimator is not svmModel:
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
    print(' ')

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True)
    clf = GridSearchCV(estimator=estimator['model'], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0, scoring=estimator["scoring"])

    # Permutation Tests for Studying Classifier Performance Markus Ojala
    # long-hand full permutation testing of input features

    # outer_cv = StratifiedKFold(n_splits=5)
    # scores_permTest = []
    # for nMC in range(0):
    #     Xhere = copy.deepcopy(X)
    #     # Xhere = np.vstack((np.transpose(np.reshape(np.arange(np.sum(y==0)*X.shape[1]), (X.shape[1], np.sum(y==0)))),
    #     #                    np.transpose(np.reshape(-np.arange(np.sum(y == 1) * X.shape[1]), (X.shape[1], np.sum(y == 1))))))
    #     for iFeat in range(X.shape[1]):
    #         for c in [0, 1]:
    #             thisFeat = Xhere[y == c, iFeat]
    #             Xhere[y == c, iFeat] = np.random.permutation(thisFeat)
    #     cv_res = cross_validate(clf, X=Xhere, y=y, cv=outer_cv, scoring=["accuracy", "roc_auc", "f1"], return_estimator=True, verbose=0, n_jobs=n_jobs)
    #     scores_permTest.append(np.mean(cv_res['test_roc_auc']))
    #     #print(str(nMC) + ' ' + str(np.quantile(scores_permTest,[0.05, 0.95])))



    n_repeats = 20
    n_permutations = n_repeats

    outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_repeats)
    cv_result = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring=["accuracy", "roc_auc", "f1"], return_estimator=True, verbose=0, n_jobs=n_jobs)
    # get scores for each repeat, averaged over the CV-folds
    scores_roc_auc = np.mean(np.reshape(cv_result['test_roc_auc'], (n_repeats, -1)), axis=1)
    scores_accuracy = np.mean(np.reshape(cv_result['test_accuracy'], (n_repeats, -1)), axis=1)
    scores_f1 = np.mean(np.reshape(cv_result['test_f1'], (n_repeats, -1)), axis=1)
    scores_accuracy_cv = np.mean(scores_accuracy)

    print('AUCROC   (CV)    = \033[1m' + str(np.mean(scores_roc_auc).round(3)) + '\033[0m') # + ' \u00B1 ' + str(np.std(scores).round(3)))
    print('Accuracy (CV)    = \033[1m' + str(scores_accuracy_cv.round(3)) + '\033[0m') # + ' \u00B1 ' + str(np.std(scores).round(3)))
    print('F1       (CV)    = \033[1m' + str(np.mean(scores_f1).round(3)) + '\033[0m') # + ' \u00B1 ' + str(np.std(scores).round(3)))

    # plt.hist(scores_permTest)
    # plt.plot(np.mean(scores_roc_auc), 0, marker='o')
    # plt.show()

    # if 'MeasurableECE' not in df:
    #     netAccuracy = (len(y)*scores_accuracy_cv + nPositiveMeasurableECE*accuracyPositiveMeasurableECE) / (len(y) + nPositiveMeasurableECE)
    #     print('Net accuracy = ' + str(netAccuracy.round(3)))

    # # permutation test needs to use the same type of splitter as for outer_cv, but only needs to use one repeat
    # outer_cv.n_repeats = 1
    # _, perm_scores, _ = permutation_test_score(clf, X, y, scoring="roc_auc", cv=outer_cv, n_permutations=n_permutations, verbose=0, n_jobs=n_jobs)
    #
    # p_values = []
    # for score in scores_roc_auc:
    #     p_values.append((np.count_nonzero(perm_scores >= score) + 1) / (n_permutations + 1))
    # print('p-value        = \033[1m' + str(np.mean(p_values).round(4)) + '\033[0m') # + ' (' + str(np.quantile(p_values, 0.025).round(4)) + ', ' + str(np.quantile(p_values, 0.975).round(4)) + ')')

