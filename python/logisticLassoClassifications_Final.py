import sys
import numpy as np
import pandas as pd
from itertools import compress
import copy
import matplotlib.pyplot as plt
from featureSelect_correlation import featureSelect_correlation
from pyirr import intraclass_correlation
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold, RepeatedStratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, confusion_matrix
from scipy.stats import spearmanr

# use all the processors unless we are in debug mode
n_jobs = -1
if getattr(sys, 'gettrace', None)():
    n_jobs = 1

def loadTwoReadersData():

    # read spreadsheet
    df = pd.read_excel('/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/GG_MG.xlsx', sheet_name='GG_MG')

    # remove features, as with the discovery/test data
    df.drop(['IndexLesion_GG', 'IndexLesionMG', 'GlobalStageGG', 'GlobalStageMG'], axis=1, inplace=True)

    # remove rows with missing data - need to check that this leaves the same patients for dfGG as in the discovery data set
    df.dropna(inplace=True)

    # split to each reader
    dfGG = df.filter(regex = 'GG|PID', axis = 1)
    dfMG = df.filter(regex='MG|PID', axis=1)

    # match column names by removing subscripts
    dfGG = dfGG.rename(columns=lambda x: x.replace('_GG','').replace('GG',''))
    dfMG = dfMG.rename(columns=lambda x: x.replace('_MG','').replace('MG',''))

    # change some column names to match the discovery/test data sets
    renameDict = {'LocIndexL':'AnatDev01',
                  'LocAnat':'AnatDev02',
                  'Division':'AnatDev03',
                  'DivisionLat':'AnatDev04',
                  'LesionSize':'MajorLengthIndex',
                  'SmoothCapsularBulgin':'SmoothCapsularBulging',
                  'UnsharpMargins':'UnsharpMargin',
                  'irregularContour':'IrregularContour',
                  'BlackEstrition':'BlackEstritionPeripFat',
                  'measurableECE':'MeasurableECE',
                  'retroprostaticAngleObl':'RetroprostaticAngleOblit'}
    dfGG.rename(renameDict, axis=1, inplace=True)
    dfMG.rename(renameDict, axis=1, inplace=True)

    # highsignalT1FS is missing from this spreadsheet, so fill in with default value.
    # Fortunately, this feature is not selected in the final model, but we need it there for compatibility.
    dfGG.loc[:, 'highsignalT1FS'] = 0
    dfMG.loc[:, 'highsignalT1FS'] = 0

    iccDict = {}
    for col in dfGG.drop(['PID', 'highsignalT1FS'], axis=1):
        data = np.stack((dfGG[col], dfMG[col]), axis=1)
        iccDict[col] = intraclass_correlation(data, "twoway", "agreement").value

    return dfGG, dfMG, iccDict

def fitTrainingData(XTrain, yTrain, featureNames, iccDict, crossValidateFit=True, printResubtitutionMetrics=False):
    # reproducible execution
    seed = 42
    np.random.seed(seed)

    # logistic LASSO tuning parameter optimised using function with in-built CV
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('logistic',
                                                              LogisticRegressionCV(Cs=20, cv=10, solver="liblinear",
                                                                                   max_iter=10000, penalty='l1',
                                                                                   random_state=seed))])

    # check for data leak
    if 'yTest' in locals() or 'yTest' in globals():
        print('Test data is accessible to training function - check for data leak!!!')
        return None

    # fit to all data
    pipeline.fit(XTrain, yTrain)

    # print some performance metrics
    if printResubtitutionMetrics:
        y_pred_score = pipeline.predict_proba(XTrain)[:, 1]
        y_pred_class = pipeline.predict(XTrain)
        resubAUROC = roc_auc_score(yTrain, y_pred_score)
        resubAccuracy = accuracy_score(yTrain, y_pred_class)
        resubF1 = f1_score(yTrain, y_pred_class)

        print('AUCROC  (resub) = ' + str(np.round(resubAUROC, 3)))
        print('Accuracy (resub) = ' + str(np.round(resubAccuracy, 3)))
        print('F1 (resub) = ' + str(np.round(resubF1, 3)))
        print(' ')

    if crossValidateFit:

        # cross-validate
        n_repeats = 10
        n_splits = 10
        outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
        cv_result = cross_validate(pipeline, X=XTrain, y=yTrain, cv=outer_cv, scoring=['accuracy', 'roc_auc', 'f1'],
                                   return_estimator=True, verbose=0, n_jobs=n_jobs)

        # get frequency that features are non-zero across the repeated cv splits
        coef_cv = np.zeros((len(cv_result['estimator']), XTrain.shape[1]))
        for n, res in enumerate(cv_result['estimator']):
            coef_cv[n, :] = res._final_estimator.coef_
        coef_freq = np.sum(coef_cv != 0, axis=0) / (n_repeats * n_splits)

        # put icc values in array for including in DataFrame
        iccList = []
        for feat in featureNames:
            if feat in iccDict:
                iccList.append(iccDict[feat])
            else:
                iccList.append('-')

        # display sorted coefficients and selection frequency
        coeffs = np.squeeze(pipeline._final_estimator.coef_)
        dfCoefResults = pd.DataFrame({'Feature': featureNames, 'Coefficient': coeffs, 'Selection frequency': coef_freq, 'ICC':iccList})
        dfCoefResults.sort_values(by='Coefficient', key=abs, inplace=True, ascending=False)
        print('\n')
        print(dfCoefResults.to_string(index=False))
        print('\n')


        # print CV scores
        print('AUCROC   (CV) = ' + str(np.mean(cv_result['test_roc_auc']).round(3)))
        print('Accuracy (CV) = ' + str(np.mean(cv_result['test_accuracy']).round(3)))
        print('F1       (CV) = ' + str(np.mean(cv_result['test_f1']).round(3)))

        # permutation testing
        outer_cv.n_repeats = 1
        n_permutations = 10
        scoreDirect, perm_scores, pValueDirect = permutation_test_score(pipeline, XTrain, yTrain, scoring="roc_auc",
                                                                        cv=outer_cv, n_permutations=n_permutations,
                                                                        verbose=0, n_jobs=n_jobs)

        # pValueDirect is computed using scoreDirect and assumes only one outer CV run
        # We have used repeated outer CV, so the following code correctly computes the p-value of our repeated CV performance estimate
        # Actually, it doesn't seem to make much difference, so am relaxed about that.

        p_values = []
        scores_roc_auc = np.mean(np.reshape(cv_result['test_roc_auc'], (n_repeats, -1)), axis=1)
        for score in scores_roc_auc:
            p_values.append((np.count_nonzero(perm_scores >= score) + 1) / (n_permutations + 1))
        print('p-value       = ' + str(np.mean(p_values).round(4)) + '\n\n')

    return pipeline

def load_fitTraining_test():

    # load data
    dfTrain = pd.read_csv('/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/discovery.csv')
    dfTest  = pd.read_csv('/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/external.csv')
    dfGG, dfMG, iccDict = loadTwoReadersData()

    # drop features we are not going to use for classification
    dfTrain.drop(['Gleason biopsy','TumorGradeMRI'], inplace=True, axis=1)
    dfTest.drop(['Gleason biopsy','TumorGradeMRI'], inplace=True, axis=1)

    # merge with clinical features from training data
    featuresFromTrainingData = ['PID', 'GleasonBinary', 'ProstateVolume', 'PSA', 'IndLesPIRADS_V2', 'ECE_Pathology']
    dfGG = dfGG.merge(dfTrain[featuresFromTrainingData], on='PID')
    dfMG = dfMG.merge(dfTrain[featuresFromTrainingData], on='PID')

    # make sure columns are ordered the same
    dfGG = dfGG[dfTrain.columns]
    dfMG = dfMG[dfTrain.columns]

    # make these features binary 0/1
    toBinary = ['SmoothCapsularBulging' ,'CapsularDisruption', 'UnsharpMargin', 'IrregularContour', 'BlackEstritionPeripFat', 'MeasurableECE', 'RetroprostaticAngleOblit', 'highsignalT1FS']
    for tb in toBinary:
        dfTrain[tb]  = dfTrain[tb].map(dict(YES=1, NO=0))
        dfTest[tb] = dfTest[tb].map(dict(YES=1, NO=0))

    # is missing in test and training, so replace both with median from the training data
    psaTrainMedian = np.nanmedian(np.array(dfTrain.PSA))
    dfTrain.PSA.fillna(psaTrainMedian, inplace=True)
    dfTest.PSA.fillna(psaTrainMedian, inplace=True)

    # this feature is not selected in the semantic model, so this has no effect
    # fill in with the most common value
    dfTest.highsignalT1FS.fillna(0, inplace=True)

    # extract data into numpy arrays, but keep the feature names
    yTrain = np.array(dfTrain.ECE_Pathology)
    yTest = np.array(dfTest.ECE_Pathology)
    yMG_GG = np.array(dfGG.ECE_Pathology)

    XTrain = dfTrain.drop(['PID', 'ECE_Pathology'], axis=1)
    XTest = dfTest.drop(['PID', 'ECE_Pathology'], axis=1)
    X_GG = dfGG.drop(['PID', 'ECE_Pathology'], axis=1)
    X_MG = dfMG.drop(['PID', 'ECE_Pathology'], axis=1)

    featureNames = list(XTrain.columns)
    XTrain = np.array(XTrain)
    XTest = np.array(XTest)
    X_GG = np.array(X_GG)
    X_MG = np.array(X_MG)

    pipeline = fitTrainingData(XTrain, yTrain, featureNames, iccDict, crossValidateFit=True)

    if pipeline is not None:

        # print the test performance metrics
        y_pred_score_test = pipeline.predict_proba(XTest)[:, 1]
        y_pred_class_test = pipeline.predict(XTest)
        testAUROC = roc_auc_score(yTest, y_pred_score_test)
        testAccuracy = accuracy_score(yTest, y_pred_class_test)
        testF1 = f1_score(yTest, y_pred_class_test)

        print('AUCROC  (test)  = ' + str(np.round(testAUROC,3)))
        print('Accuracy (test) = ' + str(np.round(testAccuracy,3)))
        print('F1 (test)       = ' + str(np.round(testF1,3)))

        # re-fit using features selected based on the frequency the logisticLASSO selects each feature being > 0.9
        XTrain_freqSel = dfTrain[['GleasonBinary', 'MeasurableECE', 'CapsularContactLength', 'IrregularContour', 'CapsularDisruption']]
        XTest_freqSel = dfTest[['GleasonBinary', 'MeasurableECE', 'CapsularContactLength', 'IrregularContour', 'CapsularDisruption']]
        pipeline_freqSel = fitTrainingData(XTrain_freqSel, yTrain, featureNames, iccDict, crossValidateFit=False)

        # print the test performance metrics
        y_pred_score_test_freqSel = pipeline_freqSel.predict_proba(XTest_freqSel)[:, 1]
        y_pred_class_test_freqSel = pipeline_freqSel.predict(XTest_freqSel)
        test_freqSel_AUROC = roc_auc_score(yTest, y_pred_score_test_freqSel)
        test_freqSel_Accuracy = accuracy_score(yTest, y_pred_class_test_freqSel)
        test_freqSel_F1 = f1_score(yTest, y_pred_class_test_freqSel)

        print('\nModel using only: GleasonBinary, MeasurableECE, CapsularContactLength, IrregularContour, CapsularDisruption')
        print('AUCROC   = ' + str(np.round(test_freqSel_AUROC,3)))
        print('Accuracy = ' + str(np.round(test_freqSel_Accuracy,3)))
        print('F1       = ' + str(np.round(test_freqSel_F1,3)))
        print(' ')

        # get scores for GG and MG
        y_pred_score_GG = pipeline.predict_proba(X_GG)[:, 1]
        y_pred_score_MG = pipeline.predict_proba(X_MG)[:, 1]

        data = np.stack((y_pred_score_MG, y_pred_score_GG), axis=1)
        iccScore = intraclass_correlation(data, "twoway", "agreement").value
        print('ICC comparing GG and MG scores  = ' + str(np.round(iccScore,3)) + '\n')

        # plot comparing scores
        plt.scatter(y_pred_score_GG, y_pred_score_MG, c=yMG_GG)
        plt.show()

        # plot comparing ROCs
        fprGG, tprGG, _ = roc_curve(yMG_GG, y_pred_score_GG)
        fprMG, tprMG, _ = roc_curve(yMG_GG, y_pred_score_MG)
        fprTest, tprTest, _ = roc_curve(yTest, y_pred_score_test)
        plt.plot(fprGG, tprGG,     label='Train, reader 1, AUROC = ' + str(np.round(roc_auc_score(yMG_GG, y_pred_score_GG),3)))
        plt.plot(fprMG, tprMG,     label='Train, reader 2, AUROC = ' + str(np.round(roc_auc_score(yMG_GG, y_pred_score_MG),3)))
        plt.plot(fprTest, tprTest, label='Test,  reader 1, AUROC = ' + str(np.round(roc_auc_score(yTest, y_pred_score_test),3)))
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROCs')
        plt.legend()
        plt.show()

        def roc_curve_thresholds(yTrue, yScore, thresholds):
            tnArr, fpArr, fnArr, tpArr = [], [], [], []
            nSamples = len(yTrue)
            for thresh in thresholds:
                tn, fp, fn, tp = confusion_matrix(yTrue, yScore>thresh).ravel()
                tnArr.append(tn/nSamples)
                fpArr.append(fp/nSamples)
                fnArr.append(fn/nSamples)
                tpArr.append(tp/nSamples)
            return np.array(tnArr), np.array(fpArr), np.array(fnArr), np.array(tpArr)

        # plots comparing TP, FP, Sensitivity etc.

        thresh = np.linspace(0, 1, 500)
        tnTest, fpTest, fnTest, tpTest = roc_curve_thresholds(yTest, y_pred_score_test, thresh)
        tnGG, fpGG, fnGG, tpGG = roc_curve_thresholds(yMG_GG, y_pred_score_GG, thresh)
        tnMG, fpMG, fnMG, tpMG = roc_curve_thresholds(yMG_GG, y_pred_score_MG, thresh)

        _, ax = plt.subplots(3,2, figsize=(8,12))

        ax[0, 0].plot(thresh, tpGG/(tpGG + fnGG), label='Train, reader 1')
        ax[0, 0].plot(thresh, tpMG/(tpMG + fnMG), label='Train, reader 2')
        ax[0, 0].plot(thresh, tpTest/(tpTest + fnTest), label='Test,  reader 1')
        ax[0, 0].set_title('True positive rate (Sensitivity)')
        ax[0, 0].set_xlabel('Threshold')
        ax[0, 0].set_ylabel('TPR')

        ax[0, 1].plot(thresh, tnGG/(tnGG + fpGG), label='Train, reader 1')
        ax[0, 1].plot(thresh, tnMG/(tnMG + fpMG), label='Train, reader 2')
        ax[0, 1].plot(thresh, tnTest/(tnTest + fpTest), label='Test,  reader 1')
        ax[0, 1].set_title('True negative rate (Specificity)')
        ax[0, 1].set_xlabel('Threshold')
        ax[0, 1].set_ylabel('TNR')
        ax[0, 1].legend()

        ax[1, 0].plot(thresh, fnGG/(fnGG + tpGG))
        ax[1, 0].plot(thresh, fnMG/(fnMG + tpMG))
        ax[1, 0].plot(thresh, fnTest/(fnTest + tpTest))
        ax[1, 0].set_title('False negative rate')
        ax[1, 0].set_xlabel('Threshold')
        ax[1, 0].set_ylabel('FNR')

        ax[1, 1].plot(thresh, fpGG/(fpGG + tnGG))
        ax[1, 1].plot(thresh, fpMG/(fpMG + tnMG))
        ax[1, 1].plot(thresh, fpTest/(fpTest + tnTest))
        ax[1, 1].set_title('False positive rate')
        ax[1, 1].set_xlabel('Threshold')
        ax[1, 1].set_ylabel('FPR')

        ax[2, 0].plot(thresh, (tnGG + tpGG)/(tpGG + fnGG + tnGG + fpGG))
        ax[2, 0].plot(thresh, (tnMG + tpMG)/(tpMG + fnMG + tnMG + fpMG))
        ax[2, 0].plot(thresh, (tnTest + tpTest)/(tpTest + fnTest + tnTest + fpTest))
        ax[2, 0].set_title('Accuracy')
        ax[2, 0].set_xlabel('Threshold')
        ax[2, 0].set_ylabel('Accuracy')

        ax[2, 1].plot(thresh, 2*tpGG/(2*tpGG + fnGG + fpGG))
        ax[2, 1].plot(thresh, 2*tpMG/(2*tpMG + fnMG + fpMG))
        ax[2, 1].plot(thresh, 2*tpTest/(2*tpTest + fnTest + fpTest))
        ax[2, 1].set_title('F1')
        ax[2, 1].set_xlabel('Threshold')
        ax[2, 1].set_ylabel('F1')

        plt.show()

        _, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].plot(thresh, tpGG/(tpGG + fnGG), label='Train, reader 1')
        ax[0].plot(thresh, tpMG/(tpMG + fnMG), label='Train, reader 2')
        ax[0].plot(thresh, tpTest/(tpTest + fnTest), label='Test,  reader 1')
        ax[0].set_title('True positive rate (Sensitivity)')
        ax[0].set_xlim([0, 0.5])
        ax[0].set_ylim([0.5, 1.05])
        ax[0].legend()
        ax[0].set_xlabel('Threshold')
        ax[0].set_ylabel('TPR')

        ax[1].plot(thresh, fnGG/(fnGG + tpGG), label='Train, reader 1')
        ax[1].plot(thresh, fnMG/(fnMG + tpMG), label='Train, reader 2')
        ax[1].plot(thresh, fnTest/(fnTest + tpTest), label='Test,  reader 1')
        ax[1].set_title('False negative rate')
        ax[1].set_xlim([-0.05, 0.5])
        ax[1].set_ylim([-0.05, 0.5])
        ax[1].legend()
        ax[1].set_xlabel('Threshold')
        ax[1].set_ylabel('FNR')

        plt.show()

        _, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].plot(thresh, tpGG, label='Train, reader 1')
        ax[0].plot(thresh, tpMG, label='Train, reader 2')
        ax[0].plot(thresh, tpTest, label='Test,  reader 1')
        ax[0].set_xlim([0, 0.5])
        ax[0].legend()
        ax[0].set_xlabel('Threshold')
        ax[0].set_ylabel('TP')
        ax[0].set_title('True positives')

        ax[1].plot(thresh, fnGG, label='Train, reader 1')
        ax[1].plot(thresh, fnMG, label='Train, reader 2')
        ax[1].plot(thresh, fnTest, label='Test,  reader 1')
        ax[1].set_title('False negatives')
        ax[1].set_xlim([0, 0.5])
        #ax[1].set_ylim([0, 0.5])
        ax[1].legend()
        ax[0].set_xlabel('Threshold')
        ax[1].set_ylabel('FN')

        plt.show()


# Execution is split into two functions: load_fitTraining_test() and fitTrainingData()
# This is so that the scope of the test data should only include fitTrainingData()
# We include a test inside fitTrainingData() to check that the test data are not accesible to this function, which ensures no data leakage

load_fitTraining_test()





