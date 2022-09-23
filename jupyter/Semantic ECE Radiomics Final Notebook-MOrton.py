
import pandas as pd
import seaborn as sns
from pandas.api.types import is_string_dtype, is_numeric_dtype
import cufflinks as cf
import numpy as np
from IPython.display import display,HTML
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# In[2]:


import pandas as pd 
data = pd.read_csv('/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/discovery.csv')
#reader01 = pd.read_csv('/Users/npapan/pycaret/MyData/Gisa/Data_Aug2021/Reader01.csv')
#reader02 = pd.read_csv('/Users/npapan/pycaret/MyData/Gisa/Data_Aug2021/Reader02.csv')
test = pd.read_csv('/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/external.csv')


# In[3]:


import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate,KFold
#scoring functions
from sklearn.metrics import recall_score,precision_score,make_scorer
#model example 
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
sensitivity = make_scorer(recall_score, pos_label=1)
specificity = make_scorer(recall_score, pos_label=0)
PPV = make_scorer(precision_score, pos_label=1)
NPV = make_scorer(precision_score, pos_label=0)
score_metrics = {'roc_auc':'roc_auc', 'accuracy':'accuracy', 'bal_acc':'balanced_accuracy', 'sensitivity' : sensitivity, 'specificity': specificity, 'PPV': PPV, 'NPV' : NPV, 'f1':'f1'}



from pycaret.classification import *
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SMOTE, SMOTENC, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
#ru = RandomUnderSampler(random_state=default_setup['session_id'])
exp = setup(data = data, experiment_name = 'Semantic Model ECE', test_data=test,
                   target = 'ECE_Pathology', session_id=124, log_experiment=True, log_data=True, log_profile=False, log_plots=True,
                   normalize = False, normalize_method = 'robust', transformation = False, transformation_method = 'yeo-johnson',
                   data_split_shuffle = True, data_split_stratify=True, categorical_imputation = 'mode', numeric_imputation = 'mean',
                   ignore_features = ['PID'],       
            #ignore_features = ['PID', 'Gleason biopsy','IndLesPIRADS_V2', 'TumorGradeMRI', 'AnatDev01', 'AnatDev02',
             #      'AnatDev03', 'AnatDev04','SmoothCapsularBulging', 'CapsularDisruption', 'UnsharpMargin',
              #     'IrregularContour', 'BlackEstritionPeripFat','RetroprostaticAngleOblit', 'highsignalT1FS'],
                   #categorical_features = ['CapsularDisruption', 'IrregularContour', "measurableECE"],
                   #numeric_features = ['CapsulaContactLength', 'MajLength'],
                   #date_features = ['referenceDate'],
                   remove_outliers = True, outliers_threshold = 0.05,
                   #high_cardinality_features = ['None'], high_cardinality_method = ‘frequency’, 
                   #pca = True, pca_method = 'linear', pca_components = 20,
                   #remove_multicollinearity = True, multicollinearity_threshold = 0.95,
                   #create_clusters = True, cluster_iter = 20,
                   ignore_low_variance = True, 
                   profile = False, silent=True,
                   feature_selection = True, feature_selection_threshold =1.0, feature_selection_method = 'boruta', 
                   #feature_interaction = True, feature_ratio = True, interaction_threshold = 0.01,
                   fix_imbalance = True, #fix_imbalance_method = RandomUnderSampler(sampling_strategy='majority'),
                   verbose = True)


# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
eda(data, target='ECE_Pathology')


# In[ ]:


X_train = get_config('X_train')
Y_train = get_config('y_train')
X_test = get_config('X_test')
Y_test = get_config('y_test')
features = X_train.columns
features


# In[ ]:


#X_train = X_train.astype('int64')
#X_test = X_test.astype('int64')
#Y_train= Y_train.astype('int64')
#Y_test= Y_test.astype('int64')


# In[ ]:





# In[ ]:


X_train.to_csv('X_train_sel.csv')
Y_train.to_csv('Y_train_sel.csv')
X_test.to_csv('X_test_sel.csv')
Y_test.to_csv('Y_test_sel.csv')


# In[ ]:


x_train= X_train.to_numpy()
x_test = X_test.to_numpy()
y_train = Y_train.to_numpy()
y_test = Y_test.to_numpy()


# In[ ]:


Xtest=X_test
ytest=Y_test
Xtrain=X_train
ytrain=Y_train


# In[ ]:


from sklearn.metrics import balanced_accuracy_score
add_metric('Balanced Accuracy', 'Balanced Accuracy', balanced_accuracy_score)


# In[ ]:


compare_models(sort="f1")


# In[ ]:


lr = tune_model(create_model('lr'), optimize='f1', return_train_score=True)


# In[ ]:


# lrc = calibrate_model(lrc)


# # Model Evaluation

# In[ ]:


sns.set_style('whitegrid')
plt.rcParams["figure.figsize"]=5,5
plt.style.use('seaborn')
# get_ipython().run_line_magic('matplotlib', 'inline')
evaluate_model(lr, use_train_data=True)


# In[ ]:


sns.set_style('whitegrid')
plt.rcParams["figure.figsize"]=10,10
plt.style.use('seaborn')
evaluate_model(lr, use_train_data=False)


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import LearningCurve
cv = StratifiedKFold(n_splits=5)
sizes = np.linspace(0.1, 1.0, 100)
visualizer = LearningCurve(
    lr, cv=cv, scoring='f1', train_sizes=sizes, n_jobs=4
)

visualizer.fit(X_train, Y_train)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(lr, random_state=1).fit(X_train, Y_train)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[ ]:


from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix


# In[ ]:


y_pred=pd.DataFrame(lr.predict(X_test),columns=['pred'],index=X_test.index)
CM = confusion_matrix(Y_test, y_pred)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]


# In[ ]:


# Test Set
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
#data=[[TN,FP],[FN,TP]] #Model M1 table
#Chi square statistic,pvalue,DOF,expected table
CM = confusion_matrix(Y_test, y_pred)
stat, p, dof, expected = chi2_contingency(CM) 
print('Chi-square statistic=',stat)
print('Pvalue=',p)
alpha=0.05
if p < alpha:
    print('Not a random guesser')
else:
    print('Model is a random guesser')


# # Dummy Classifier

# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, Y_train)
DummyClassifier(strategy='most_frequent')
print('Dummy accuracy=', dummy_clf.score(X_test, Y_test))
print ('LR accuracy=', lr.score(X_test, Y_test))      


# In[ ]:


from yellowbrick.classifier import ConfusionMatrix
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(dummy_clf, classes=[0,1])

# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(X_train, Y_train)

# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(X_test, Y_test)

# How did we do?
cm.show()


# In[ ]:


from yellowbrick.classifier import ConfusionMatrix
# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(lr, classes=[0,1])

# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(X_train, Y_train)

# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(X_test, Y_test)

# How did we do?
cm.show()


# # CI determination Training
# https://towardsdatascience.com/get-confidence-intervals-for-any-model-performance-metrics-in-machine-learning-f9e72a3becb2

# In[ ]:


import datetime
print(datetime.datetime.now())
import numpy as np
import pandas as pd
import re
from sklearn import metrics, model_selection



hardpredtst=lr.predict(X_train)

def conf_matrix(y,pred):
    ((tn, fp), (fn, tp)) = metrics.confusion_matrix(y, pred)
    ((tnr,fpr),(fnr,tpr))= metrics.confusion_matrix(y, pred, 
            normalize='true')
    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})', 
                                f'FP = {fp} (FPR = {fpr:1.2%})'], 
                         [f'FN = {fn} (FNR = {fnr:1.2%})', 
                                f'TP = {tp} (TPR = {tpr:1.2%})']],
            index=['True 0(No ECE)', 'True 1(ECE)'], 
            columns=['Pred 0(Predicted as No ECE)', 
                            'Pred 1(Predicted as ECE)'])
display(conf_matrix(Y_train,hardpredtst))


# In[ ]:


predtst=lr.predict_proba(X_train)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(Y_train, predtst)
dfplot=pd.DataFrame({'Threshold':thresholds, 
        'False Positive Rate':fpr, 
        'False Negative Rate': 1.-tpr})
ax=dfplot.plot(x='Threshold', y=['False Positive Rate',
        'False Negative Rate'], figsize=(10,6))
ax.plot([0.33,0.33],[0,1]) #mark example thresh.
ax.set_xbound(0,1); ax.set_ybound(0,1) #zoom in


# In[ ]:


from boot_conf_intervals_ml import specificity_score,          make_boot_df, raw_metric_samples, ci


# In[ ]:


hardpredtst_tuned_thresh = np.where(predtst >= 0.55, 1, 0)
conf_matrix(Y_train, hardpredtst_tuned_thresh)


# In[ ]:


# reset random row nums from #train_test_split() #before #concat!
ytrain=Y_train.reset_index(drop=True) #else #concat #ignore_index=True
hardpredtst_tuned_thresh = pd.Series  (hardpredtst_tuned_thresh, name='PredClass') #was #Numpy #array
dforig = pd.concat([ytrain, hardpredtst_tuned_thresh], axis=1)

np.random.seed(13)  # to get same #sample #datasets #every #time
df0, df1, df2 = make_boot_df(dforig), make_boot_df(dforig), make_boot_df(dforig)
pd.concat([dforig,df0,df1,df2], axis=1, keys=['Orig','Boot0','Boot1', 'Boot3'])


# In[ ]:


met=[ metrics.recall_score, specificity_score, 
      metrics.balanced_accuracy_score, metrics.roc_auc_score, metrics.f1_score, 
    ]
np.random.seed(13)
raw_metric_samples(met, ytrain, hardpredtst_tuned_thresh, 
         nboots=10).style.format('{:.2%}')


# In[ ]:


import matplotlib, matplotlib.ticker as mtick
DFLT_QUANTILES=[0.025,0.975]
def metric_boot_histogram( metric, *data_args, quantiles=DFLT_QUANTILES, 
                           nboots=1000, **metric_kwargs
                         ):
    point = metric(*data_args, **metric_kwargs)
    data = raw_metric_samples(metric, *data_args, **metric_kwargs).transpose()
    (lower, upper) = data.quantile(quantiles).iloc[:,0]
    import seaborn; seaborn.set_style('whitegrid')  #optional
    matplotlib.rcParams["figure.dpi"] = 300
    ax = data.hist(bins=50, figsize=(5, 2), alpha=0.4)[0][0]
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    for x in lower, point, upper:
        ax.plot([x, x],[0, 30], lw=2.5)
        


# In[ ]:


np.random.seed(13)
metric_boot_histogram  (metrics.roc_auc_score, ytrain, hardpredtst_tuned_thresh)


# In[ ]:


# Training Data
np.random.seed(13)
ci(met,ytrain, hardpredtst_tuned_thresh, nboots=1000).style.format('{:.2%}')


# # CI Test set

# In[ ]:


import datetime
print(datetime.datetime.now())
import numpy as np
import pandas as pd
import re
from sklearn import metrics, model_selection



hardpredtst=lr.predict(X_test)

def conf_matrix(y,pred):
    ((tn, fp), (fn, tp)) = metrics.confusion_matrix(y, pred)
    ((tnr,fpr),(fnr,tpr))= metrics.confusion_matrix(y, pred, 
            normalize='true')
    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})', 
                                f'FP = {fp} (FPR = {fpr:1.2%})'], 
                         [f'FN = {fn} (FNR = {fnr:1.2%})', 
                                f'TP = {tp} (TPR = {tpr:1.2%})']],
            index=['True 0(No ECE)', 'True 1(ECE)'], 
            columns=['Pred 0(Predicted as No ECE)', 
                            'Pred 1(Predicted as ECE)'])
display(conf_matrix(Y_test,hardpredtst))


# In[ ]:


predtst=lr.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(Y_test, predtst)
dfplot=pd.DataFrame({'Threshold':thresholds, 
        'False Positive Rate':fpr, 
        'False Negative Rate': 1.-tpr})
ax=dfplot.plot(x='Threshold', y=['False Positive Rate',
        'False Negative Rate'], figsize=(10,6))
ax.plot([0.33,0.33],[0,1]) #mark example thresh.
ax.set_xbound(0,1); ax.set_ybound(0,1) #zoom in


# In[ ]:


from boot_conf_intervals_ml import specificity_score,          make_boot_df, raw_metric_samples, ci


# In[ ]:


hardpredtst_tuned_thresh = np.where(predtst >= 0.45, 1, 0)
conf_matrix(Y_test, hardpredtst_tuned_thresh)


# In[ ]:


# reset random row nums from #train_test_split() #before #concat!
ytest=Y_test.reset_index(drop=True) #else #concat #ignore_index=True
hardpredtst_tuned_thresh = pd.Series  (hardpredtst_tuned_thresh, name='PredClass') #was #Numpy #array
dforig = pd.concat([ytest, hardpredtst_tuned_thresh], axis=1)

np.random.seed(13)  # to get same #sample #datasets #every #time
df0, df1, df2 = make_boot_df(dforig), make_boot_df(dforig), make_boot_df(dforig)
pd.concat([dforig,df0,df1,df2], axis=1, keys=['Orig','Boot0','Boot1', 'Boot3'])


# In[ ]:


met=[ metrics.recall_score, specificity_score, 
      metrics.balanced_accuracy_score, metrics.roc_auc_score, metrics.f1_score, 
    ]
np.random.seed(13)
raw_metric_samples(met, ytest, hardpredtst_tuned_thresh, 
         nboots=10).style.format('{:.2%}')


# In[ ]:


import matplotlib, matplotlib.ticker as mtick
DFLT_QUANTILES=[0.025,0.975]
def metric_boot_histogram( metric, *data_args, quantiles=DFLT_QUANTILES, 
                           nboots=1000, **metric_kwargs
                         ):
    point = metric(*data_args, **metric_kwargs)
    data = raw_metric_samples(metric, *data_args, **metric_kwargs).transpose()
    (lower, upper) = data.quantile(quantiles).iloc[:,0]
    import seaborn; seaborn.set_style('whitegrid')  #optional
    matplotlib.rcParams["figure.dpi"] = 300
    ax = data.hist(bins=50, figsize=(5, 2), alpha=0.4)[0][0]
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    for x in lower, point, upper:
        ax.plot([x, x],[0, 30], lw=2.5)
        


# In[ ]:


np.random.seed(13)
metric_boot_histogram  (metrics.roc_auc_score, ytest, hardpredtst_tuned_thresh)


# In[ ]:


# Training Data
np.random.seed(13)
ci(met,ytest, hardpredtst_tuned_thresh, nboots=1000).style.format('{:.2%}')


# ## Explain the Model with SHAP

# In[ ]:


X_train = get_config('X_train')
Y_train = get_config('y_train')
X_test = get_config('X_test')
Y_test = get_config('y_test')
features = X_train.columns
features


# In[ ]:


import xgboost
import shap

# load JS visualization code to notebook
shap.initjs()
y = Y_train.to_numpy()
y


# In[ ]:


# build a Permutation explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Permutation(lr.predict_proba, X_train)
shap_values = explainer(X_test)

# get just the explanations for the positive class
shap_values = shap_values[...,1]


# In[ ]:


shap.plots.bar(shap_values)


# In[ ]:


shap.plots.beeswarm(shap_values)


# In[ ]:


shap.plots.waterfall(shap_values[1])


# # Permutation Score
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_permutation_test_for_classification.html#sphx-glr-auto-examples-feature-selection-plot-permutation-test-for-classification-py

# Next, we calculate the permutation_test_score using the original iris dataset, which strongly predict the labels and the randomly generated features and iris labels, which should have no dependency between features and labels. We use the Best classifier and AUC score to evaluate the model at each round.
# 
# permutation_test_score generates a null distribution by calculating the accuracy of the classifier on 1000 different permutations of the dataset, where features remain the same but labels undergo different permutations. This is the distribution for the null hypothesis which states there is no dependency between the features and labels. An empirical p-value is then calculated as the percentage of permutations for which the score obtained is greater that the score obtained using the original data.

# In[ ]:


from sklearn.model_selection import permutation_test_score


# In[ ]:


import numpy as np

n_uncorrelated_features = 10
rng = np.random.RandomState(seed=0)
# Use same number of samples as in iris and 2200 features
X_rand = rng.normal(size=(X_test.shape[0], n_uncorrelated_features))


# In[ ]:


from sklearn.metrics import fbeta_score, make_scorer, roc_auc_score
ftwo_scorer = make_scorer(fbeta_score, beta=2)
roc_auc_scorer = make_scorer(roc_auc_score)


# In[ ]:


from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(2, shuffle=True, random_state=0)
score_data, perm_scores_data, pvalue = permutation_test_score(
    lr, X_test, Y_test, scoring=roc_auc_scorer, cv=cv, n_permutations=100)
score_rand, perm_scores_rand, pvalue_rand = permutation_test_score(
    lr, X_rand, Y_test, scoring=roc_auc_scorer, cv=cv, n_permutations=100)


# Below we plot a histogram of the permutation scores (the null distribution). The red line indicates the score obtained by the classifier on the original data. The score is much better than those obtained by using permuted data and the p-value is thus very low. This indicates that there is a low likelihood that this good score would be obtained by chance alone. It provides evidence that the iris dataset contains real dependency between features and labels and the classifier was able to utilize this to obtain good results.

# In[ ]:


sns.set_style('darkgrid')
sns.set()
p = sns.histplot(data=perm_scores_data, bins=30)
p.set_xlabel("AUC score", fontsize = 15)
p.set_ylabel("Probability", fontsize = 15)
p.set_title("Target Shuffling", fontsize = 20)
p.axvline(score_data, ls='-', color='r')
score_label = (f"Score on original\ndata: {score_data:.3f}\n"
               f"(p-value: {pvalue:.3f})")
p.text(0, 10, score_label, fontsize=12)
p.set_xlim(0)


# Below we plot the null distribution for the randomized data. The permutation scores are similar to those obtained using the original iris dataset because the permutation always destroys any feature label dependency present. The score obtained on the original randomized data in this case though, is very poor. This results in a large p-value, confirming that there was no feature label dependency in the original data.

# In[ ]:


sns.set()
p = sns.histplot(data=perm_scores_rand, bins=30)
p.set_xlabel("AUC score", fontsize = 15)
p.set_ylabel("Probability", fontsize = 15)
p.set_title("Input Shuffling", fontsize = 20)
p.axvline(score_rand, ls='-', color='r')
score_label = (f"Score on original\ndata: {score_rand:.4f}\n"
               f"(p-value: {pvalue_rand:.4f})")
p.text(0, 10, score_label, fontsize=12)
p.set_xlim(0)


# # Fast Nested-CV

# In[ ]:


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import GridSearchCV,KFold,cross_val_predict,cross_val_score,StratifiedKFold

from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import classification_report,accuracy_score
# Following kf is the outer loop
outer_kf = StratifiedKFold(n_splits=8,shuffle=True,random_state=1)
inner_kf = StratifiedKFold(n_splits=3,shuffle=True,random_state=2)
model = LogisticRegression(solver='liblinear', l1_ratio=1, class_weight='balanced')
params ={"C":np.logspace(-3,3,7), "penalty":["l1", 'l2']}# l1 lasso l2 ridge
features = X_train.to_numpy()
target = Y_train.to_numpy()


# In[ ]:


from sklearn.metrics import auc

clf = GridSearchCV(estimator=model,param_grid=params,cv=inner_kf, scoring='roc_auc')
clf.fit(X_train,Y_train)
print('Non nested best score:',clf.best_score_)

nested_score = cross_val_score(clf,X_train,Y_train,cv=outer_kf, scoring='roc_auc')
print('Nested scores:',nested_score)
print('Nested score mean:',nested_score.mean())


# In[ ]:


outer_loop_accuracy_scores = []
inner_loop_won_params = []
inner_loop_accuracy_scores = []

# Looping through the outer loop, feeding each training set into a GSCV as the inner loop
for train_index,test_index in outer_kf.split(features,target):
    
    GSCV = GridSearchCV(estimator=model,param_grid=params,cv=inner_kf)
    
    # GSCV is looping through the training data to find the best parameters. This is the inner loop
    GSCV.fit(features[train_index],target[train_index])
    
    # The best hyper parameters from GSCV is now being tested on the unseen outer loop test data.
    pred = GSCV.predict(features[test_index])
    
    # Appending the "winning" hyper parameters and their associated accuracy score
    inner_loop_won_params.append(GSCV.best_params_)
    outer_loop_accuracy_scores.append(accuracy_score(target[test_index],pred))
    inner_loop_accuracy_scores.append(GSCV.best_score_)

for i in zip(inner_loop_won_params,outer_loop_accuracy_scores,inner_loop_accuracy_scores):
    print(i)

print('Mean of outer loop accuracy score:',np.mean(outer_loop_accuracy_scores))


# # https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

# In[ ]:


X_train = get_config('X_train')
Y_train = get_config('y_train')
X_test = get_config('X_test')
Y_test = get_config('y_test')
features = X_train.columns
features


# # Univariate Analysis

# from statsmodels.stats.multitest import multipletests

# In[ ]:


X_test2 = X_train.join(data['PID'])
X_test2


# In[ ]:


Y_test = Y_train.to_frame()
Y_test2 = Y_test.join(data['PID'])
Y_test2


# In[ ]:


#@title Load radiomic features and corresponding labels
features_df = X_test2
outcome_df = Y_test2
features_df.head()
outcome_df.head()
indx_features = pd.Index(features_df.PID)
indx_outcomes = pd.Index(outcome_df.PID)
indx_used_patients = indx_features.intersection(indx_outcomes).to_list()

features_df = features_df.set_index('PID')
outcome_df = outcome_df.set_index('PID')

X = features_df.loc[indx_used_patients]
Y = outcome_df.loc[indx_used_patients,'ECE_Pathology']
print("Labels:")
print(Y_train.head())
print("")
print("Radiomic Features:")
X_train.head()


# In[ ]:



from sklearn.preprocessing import LabelEncoder
test_prcntg = 0.2 #@param {type:"slider", min:0.1, max:0.4, step:0.05}

random_state_value = 1 # necessary for reproducibility


X_train3, X_test3, Y_train3, Y_test3 = train_test_split(
    X, Y, test_size=test_prcntg, random_state=random_state_value, stratify=Y.to_list())

le = LabelEncoder()
y_train = le.fit_transform(Y_train3)
y_test = le.transform(Y_test3)


# In[ ]:


from scipy.stats import shapiro, mannwhitneyu, ttest_ind
from sklearn import metrics

feature_names, sensitivity_list, specificity_list, auc_list, threshold_list, test_type_list, pvalue_list, pos_vs_neg_list = ([] for i in range(8))
for column in X_train3:
  # normality test
  stat, p = shapiro(X_train3[column])
  # print('Name ', column)
  # print('Statistics=%.3f, p=%.3f' % (stat, p))
  a_dist = X_train3[column][y_train==0]
  b_dist = X_train3[column][y_train==1]
  feature_names.append(column)
  # interpret
  alpha = 0.05
  if p > alpha:
    test_type_list.append('t-test')
    stats, pval = ttest_ind(a_dist, b_dist)
    # print('Sample looks Gaussian (fail to reject H0)')
  else:
    test_type_list.append('mann-whitney U-test')
    stats, pval = mannwhitneyu(a_dist, b_dist)
    # print('Sample does not look Gaussian (reject H0)')
  pvalue_list.append(pval)
  fpr, tpr, thresholds = metrics.roc_curve(y_train, X_train3[column], pos_label=1)
  auc = metrics.auc(fpr, tpr)
  pos_vs_neg = ">"
  if auc <0.5:
    fpr, tpr, thresholds = metrics.roc_curve(y_train, X_train3[column], 
                                             pos_label=0)
    auc = metrics.auc(fpr, tpr)
    pos_vs_neg = "<"
  auc_list.append(auc)
  pos_vs_neg_list.append(pos_vs_neg)
  ####################################
  # The optimal cut off would be where tpr is high and fpr is low
  # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
  ####################################
  i = np.arange(len(tpr)) # index for df
  roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
  cutoff_df = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

  sensitivity_list.append(cutoff_df['tpr'].values[0])
  specificity_list.append(cutoff_df['1-fpr'].values[0])
  threshold_list.append(cutoff_df['thresholds'].values[0])

train_auc_pvals_df = pd.DataFrame(list(zip(auc_list, pos_vs_neg_list,threshold_list, 
                                           sensitivity_list, specificity_list, 
                                           test_type_list, pvalue_list)), 
                                  index = feature_names,
                                  columns =['AUC', 'Pos.vs.Neg.', 
                                            'Cutoff-Threshold', 'Sensitivity', 
                                            'Specificity', 'Test', 'p-value'])

train_auc_pvals_df.sort_values(by='p-value', ascending=True)


# In[ ]:


#@title Lets count how many features are statistically significant between groups
print(sum(i < 0.05 for i in train_auc_pvals_df['p-value']))


# In[ ]:


from statsmodels.stats.multitest import multipletests
_, corrected_p_value, _, _ = multipletests(train_auc_pvals_df['p-value'], 
                                           alpha=0.05, method='bonferroni')

train_auc_pvals_corrected_df = pd.DataFrame(list(zip(auc_list, pos_vs_neg_list, 
                                                     threshold_list, 
                                                     sensitivity_list, 
                                                     specificity_list, 
                                                     test_type_list, 
                                                     pvalue_list, 
                                                     corrected_p_value)), 
                                  index = feature_names,
                                  columns =['AUC', 'Pos.vs.Neg.', 
                                            'Cutoff-Threshold', 'Sensitivity', 
                                            'Specificity', 'Test', 'p-value',
                                            'corrected p-value'])
train_auc_pvals_corrected_df.sort_values(by='corrected p-value', ascending=True)


# In[ ]:


#@title Visualizations of discriminative power of radiomic features
dropdown = 'CapsularContactLength' #@param ['squareroot_ngtdm_Strength', 'logarithm_ngtdm_Strength', 'square_gldm_GrayLevelNonUniformity', 'log-sigma-5-mm-3D_gldm_GrayLevelNonUniformity', 'wavelet-LL_ngtdm_Strength', 'wavelet-LL_gldm_GrayLevelNonUniformity', 'wavelet-LL_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformity', 'original_gldm_GrayLevelNonUniformity', 'log-sigma-3-mm-3D_gldm_GrayLevelNonUniformity', 'original_ngtdm_Strength', 'wavelet-LL_glszm_GrayLevelNonUniformity', 'log-sigma-5-mm-3D_glszm_ZoneVariance', 'original_glszm_GrayLevelNonUniformity', 'log-sigma-3-mm-3D_ngtdm_Coarseness', 'log-sigma-5-mm-3D_ngtdm_Coarseness', 'log-sigma-5-mm-3D_glszm_LargeAreaEmphasis', 'wavelet-LL_ngtdm_Busyness', 'squareroot_ngtdm_Busyness', 'logarithm_ngtdm_Coarseness']

import seaborn as sns
from matplotlib import pyplot
from sklearn.metrics import roc_curve
sns.set(rc={'figure.figsize':(9,9)})
sns.boxplot(x=Y_train, y=X_train[dropdown])



plt.figure(0)
lr_fpr, lr_tpr, _ = roc_curve(Y_train, X_train[dropdown])
# plot the roc curve for the model
pyplot.plot(lr_fpr, lr_tpr, marker='.', label=dropdown)
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# In[ ]:


check_fairness(lr,sensitive_features = ['Gleason biopsy'])


# In[ ]:


predict_model(lr, drift_report=True)


# In[ ]:


dashboard(lr)


# In[ ]:


X_train.columns


# In[ ]:




