import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

class BorutaPy4CV(BorutaPy):
    def __init__(self, n_estimators=1000, perc=100, alpha=0.05,
                 two_step=True, max_iter=100, random_state=None, verbose=0,
                 weak=False, passThrough=False):

        self.passThrough = passThrough
        self.weak = weak
        super().__init__(estimator=RandomForestClassifier(class_weight='balanced', max_depth=5),
                         n_estimators=n_estimators,
                         perc=perc, alpha=alpha, two_step=two_step,
                         max_iter=max_iter, random_state=random_state,
                         verbose=verbose) #early_stopping=early_stopping,
                         #n_iter_no_change=n_iter_no_change)

    def _fit(self, X, y):
        if self.passThrough:
            self.support_ = np.ones(X.shape[1], dtype=bool)
            self.support_weak_ = np.ones(X.shape[1], dtype=bool)
            self.ranking_ = 1 + np.array(range(X.shape[1]))
            self.n_features_ = X.shape[1]
            self.most_imp_feature_ = 1
        else:
            super()._fit(X, y)
            self.most_imp_feature_ = np.argmax(self._get_imp(X, y))
        return self

    def _transform(self, X):
        if self.weak:
            support = np.logical_or(self.support_, self.support_weak_)
        else:
            support = self.support_

        if np.sum(support) > 0:
            return super()._transform(X, weak=self.weak)
        else:
            print('Warning: No confirmed feature; Outputting the best rejected feature.')
            return X[:,self.most_imp_feature_].reshape(-1, 1)

    def fit_transform(self, X, y):
        self._fit(X, y)
        return self._transform(X)

    def transform(self, X):
        return self._transform(X)

    def allow_weak_features(self):
        self.weak = True

    def disallow_weak_features(self):
        self.weak = False
