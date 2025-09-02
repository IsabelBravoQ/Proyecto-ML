
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IntensityImputer(BaseEstimator, TransformerMixin):
    def __init__(self, magnitude_col="magnitude_Mw", intensity_col="intensity"):
        self.magnitude_col = magnitude_col
        self.intensity_col = intensity_col

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.median_by_mag_ = X.groupby(self.magnitude_col)[self.intensity_col].median().to_dict()
        self.global_median_ = X[self.intensity_col].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.intensity_col] = X.apply(
            lambda r: r[self.intensity_col]
            if pd.notna(r[self.intensity_col])
            else self.median_by_mag_.get(r[self.magnitude_col], pd.NA),
            axis=1
        )
        X[self.intensity_col] = X[self.intensity_col].fillna(self.global_median_)
        X[self.intensity_col] = X[self.intensity_col].round().astype(int)
        return X