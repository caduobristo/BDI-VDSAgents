"""Pré-processamento ajustado no treino para os planos do ExploreAgent."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RobustNumericImputer(TransformerMixin, BaseEstimator):
    """Limita extremos por IQR e calcula vizinhos em escala padronizada."""

    def fit(self, X, y=None):
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        spread = 1.5 * (q3 - q1)
        self.lower_, self.upper_ = q1 - spread, q3 + spread
        clipped = X.clip(self.lower_, self.upper_, axis="columns")
        self.scaler_ = StandardScaler().fit(clipped)
        self.imputer_ = KNNImputer(n_neighbors=5, keep_empty_features=True)
        self.imputer_.fit(self.scaler_.transform(clipped))
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def transform(self, X):
        clipped = X.clip(self.lower_, self.upper_, axis="columns")
        scaled = self.scaler_.transform(clipped)
        values = self.scaler_.inverse_transform(self.imputer_.transform(scaled))
        # Colunas inteiramente vazias no treino não têm escala estimável.
        return pd.DataFrame(values, index=X.index, columns=X.columns).fillna(0.0)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def criar_preprocessador(features: pd.DataFrame, robusto: bool) -> ColumnTransformer:
    numericas = features.select_dtypes(include="number").columns.tolist()
    categoricas = [col for col in features if col not in numericas]
    numeric = RobustNumericImputer() if robusto else SimpleImputer(
        strategy="mean", keep_empty_features=True
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent", keep_empty_features=True),
        OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer(
        [("numericas", numeric, numericas), ("categoricas", categorical, categoricas)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
