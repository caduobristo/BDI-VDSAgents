"""Módulo de ferramentas de manipulação tabular e pré-processamento para ciência de dados."""
from functools import partial
import json
import os
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


def fill_missing_values_tools(
    data: pd.DataFrame,
    target_columns: Union[str, List[str]],
    method: str = "auto",
    group_columns: Optional[Union[str, List[str]]] = None,
    time_column: Optional[str] = None,
    fill_value: Optional[Any] = None,
    max_group_null_ratio: float = 0.8,
    **params,
) -> pd.DataFrame:
    """Realiza imputação de valores ausentes em colunas selecionadas."""
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            raise ValueError("Formato de parâmetros JSON inválido.")

    df = data.copy()
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    if isinstance(group_columns, str):
        group_columns = [group_columns]

    for target_col in target_columns:
        if target_col not in df.columns:
            raise ValueError(f"Coluna {target_col} não encontrada nos dados.")

        if not df[target_col].isna().any():
            continue

        if pd.api.types.is_numeric_dtype(df[target_col]):
            df[target_col] = df[target_col].astype(float)

        if method == "auto":
            if pd.api.types.is_numeric_dtype(df[target_col]):
                fill_val = df[target_col].mean()
                df[target_col] = df[target_col].fillna(fill_val)
            else:
                fill_val = df[target_col].mode()[0] if not df[target_col].mode().empty else "Unknown"
                df[target_col] = df[target_col].fillna(fill_val)
            continue

        if method == "constant" and fill_value is None:
            fill_value = 0 if pd.api.types.is_numeric_dtype(df[target_col]) else "Unknown"

        if group_columns:
            if method in ["mean", "median", "mode"]:
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    if method == "mean":
                        fill_values = df.groupby(group_columns)[target_col].transform("mean")
                    elif method == "median":
                        fill_values = df.groupby(group_columns)[target_col].transform("median")
                    else:
                        fill_values = df.groupby(group_columns)[target_col].transform(
                            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                        )
                else:
                    fill_values = df.groupby(group_columns)[target_col].transform(
                        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                    )

                mask = df[target_col].isna()
                group_null_ratios = df[mask].groupby(group_columns).size() / df.groupby(group_columns).size()
                valid_groups = group_null_ratios[group_null_ratios <= max_group_null_ratio].index

                if len(valid_groups) > 0:
                    valid_mask = mask & df[group_columns].isin(valid_groups).all(axis=1)
                    df.loc[valid_mask, target_col] = fill_values[valid_mask]

            elif method in ["ffill", "bfill"]:
                if time_column:
                    df[target_col] = df.sort_values([time_column]).groupby(group_columns)[target_col].transform(
                        lambda x: x.ffill() if method == "ffill" else x.bfill()
                    )
                else:
                    df[target_col] = df.groupby(group_columns)[target_col].transform(
                        lambda x: x.ffill() if method == "ffill" else x.bfill()
                    )

            elif method == "interpolate":
                df[target_col] = df.groupby(group_columns)[target_col].transform(
                    lambda x: x.interpolate(method="linear", limit_direction="both")
                )
        else:
            if method == "constant":
                df[target_col] = df[target_col].fillna(fill_value)
            elif method in ["mean", "median", "mode"]:
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    fill_val = df[target_col].mean() if method == "mean" else (
                        df[target_col].median() if method == "median" else df[target_col].mode()[0]
                    )
                else:
                    fill_val = df[target_col].mode()[0] if not df[target_col].mode().empty else "Unknown"
                df[target_col] = df[target_col].fillna(fill_val)
            elif method == "ffill":
                df[target_col] = df[target_col].ffill()
            elif method == "bfill":
                df[target_col] = df[target_col].bfill()
            elif method == "interpolate":
                df[target_col] = df[target_col].interpolate(method="linear", limit_direction="both")

        if method == "knn":
            n_neighbors = params.get("n_neighbors", 5)
            weights = params.get("weights", "uniform")

            if group_columns:
                features = group_columns.copy() if isinstance(group_columns, list) else [group_columns]
            else:
                features = df.select_dtypes(include=[np.number]).columns.tolist()

            if target_col in features:
                features.remove(target_col)

            if not features:
                # Fallback para média caso não existam colunas numéricas de suporte
                df[target_col] = df[target_col].fillna(df[target_col].mean())
                continue

            X = df[features].copy()
            for cat_col in X.select_dtypes(include=["object", "category"]).columns:
                X[cat_col] = pd.Categorical(X[cat_col]).codes

            numeric_features = X.select_dtypes(include=[np.number]).columns
            if len(numeric_features) > 0:
                scaler = StandardScaler()
                X[numeric_features] = scaler.fit_transform(X[numeric_features].fillna(0))

            imputer = KNNImputer(n_neighbors=min(n_neighbors, max(1, len(df) - 1)), weights=weights)
            data_to_impute = df[[target_col]].copy()

            if pd.api.types.is_numeric_dtype(data_to_impute[target_col]):
                combined = pd.concat([data_to_impute, X], axis=1)
                imputed_res = imputer.fit_transform(combined)[:, 0]
                df[target_col] = imputed_res
            else:
                le = LabelEncoder()
                non_missing_mask = ~data_to_impute[target_col].isna()
                temp_values = data_to_impute[target_col].copy()
                temp_values[non_missing_mask] = le.fit_transform(temp_values[non_missing_mask])
                temp_values = temp_values.astype(float)
                combined = pd.concat([pd.DataFrame(temp_values), X], axis=1)
                imputed_res = imputer.fit_transform(combined)[:, 0]
                temp_round = np.round(imputed_res).astype(int)
                temp_round = np.clip(temp_round, 0, len(le.classes_) - 1)
                df[target_col] = le.inverse_transform(temp_round)

    return df


def detect_outliers_tools(
    series: pd.Series,
    method: str = "iqr",
    params: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """Detecta outliers em uma série numérica e retorna máscara booleana."""
    params = params or {}

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        threshold = params.get("threshold", 1.5)
        return (series < (q1 - threshold * iqr)) | (series > (q3 + threshold * iqr))

    elif method == "zscore":
        threshold = params.get("threshold", 3.0)
        z_scores = np.abs(stats.zscore(series.dropna()))
        out_mask = pd.Series(False, index=series.index)
        out_mask.loc[series.dropna().index] = z_scores > threshold
        return out_mask

    elif method == "mad":
        threshold = params.get("threshold", 3.5)
        median = series.median()
        mad = stats.median_abs_deviation(series.dropna())
        if mad == 0:
            return pd.Series(False, index=series.index)
        modified_z = 0.6745 * (series - median) / mad
        return np.abs(modified_z) > threshold

    return pd.Series(False, index=series.index)


def handle_outliers_tools(
    data: pd.DataFrame,
    target_columns: Union[str, List[str]],
    method: str = "iqr",
    strategy: str = "clip",
    sensitivity: str = "medium",
    group_columns: Optional[Union[str, List[str]]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Trata outliers em variáveis numéricas com suporte a recorte ou remoção."""
    sensitivity_params = {
        "low": {"iqr": {"threshold": 3.0}, "zscore": {"threshold": 4.0}},
        "medium": {"iqr": {"threshold": 1.5}, "zscore": {"threshold": 3.0}},
        "high": {"iqr": {"threshold": 1.0}, "zscore": {"threshold": 2.0}},
    }

    if sensitivity not in sensitivity_params:
        sensitivity = "medium"

    default_params = sensitivity_params[sensitivity].get(method, {"threshold": 1.5})
    merged_params = {**default_params, **(params or {})}

    df = data.copy()
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    for target_col in target_columns:
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            continue

        df[target_col] = df[target_col].astype(float)
        threshold = merged_params.get("threshold", 1.5)

        if method == "iqr":
            q1 = df[target_col].quantile(0.25)
            q3 = df[target_col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr

            if strategy == "clip":
                df[target_col] = df[target_col].clip(lower=lower, upper=upper)
            else:
                mask = (df[target_col] >= lower) & (df[target_col] <= upper)
                df = df.loc[mask]
        else:
            mask = detect_outliers_tools(df[target_col], method=method, params=merged_params)
            if strategy == "clip":
                normal_mean = df.loc[~mask, target_col].mean()
                df.loc[mask, target_col] = normal_mean
            else:
                df = df.loc[~mask]

    return df


def encode_categorical_tools(
    data: pd.DataFrame,
    target_columns: Union[str, List[str]],
    method: str = "auto",
    group_columns: Optional[Union[str, List[str]]] = None,
    keep_original: bool = False,
) -> pd.DataFrame:
    """Codifica colunas categóricas utilizando One-Hot, Label ou frequências."""
    df = data.copy()
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    columns_to_encode = [col for col in target_columns if col in df.columns]

    for target_col in columns_to_encode:
        if method in ["auto", "onehot"]:
            dummies = pd.get_dummies(df[target_col], prefix=target_col, drop_first=True, dtype=float)
            df = pd.concat([df, dummies], axis=1)
            if not keep_original:
                df.drop(columns=[target_col], inplace=True)
        elif method == "label":
            le = LabelEncoder()
            df[f"{target_col}_encoded"] = le.fit_transform(df[target_col].astype(str))
            if not keep_original:
                df.drop(columns=[target_col], inplace=True)
        elif method == "frequency":
            freq = df[target_col].value_counts(normalize=True)
            df[f"{target_col}_freq"] = df[target_col].map(freq)
            if not keep_original:
                df.drop(columns=[target_col], inplace=True)

    return df


def select_features(
    data: pd.DataFrame,
    target: Optional[pd.Series] = None,
    method: str = "variance",
    n_features: Optional[int] = None,
    params: Optional[dict] = None,
) -> List[str]:
    """Seleciona as variáveis mais relevantes por variância, correlação ou informação mútua."""
    df = data.copy()
    params = params or {}

    if target is not None and target.name in df.columns:
        df = df.drop(columns=[target.name])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return list(df.columns)

    df_num = df[numeric_cols].fillna(0)

    if method == "variance":
        threshold = params.get("threshold", 0.0)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df_num)
        selected = [col for col, sup in zip(numeric_cols, selector.get_support()) if sup]
        if n_features and len(selected) > n_features:
            variances = df_num[selected].var()
            selected = variances.nlargest(n_features).index.tolist()
        return selected

    if method == "correlation" and target is not None:
        corrs = {}
        target_clean = pd.to_numeric(target, errors="coerce").fillna(0)
        for col in numeric_cols:
            val, _ = stats.spearmanr(df_num[col], target_clean)
            corrs[col] = abs(val) if not np.isnan(val) else 0.0

        sorted_cols = sorted(corrs, key=corrs.get, reverse=True)
        k = n_features if n_features is not None else len(sorted_cols)
        return sorted_cols[:k]

    return numeric_cols


def remove_columns_tools(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Remove colunas especificadas do DataFrame."""
    df = data.copy()
    if columns:
        cols_to_drop = [c for c in columns if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
    return df
