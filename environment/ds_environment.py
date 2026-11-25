"""Ambiente de execução para tarefas de ciência de dados e princípios PCS."""
import time
from typing import Any, Dict, List, Optional, Tuple
from maspy import Environment
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tools.ml_tools import (
    encode_categorical_tools,
    fill_missing_values_tools,
    handle_outliers_tools,
    remove_columns_tools,
)
from tools.unit_test import run_data_cleaning_tests


class DataScienceEnv(Environment):
    """Ambiente MASPY para manipulação de datasets, perturbações e auditoria PCS."""

    def __init__(self, env_name: Optional[str] = "DataScienceEnv"):
        super().__init__(env_name)
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.perturbed_datasets: List[pd.DataFrame] = []
        self.attempt_counter: int = 0
        self.target_column: str = "Exited"
        self.identifier_columns: List[str] = ["id", "CustomerId", "Surname"]
        self.numeric_columns: List[str] = [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
        ]
        self.categorical_columns: List[str] = ["Geography", "Gender"]

    def carregar_dataset(
        self,
        agt: str,
        path: str = "data/bank_churn/train.csv",
        sample_size: int = 1000,
    ) -> Dict[str, Any]:
        """Carrega e armazena os dados brutos no ambiente."""
        df = pd.read_csv(path)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        null_counts = df.isnull().sum()
        # Injeta nulos e outliers controlados na amostra em memória se o dataset original for 100% preenchido
        if null_counts.sum() == 0 and "Age" in df.columns:
            np.random.seed(42)
            for col in ["Age", "Balance", "EstimatedSalary"]:
                if col in df.columns:
                    mask = np.random.rand(len(df)) < 0.05
                    df.loc[mask, col] = np.nan
            outlier_idx = np.random.choice(df.index, size=15, replace=False)
            df.loc[outlier_idx[:8], "Balance"] = df["Balance"].max() * 3.5
            df.loc[outlier_idx[8:], "EstimatedSalary"] = df["EstimatedSalary"].max() * 4.0
            null_counts = df.isnull().sum()

        self.raw_data = df
        cols_with_nulls = null_counts[null_counts > 0].to_dict()

        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_columns": cols_with_nulls,
            "has_missing": len(cols_with_nulls) > 0,
        }

    def verificar_nulidades(self, agt: str) -> Dict[str, Any]:
        """Inspeciona valores nulos no dataset corrente."""
        df = self.raw_data if self.processed_data is None else self.processed_data
        if df is None:
            raise ValueError("Nenhum dado carregado no ambiente.")

        null_series = df.isnull().sum()
        null_dict = null_series[null_series > 0].to_dict()
        return {
            "total_nulls": int(null_series.sum()),
            "columns_with_nulls": null_dict,
            "has_nulls": len(null_dict) > 0,
        }

    def executar_limpeza_padrao(self, agt: str) -> Dict[str, Any]:
        """Executa o Plano A: imputação ingênua (média/moda) sem tratamento de outliers."""
        if self.raw_data is None:
            self.carregar_dataset(agt)

        self.attempt_counter += 1
        df = self.raw_data.copy()

        # Remove identificadores sem poder preditivo
        df = remove_columns_tools(df, columns=self.identifier_columns)

        # Imputação básica por média para numéricas e moda para categóricas
        missing_targets = [c for c in df.columns if df[c].isna().any()]
        if missing_targets:
            df = fill_missing_values_tools(df, target_columns=missing_targets, method="auto")

        # Codificação de categóricas
        cats_to_encode = [c for c in self.categorical_columns if c in df.columns]
        if cats_to_encode:
            df = encode_categorical_tools(df, target_columns=cats_to_encode, method="onehot")

        # Executa testes unitários de limpeza
        test_results = run_data_cleaning_tests(df, original_data_or_path=self.raw_data)
        self.processed_data = df

        return {
            "attempt": self.attempt_counter,
            "strategy": "plano_a_padrao",
            "shape": df.shape,
            "tests_passed": test_results["all_passed"],
            "report": test_results["report"],
        }

    executar_limpeza_padrao_env = executar_limpeza_padrao
    executar_limpeza_robusta_env = None

    def executar_limpeza_robusta(self, agt: str) -> Dict[str, Any]:
        """Executa o Plano B (Contingência): imputação KNN e tratamento de outliers com IQR."""
        if self.raw_data is None:
            self.carregar_dataset(agt)

        self.attempt_counter += 1
        df = self.raw_data.copy()

        # Remove identificadores
        df = remove_columns_tools(df, columns=self.identifier_columns)

        # Trata outliers com IQR antes ou durante a imputação para evitar contaminação
        existing_numeric = [c for c in self.numeric_columns if c in df.columns and c != self.target_column]
        df = handle_outliers_tools(
            df,
            target_columns=existing_numeric,
            method="iqr",
            strategy="clip",
            sensitivity="medium",
        )

        # Imputação robusta baseada em vizinhos mais próximos (KNN)
        missing_targets = [c for c in df.columns if df[c].isna().any()]
        if missing_targets:
            df = fill_missing_values_tools(
                df,
                target_columns=missing_targets,
                method="knn",
                n_neighbors=5,
            )

        # Codificação de categóricas
        cats_to_encode = [c for c in self.categorical_columns if c in df.columns]
        if cats_to_encode:
            df = encode_categorical_tools(df, target_columns=cats_to_encode, method="onehot")

        # Executa testes unitários de limpeza
        test_results = run_data_cleaning_tests(df, original_data_or_path=self.raw_data)
        self.processed_data = df

        return {
            "attempt": self.attempt_counter,
            "strategy": "plano_b_contingencia_knn_iqr",
            "shape": df.shape,
            "tests_passed": test_results["all_passed"],
            "report": test_results["report"],
        }

    executar_limpeza_robusta_env = executar_limpeza_robusta

    def gerar_perturbacoes_bootstrap(
        self,
        agt: str,
        k: int = 3,
        sample_ratio: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """Gera k amostras perturbadas via bootstrap para avaliação de estabilidade."""
        if self.processed_data is None:
            raise ValueError("Dados pré-processados não encontrados para perturbação.")

        self.perturbed_datasets = []
        n_samples = int(len(self.processed_data) * sample_ratio)

        for i in range(k):
            perturbed_df = self.processed_data.sample(
                n=n_samples,
                replace=True,
                random_state=42 + i * 17,
            ).reset_index(drop=True)
            self.perturbed_datasets.append(perturbed_df)

        return [{"index": i + 1, "shape": d.shape} for i, d in enumerate(self.perturbed_datasets)]

    def calcular_estabilidade_pcs(self, agt: str, limiar: float = 0.75) -> Tuple[float, bool, str]:
        """Calcula o score de estabilidade empírica do PCS com base nas perturbações."""
        if not self.perturbed_datasets:
            self.gerar_perturbacoes_bootstrap(agt)

        # Mede a estabilidade através da variância das correlações e dispersão sob bootstrap
        corr_diffs = []
        k = len(self.perturbed_datasets)
        numeric_cols = [c for c in self.processed_data.select_dtypes(include=[np.number]).columns if c != self.target_column]

        for i in range(k):
            for j in range(i + 1, k):
                corr_i = self.perturbed_datasets[i][numeric_cols].corr().fillna(0).values
                corr_j = self.perturbed_datasets[j][numeric_cols].corr().fillna(0).values
                diff = np.mean(np.abs(corr_i - corr_j))
                corr_diffs.append(diff)

        if not corr_diffs or not numeric_cols:
            return 0.0, False, "A estabilidade exige features numéricas e ao menos duas perturbações."
        # A diferença absoluta entre duas correlações varia de 0 a 2.
        score = 1.0 - float(np.mean(corr_diffs)) / 2.0

        # Verifica se os outliers foram tratados na tentativa corrente
        for col in ["Balance", "EstimatedSalary"]:
            if col in self.processed_data.columns:
                q99 = self.processed_data[col].quantile(0.99)
                max_val = self.processed_data[col].max()
                if max_val > q99 * 2.0:
                    return score, False, f"Outlier em {col}: máximo superior ao dobro do percentil 99."

        return score, score >= limiar, f"Estabilidade bootstrap {score:.4f}, limiar {limiar:.4f}."

    def obter_matriz_features_aprovadas(self, agt: str) -> Dict[str, Any]:
        """Retorna metadados da matriz de features aprovada para o ModelAgent."""
        if self.processed_data is None:
            raise ValueError("Nenhum dado pré-processado disponível.")

        feature_cols = [c for c in self.processed_data.columns if c != self.target_column]
        return {
            "n_samples": len(self.processed_data),
            "n_features": len(feature_cols),
            "features": feature_cols,
            "target": self.target_column,
        }

    def treinar_e_avaliar_hipoteses(self, agt: str) -> Dict[str, Any]:
        """Treina hipóteses concorrentes (Random Forest e Logistic Regression) e avalia perturbações."""
        if self.processed_data is None:
            raise ValueError("Dados processados indisponíveis para modelagem.")

        df = self.processed_data.copy()
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        start_time = time.time()

        # Hipótese 1: Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred, zero_division=0)

        # Hipótese 2: Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_acc = accuracy_score(y_test, lr_pred)
        lr_f1 = f1_score(y_test, lr_pred, zero_division=0)

        elapsed_time = time.time() - start_time

        # Avaliação de estabilidade das predições nas amostras perturbadas
        rf_pert_preds = []
        lr_pert_preds = []

        for p_df in self.perturbed_datasets:
            p_X = p_df.drop(columns=[self.target_column])
            # Predição na mesma base de teste usando modelos treinados nas perturbações
            p_y = p_df[self.target_column].astype(int)
            m_rf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)
            m_rf.fit(p_X, p_y)
            rf_pert_preds.append(m_rf.predict(X_test))

            m_lr = LogisticRegression(max_iter=500, random_state=42)
            m_lr.fit(scaler.transform(p_X), p_y)
            lr_pert_preds.append(m_lr.predict(X_test_scaled))

        # Calcula a concordância média (estabilidade) entre as perturbações
        def calcular_concordancia(preds_list):
            if not preds_list:
                return 1.0
            agreements = []
            for i in range(len(preds_list)):
                for j in range(i + 1, len(preds_list)):
                    agreements.append(np.mean(preds_list[i] == preds_list[j]))
            return float(np.mean(agreements))

        rf_stability = calcular_concordancia(rf_pert_preds)
        lr_stability = calcular_concordancia(lr_pert_preds)

        # Concordância entre as duas hipóteses concorrentes
        inter_model_agreement = float(np.mean(rf_pred == lr_pred))

        return {
            "elapsed_time_seconds": round(elapsed_time, 3),
            "random_forest": {
                "accuracy": round(rf_acc, 4),
                "f1_score": round(rf_f1, 4),
                "perturbation_stability": round(rf_stability, 4),
            },
            "logistic_regression": {
                "accuracy": round(lr_acc, 4),
                "f1_score": round(lr_f1, 4),
                "perturbation_stability": round(lr_stability, 4),
            },
            "inter_model_agreement": round(inter_model_agreement, 4),
            "test_sample_size": len(y_test),
        }

    def auditar_predicoes_pcs(self, agt: str, predictions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Emite o certificado final PCS com base nos critérios de Predictability, Computability e Stability."""
        rf_metrics = predictions_data["random_forest"]
        lr_metrics = predictions_data["logistic_regression"]

        # Predictability: desempenho preditivo verificado
        best_model_name, best_metrics = max(
            [("RandomForest", rf_metrics), ("LogisticRegression", lr_metrics)],
            key=lambda item: item[1]["accuracy"],
        )
        best_acc = best_metrics["accuracy"]
        best_f1 = best_metrics["f1_score"]

        # Computability: convergência dos algoritmos e viabilidade computacional
        computability_passed = predictions_data["elapsed_time_seconds"] < 30.0

        # Stability: consistência entre perturbações (> 0.80)
        mean_stability = (rf_metrics["perturbation_stability"] + lr_metrics["perturbation_stability"]) / 2.0
        stability_passed = mean_stability >= 0.80

        certified = computability_passed and stability_passed and (best_acc > 0.70)

        certificate = {
            "status": "CERTIFICADO" if certified else "REPROVADO",
            "best_hypothesis": best_model_name,
            "predictability": {
                "accuracy": best_acc,
                "f1_score": best_f1,
                "verified": best_acc > 0.70,
            },
            "computability": {
                "execution_time_seconds": predictions_data["elapsed_time_seconds"],
                "converged": True,
                "verified": computability_passed,
            },
            "stability": {
                "mean_perturbation_stability": round(mean_stability, 4),
                "inter_model_agreement": predictions_data["inter_model_agreement"],
                "verified": stability_passed,
            },
        }

        return certificate
