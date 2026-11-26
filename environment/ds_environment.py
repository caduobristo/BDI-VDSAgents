"""Ambiente de execução para tarefas de ciência de dados e princípios PCS."""
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from maspy import Environment
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tools.modeling import METRICAS_SELECAO, avaliar_modelos
from tools.preprocessing import criar_preprocessador
from tools.unit_test import run_data_cleaning_tests


BANK_CHURN = "data/bank_churn/train.csv"


class DataScienceEnv(Environment):
    """Ambiente MASPY para manipulação de datasets, perturbações e auditoria PCS."""

    def __init__(
        self, env_name: Optional[str] = "DataScienceEnv", simular_falhas: bool = True, *,
        alvo: str = "Exited", tarefa: str = "classificacao",
        ignorar: Optional[List[str]] = None,
    ):
        if tarefa not in ("classificacao", "regressao"):
            raise ValueError("A tarefa deve ser classificação ou regressão.")
        super().__init__(env_name)
        self.simular_falhas = simular_falhas
        self.raw_test: Optional[pd.DataFrame] = None
        self.preprocessador = None
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.perturbed_datasets: List[pd.DataFrame] = []
        self.attempt_counter: int = 0
        self.target_column = alvo
        self.tarefa = tarefa
        self.metrica_principal = METRICAS_SELECAO[tarefa][0]
        self.identifier_columns = ignorar
        self._colunas_ignoradas: List[str] = []
        self._bank_churn = False

    def carregar_dataset(
        self,
        agt: str,
        path: str = BANK_CHURN,
        sample_size: int = 1000,
        separador: str = ",",
        metrica: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Carrega e armazena os dados brutos no ambiente."""
        if sample_size < 0:
            raise ValueError("O tamanho da amostra deve ser zero ou positivo.")
        df = pd.read_csv(path, sep=separador)
        alvo = self.target_column
        if alvo not in df.columns:
            raise ValueError(f"Coluna alvo '{alvo}' ausente no CSV: {', '.join(df.columns)}.")
        bank_churn = Path(path).resolve() == Path(BANK_CHURN).resolve() and alvo == "Exited"
        opcoes = METRICAS_SELECAO[self.tarefa]
        if metrica is None:
            metrica = "roc_auc" if bank_churn and self.tarefa == "classificacao" else opcoes[0]
        if metrica not in opcoes:
            raise ValueError(f"Métrica '{metrica}' incompatível com {self.tarefa}. Use {', '.join(opcoes)}.")
        if metrica == "roc_auc" and df[alvo].nunique() != 2:
            raise ValueError("ROC AUC requer um alvo de classificação binária.")
        ignoradas = list(self.identifier_columns or [])
        if self.identifier_columns is None:
            ignoradas = ["id", "CustomerId", "Surname"] if bank_churn else [
                col for col in df.columns if col.casefold() == "id" and col != alvo
            ]
        if alvo in ignoradas or set(ignoradas) - set(df.columns):
            raise ValueError("As colunas a ignorar devem existir no CSV e ser diferentes do alvo.")
        if len(set(df.columns) - {alvo, *ignoradas}) == 0:
            raise ValueError("O CSV precisa de pelo menos uma coluna de entrada.")
        if not bank_churn:
            df = df.drop_duplicates()
        if df[alvo].isna().any():
            raise ValueError(f"A coluna alvo '{alvo}' contém valores ausentes.")
        if self.tarefa == "regressao":
            df = df.assign(**{alvo: pd.to_numeric(df[alvo], errors="coerce")})
            if not np.isfinite(df[alvo]).all():
                raise ValueError(f"A coluna alvo '{alvo}' deve conter números finitos para regressão.")
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        if len(df) < 5:
            raise ValueError("São necessárias ao menos cinco linhas para dividir treino e teste.")
        classe = df[alvo] if self.tarefa == "classificacao" else None
        if classe is not None and (classe.nunique() < 2 or classe.value_counts().min() < 2):
            raise ValueError("Use uma amostra maior com pelo menos duas classes e dois registros por classe.")
        df, test = train_test_split(
            df, test_size=0.25, stratify=classe, random_state=42
        )
        if classe is not None and df[alvo].nunique() != classe.nunique():
            raise ValueError("A amostra de treino deve conter todas as classes do alvo.")
        if metrica == "roc_auc" and test[alvo].nunique() != 2:
            raise ValueError("ROC AUC requer as duas classes na validação interna. Use uma amostra maior.")
        df = df.copy()
        null_counts = df.isnull().sum()
        if bank_churn and self.simular_falhas and null_counts.sum() == 0:
            rng = np.random.default_rng(42)
            for col in ["Age", "Balance", "EstimatedSalary"]:
                if col in df.columns:
                    mask = rng.random(len(df)) < 0.05
                    df.loc[mask, col] = np.nan
            outlier_idx = rng.choice(df.index, size=min(15, len(df)), replace=False)
            df.loc[outlier_idx[:8], "Balance"] = df["Balance"].max() * 3.5
            df.loc[outlier_idx[8:], "EstimatedSalary"] = df["EstimatedSalary"].max() * 4.0
            null_counts = df.isnull().sum()

        self.raw_data = df
        self.raw_test = test
        self._bank_churn = bank_churn
        self.metrica_principal = metrica
        self._colunas_ignoradas = ignoradas
        self.processed_data = None
        self.preprocessador = None
        self.perturbed_datasets = []
        self.attempt_counter = 0
        cols_with_nulls = null_counts[null_counts > 0].to_dict()

        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_columns": cols_with_nulls,
            "has_missing": len(cols_with_nulls) > 0,
        }

    def verificar_nulidades(self, agt: str) -> Dict[str, Any]:
        """Inspeciona valores nulos no dataset corrente."""
        if self.raw_data is None:
            self.carregar_dataset(agt)
        df = self.raw_data if self.processed_data is None else self.processed_data

        null_series = df.isnull().sum()
        null_dict = null_series[null_series > 0].to_dict()
        return {
            "total_nulls": int(null_series.sum()),
            "columns_with_nulls": null_dict,
            "has_nulls": len(null_dict) > 0,
        }

    def _features(self, dados: pd.DataFrame) -> pd.DataFrame:
        return dados.drop(columns=[self.target_column, *self._colunas_ignoradas]).replace(
            [np.inf, -np.inf], np.nan
        )

    def _preparar_treino(self, agt: str, robusto: bool) -> Dict[str, Any]:
        if self.raw_data is None:
            self.carregar_dataset(agt)

        features = self._features(self.raw_data)
        preprocessador = criar_preprocessador(features, robusto)
        self.perturbed_datasets = []
        self.attempt_counter += 1
        self.preprocessador = None
        self.processed_data = None
        try:
            df = preprocessador.fit_transform(features)
            df.insert(len(df.columns), self.target_column, self.raw_data[self.target_column])
        except (ValueError, TypeError) as erro:
            return {
                "attempt": self.attempt_counter,
                "tests_passed": False,
                "report": f"Falha no pré-processamento: {erro}",
            }
        test_results = run_data_cleaning_tests(
            df,
            original_data_or_path=self.raw_data,
            linhas_para_duplicatas=None if self._bank_churn else self.raw_data,
        )

        self.preprocessador = preprocessador
        self.processed_data = df
        return {
            "attempt": self.attempt_counter,
            "strategy": "plano_b_contingencia_knn_iqr" if robusto else "plano_a_padrao",
            "shape": df.shape,
            "tests_passed": test_results["all_passed"],
            "report": test_results["report"],
        }

    def executar_limpeza_padrao(self, agt: str) -> Dict[str, Any]:
        return self._preparar_treino(agt, robusto=False)

    executar_limpeza_padrao_env = executar_limpeza_padrao

    def executar_limpeza_robusta(self, agt: str) -> Dict[str, Any]:
        return self._preparar_treino(agt, robusto=True)

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
        n_samples = max(2, int(len(self.processed_data) * sample_ratio))

        for i in range(k):
            perturbed_df = self.processed_data.sample(
                n=n_samples,
                replace=True,
                random_state=42 + i * 17,
            )
            if self.tarefa == "classificacao":
                alvo = self.target_column
                ausentes = self.processed_data.loc[
                    ~self.processed_data[alvo].isin(perturbed_df[alvo])
                ].drop_duplicates(subset=alvo)
                perturbed_df = pd.concat([perturbed_df, ausentes])
            self.perturbed_datasets.append(perturbed_df)

        return [{"index": i + 1, "shape": d.shape} for i, d in enumerate(self.perturbed_datasets)]

    def calcular_estabilidade_pcs(self, agt: str, limiar: float = 0.75) -> Tuple[float, bool, str]:
        """Calcula o score de estabilidade empírica do PCS com base nas perturbações."""
        if not self.perturbed_datasets:
            self.gerar_perturbacoes_bootstrap(agt)

        if len(self.perturbed_datasets) < 2:
            return 0.0, False, "A estabilidade exige ao menos duas perturbações."
        numeric_cols = [c for c in self.processed_data.select_dtypes(include=[np.number]).columns if c != self.target_column]
        if not numeric_cols:
            return 0.0, False, "A estabilidade exige ao menos uma variável numérica após a codificação."
        matrizes = [amostra[numeric_cols].corr().fillna(0).to_numpy() for amostra in self.perturbed_datasets]
        corr_diffs = [np.mean(np.abs(a - b)) for a, b in combinations(matrizes, 2)]
        # A diferença absoluta entre duas correlações varia de 0 a 2.
        score = 1.0 - float(np.mean(corr_diffs)) / 2.0

        colunas_churn = ("Balance", "EstimatedSalary") if self._bank_churn else ()
        for col in colunas_churn:
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
        """Treina as hipóteses do tipo de tarefa e avalia as perturbações."""
        if self.processed_data is None or self.raw_test is None or self.preprocessador is None:
            raise ValueError("Dados processados indisponíveis para modelagem.")

        alvo = self.target_column
        return avaliar_modelos(
            self.tarefa,
            self.processed_data.drop(columns=alvo),
            self.processed_data[alvo],
            self.preprocessador.transform(self._features(self.raw_test)),
            self.raw_test[alvo],
            self.perturbed_datasets,
            alvo,
            metrica_principal=self.metrica_principal,
        )

    def auditar_predicoes_pcs(self, agt: str, predictions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Emite o certificado final PCS com base nas métricas da tarefa."""
        modelos = predictions_data["models"]
        criterio = predictions_data["primary_metric"]
        melhor, resultado = max(modelos.items(), key=lambda item: item[1]["metricas"][criterio])
        metricas = resultado["metricas"]
        limiares = {"accuracy": 0.70, "roc_auc": 0.50, "r2": 0.0}
        previsibilidade_aprovada = metricas[criterio] > limiares[criterio]

        computability_passed = predictions_data["elapsed_time_seconds"] < 30.0

        mean_stability = float(np.mean([m["perturbation_stability"] for m in modelos.values()]))
        stability_passed = mean_stability >= 0.80

        certified = computability_passed and stability_passed and previsibilidade_aprovada

        certificate = {
            "status": "CERTIFICADO" if certified else "REPROVADO",
            "best_hypothesis": melhor,
            "primary_metric": criterio,
            "validation_scores": {nome: dados["metricas"][criterio] for nome, dados in modelos.items()},
            "predictability": {
                **metricas,
                "verified": previsibilidade_aprovada,
            },
            "computability": {
                "execution_time_seconds": round(predictions_data["elapsed_time_seconds"], 3),
                "verified": computability_passed,
            },
            "stability": {
                "mean_perturbation_stability": round(mean_stability, 4),
                "inter_model_agreement": round(predictions_data["inter_model_agreement"], 4),
                "verified": stability_passed,
            },
        }

        return certificate
