"""Ambiente de execução para tarefas de ciência de dados e princípios PCS."""
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple
from maspy import Environment
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split

from tools.modeling import METRICAS_SELECAO, PLANO_COMPLETO, PLANO_REDUZIDO, avaliar_modelos
from tools.preprocessing import criar_preprocessador
from tools.unit_test import run_data_cleaning_tests


class DataScienceEnv(Environment):
    """Ambiente MASPY para manipulação de datasets, perturbações e auditoria PCS."""

    def __init__(
        self, env_name: Optional[str] = "DataScienceEnv", simular_falhas: bool = False, *,
        alvo: str, tarefa: str = "classificacao",
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
        self._linhas_originais: Optional[pd.DataFrame] = None
        self._coluna_simulada: Optional[str] = None

    def carregar_dataset(
        self,
        agt: str,
        path: str,
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
        opcoes = METRICAS_SELECAO[self.tarefa]
        if metrica is None:
            metrica = opcoes[0]
        if metrica not in opcoes:
            raise ValueError(f"Métrica '{metrica}' incompatível com {self.tarefa}. Use {', '.join(opcoes)}.")
        if metrica == "roc_auc" and df[alvo].nunique() != 2:
            raise ValueError("ROC AUC requer um alvo de classificação binária.")
        ignoradas = list(self.identifier_columns) if self.identifier_columns is not None else [
            col for col in df.columns if col.casefold() == "id" and col != alvo
        ]
        if alvo in ignoradas or set(ignoradas) - set(df.columns):
            raise ValueError("As colunas a ignorar devem existir no CSV e ser diferentes do alvo.")
        if len(set(df.columns) - {alvo, *ignoradas}) == 0:
            raise ValueError("O CSV precisa de pelo menos uma coluna de entrada.")
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
        linhas_originais = df.copy() if self.simular_falhas else df
        coluna_simulada = self._simular_falha_de_limpeza(df, ignoradas) if self.simular_falhas else None
        null_counts = df.isnull().sum()

        self.raw_data = df
        self.raw_test = test
        self._linhas_originais = linhas_originais
        self._coluna_simulada = coluna_simulada
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

    def _simular_falha_de_limpeza(self, treino: pd.DataFrame, ignoradas: List[str]) -> str:
        numericas = treino.drop(columns=[self.target_column, *ignoradas]).select_dtypes(include="number")
        for coluna, valores in numericas.items():
            valores = valores.replace([np.inf, -np.inf], np.nan)
            amplitude = valores.quantile(0.75) - valores.quantile(0.25)
            if amplitude > 0:
                rng = np.random.default_rng(42)
                treino[coluna] = treino[coluna].astype(float)
                indices = rng.permutation(treino.index)
                treino.loc[indices[:max(1, len(treino) // 20)], coluna] = np.nan
                treino.loc[indices[-1], coluna] = valores.max() + 30 * amplitude
                return coluna
        raise ValueError("A simulação exige pelo menos uma variável numérica com intervalo interquartil positivo.")

    def verificar_nulidades(self, agt: str) -> Dict[str, Any]:
        """Inspeciona valores nulos no dataset corrente."""
        if self.raw_data is None:
            raise ValueError("Carregue o dataset antes de verificar nulidades.")
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
            raise ValueError("Carregue o dataset antes de executar a limpeza.")

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
            linhas_para_duplicatas=self._linhas_originais,
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

        if self._coluna_simulada is not None:
            valores = self.processed_data[self._coluna_simulada]
            amplitude = valores.quantile(0.75) - valores.quantile(0.25)
            if valores.max() > valores.quantile(0.99) + 3 * amplitude:
                return score, False, f"Outlier em {self._coluna_simulada} após a limpeza."

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
            "permite_replanejamento": len(feature_cols) > 1,
        }

    def treinar_e_avaliar_hipoteses(self, agt: str, plano: str = PLANO_COMPLETO) -> Dict[str, Any]:
        """Treina as hipóteses do tipo de tarefa e avalia as perturbações."""
        if self.processed_data is None or self.raw_test is None or self.preprocessador is None:
            raise ValueError("Dados processados indisponíveis para modelagem.")

        alvo = self.target_column
        treino = self.processed_data.drop(columns=alvo)
        validacao = self.preprocessador.transform(self._features(self.raw_test))
        perturbacoes = self.perturbed_datasets
        if plano == PLANO_REDUZIDO:
            if treino.shape[1] < 2:
                raise ValueError("O plano reduzido requer ao menos duas variáveis.")
            pontuar = f_regression if self.tarefa == "regressao" else f_classif
            seletor = SelectKBest(pontuar, k=treino.shape[1] // 2).fit(treino, self.processed_data[alvo])
            colunas = list(seletor.get_feature_names_out())
            treino = treino[colunas]
            validacao = validacao[colunas]
            perturbacoes = [amostra[colunas + [alvo]] for amostra in perturbacoes]
        elif plano != PLANO_COMPLETO:
            raise ValueError(f"Plano de modelagem desconhecido: {plano}")
        resultado = avaliar_modelos(
            self.tarefa, treino, self.processed_data[alvo], validacao, self.raw_test[alvo],
            perturbacoes, alvo, metrica_principal=self.metrica_principal,
        )
        return {**resultado, "plano": plano, "variaveis": list(treino.columns)}

    def auditar_predicoes_pcs(self, agt: str, predictions_data: Dict[str, Any],
                              limiar_estabilidade: float = 0.80) -> Dict[str, Any]:
        """Emite o certificado final PCS com base nas métricas da tarefa."""
        modelos = predictions_data["models"]
        criterio = predictions_data["primary_metric"]
        limiares = {"accuracy": 0.70, "roc_auc": 0.50, "r2": 0.0}
        computability_passed = predictions_data["elapsed_time_seconds"] < 30.0
        auditorias = {}
        for nome, dados in modelos.items():
            valor = dados["metricas"][criterio]
            estabilidade = dados["perturbation_stability"]
            motivos = []
            if not valor > limiares[criterio]:
                motivos.append(f"{criterio}={valor:.4f} precisa superar {limiares[criterio]:.2f}")
            if not estabilidade >= limiar_estabilidade:
                motivos.append(f"estabilidade abaixo do limiar {limiar_estabilidade:g}")
            if not computability_passed:
                motivos.append("tempo da tentativa atingiu o limite de 30 s")
            auditorias[nome] = {
                "metricas": dados["metricas"], "estabilidade": estabilidade,
                "aprovado": not motivos, "motivos": motivos,
            }

        aprovados = [nome for nome, auditoria in auditorias.items() if auditoria["aprovado"]]
        melhor = max(aprovados, key=lambda nome: modelos[nome]["metricas"][criterio], default=None)
        mean_stability = float(np.mean([m["perturbation_stability"] for m in modelos.values()]))
        return {
            "status": "CERTIFICADO" if melhor is not None else "REPROVADO",
            "best_hypothesis": melhor,
            "auditoria": {
                "plano": predictions_data["plano"], "variaveis": predictions_data["variaveis"],
                "criterio": criterio, "limiar_previsibilidade": limiares[criterio],
                "limiar_estabilidade": limiar_estabilidade, "modelos": auditorias,
                "tempo_segundos": predictions_data["elapsed_time_seconds"],
                "modelo_selecionado": melhor,
            },
            "primary_metric": criterio,
            "validation_scores": {nome: dados["metricas"][criterio] for nome, dados in modelos.items()},
            "predictability": {
                **(modelos[melhor]["metricas"] if melhor is not None else {}),
                "verified": melhor is not None,
            },
            "computability": {
                "execution_time_seconds": round(predictions_data["elapsed_time_seconds"], 3),
                "verified": computability_passed,
            },
            "stability": {
                "mean_perturbation_stability": round(mean_stability, 4),
                "selected_perturbation_stability": modelos[melhor]["perturbation_stability"] if melhor is not None else None,
                "inter_model_agreement": round(predictions_data["inter_model_agreement"], 4),
                "verified": melhor is not None,
            },
        }
