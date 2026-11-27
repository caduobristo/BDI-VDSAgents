"""Ambiente de execução para tarefas de ciência de dados e princípios PCS."""
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple
from maspy import Environment
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from tools.modeling import METRICAS_SELECAO, avaliar_modelos
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
        self.colunas_entrada = []
        self.candidatos_avaliados = {}
        self.variaveis_selecionadas = None
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
        self.colunas_entrada = []
        self.candidatos_avaliados = {}
        self.variaveis_selecionadas = None
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

    def aplicar_preparo(self, agt: str, estrategia: str, construir) -> Dict[str, Any]:
        if self.raw_data is None:
            raise ValueError("Carregue o dataset antes de executar a limpeza.")

        features = self._features(self.raw_data)
        self.perturbed_datasets = []
        self.attempt_counter += 1
        self.preprocessador = None
        self.processed_data = None
        self.colunas_entrada = []
        self.candidatos_avaliados = {}
        self.variaveis_selecionadas = None
        try:
            preprocessador = construir(features)
            df = preprocessador.fit_transform(features, self.raw_data[self.target_column])
            if not isinstance(df, pd.DataFrame) or not df.index.equals(features.index) or df.shape[1] == 0:
                raise ValueError("O preparo deve devolver um DataFrame com variáveis e preservar o índice do treino.")
            df = df.copy()
            df.insert(len(df.columns), self.target_column, self.raw_data[self.target_column])
        except (ValueError, TypeError) as erro:
            return {
                "attempt": self.attempt_counter,
                "strategy": estrategia,
                "tests_passed": False,
                "report": f"Falha no pré-processamento: {erro}",
            }
        test_results = run_data_cleaning_tests(
            df,
            original_data_or_path=self.raw_data,
            linhas_para_duplicatas=self._linhas_originais,
        )

        self.preprocessador = preprocessador
        self.colunas_entrada = list(features.columns)
        self.processed_data = df
        return {
            "attempt": self.attempt_counter,
            "strategy": estrategia,
            "shape": df.shape,
            "tests_passed": test_results["all_passed"],
            "report": test_results["report"],
        }

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

    def medir_estabilidade_limpeza(self, agt: str) -> Tuple[float, Optional[str]]:
        """Mede a estabilidade da limpeza e relata problemas encontrados."""
        if len(self.perturbed_datasets) < 2:
            return 0.0, "A estabilidade exige ao menos duas perturbações."
        numeric_cols = [c for c in self.processed_data.select_dtypes(include=[np.number]).columns if c != self.target_column]
        if not numeric_cols:
            return 0.0, "A estabilidade exige ao menos uma variável numérica após a codificação."
        matrizes = [amostra[numeric_cols].corr().fillna(0).to_numpy() for amostra in self.perturbed_datasets]
        corr_diffs = [np.mean(np.abs(a - b)) for a, b in combinations(matrizes, 2)]
        # A diferença absoluta entre duas correlações varia de 0 a 2.
        score = 1.0 - float(np.mean(corr_diffs)) / 2.0

        if self._coluna_simulada in self.processed_data.columns:
            valores = self.processed_data[self._coluna_simulada]
            amplitude = valores.quantile(0.75) - valores.quantile(0.25)
            if valores.max() > valores.quantile(0.99) + 3 * amplitude:
                return score, f"Outlier em {self._coluna_simulada} após a limpeza."

        return score, None

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
            "tarefa": self.tarefa,
        }

    def treinar_e_avaliar_hipoteses(self, agt: str, plano: str, candidatos, selecionar=None) -> Dict[str, Any]:
        """Treina as hipóteses do tipo de tarefa e avalia as perturbações."""
        if self.processed_data is None or self.raw_test is None or self.preprocessador is None:
            raise ValueError("Dados processados indisponíveis para modelagem.")

        alvo = self.target_column
        treino = self.processed_data.drop(columns=alvo)
        validacao = self.preprocessador.transform(self._features(self.raw_test))
        perturbacoes = self.perturbed_datasets
        colunas = None
        if selecionar is not None:
            colunas = list(selecionar(self.tarefa, treino, self.processed_data[alvo]))
            if not colunas or len(set(colunas)) != len(colunas) or set(colunas) - set(treino.columns):
                raise ValueError("Selecione variáveis distintas e presentes no treino.")
            treino = treino[colunas]
            validacao = validacao[colunas]
            perturbacoes = [amostra[colunas + [alvo]] for amostra in perturbacoes]
        configuracoes = {nome: clone(modelo) for nome, modelo in candidatos.items()}
        resultado = avaliar_modelos(
            self.tarefa, treino, self.processed_data[alvo], validacao, self.raw_test[alvo],
            perturbacoes, alvo, candidatos=configuracoes, metrica_principal=self.metrica_principal,
        )
        self.candidatos_avaliados = configuracoes
        self.variaveis_selecionadas = colunas
        return {**resultado, "plano": plano, "variaveis": list(treino.columns)}
