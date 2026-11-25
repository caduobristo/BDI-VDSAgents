"""Testes unitários puros para validação de integridade em etapas de ciência de dados."""
import os
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class DataCleaningTest:
    """Conjunto de testes unitários para a fase de limpeza e pré-processamento."""

    @staticmethod
    def _resolve_dataframe(data_or_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
        # Converte caminho ou objeto existente para DataFrame
        if isinstance(data_or_path, pd.DataFrame):
            return data_or_path
        if not os.path.exists(data_or_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {data_or_path}")
        return pd.read_csv(data_or_path)

    @staticmethod
    def test_file_readable(data_or_path: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """Testa se a fonte de dados pode ser lida com sucesso."""
        try:
            if isinstance(data_or_path, str):
                if not os.path.exists(data_or_path):
                    return {
                        "name": "File Readability Test",
                        "passed": False,
                        "message": f"Arquivo não encontrado: {data_or_path}",
                    }
                pd.read_csv(data_or_path, nrows=5)
            return {
                "name": "File Readability Test",
                "passed": True,
                "message": "Fonte de dados acessível e legível",
            }
        except Exception as e:
            return {
                "name": "File Readability Test",
                "passed": False,
                "message": f"Falha na leitura dos dados: {str(e)}",
            }

    @staticmethod
    def test_empty_dataset(data_or_path: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """Valida se o dataset possui linhas além do cabeçalho."""
        try:
            df = DataCleaningTest._resolve_dataframe(data_or_path)
            if len(df) == 0:
                return {
                    "name": "Empty Dataset Test",
                    "passed": False,
                    "message": "Dataset está vazio",
                }
            return {
                "name": "Empty Dataset Test",
                "passed": True,
                "message": f"Dataset contém {len(df)} registros",
            }
        except Exception as e:
            return {
                "name": "Empty Dataset Test",
                "passed": False,
                "message": f"Erro na execução do teste: {str(e)}",
            }

    @staticmethod
    def test_missing_values(data_or_path: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """Verifica a presença de valores nulos."""
        try:
            df = DataCleaningTest._resolve_dataframe(data_or_path)
            missing = df.isnull().sum()
            missing_cols = missing[missing > 0]

            if missing_cols.empty:
                return {
                    "name": "Missing Values Test",
                    "passed": True,
                    "message": "Nenhum valor ausente encontrado",
                }

            details = [f"{col}: {count} nulos" for col, count in missing_cols.items()]
            return {
                "name": "Missing Values Test",
                "passed": False,
                "message": f"Valores ausentes detectados: {', '.join(details)}",
            }
        except Exception as e:
            return {
                "name": "Missing Values Test",
                "passed": False,
                "message": f"Erro na execução do teste: {str(e)}",
            }

    @staticmethod
    def test_duplicated_features(data_or_path: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """Verifica se existem colunas duplicadas."""
        try:
            df = DataCleaningTest._resolve_dataframe(data_or_path)
            duplicated = df.columns[df.columns.duplicated()].tolist()

            if not duplicated:
                return {
                    "name": "Duplicate Features Test",
                    "passed": True,
                    "message": "Nenhuma feature duplicada encontrada",
                }
            return {
                "name": "Duplicate Features Test",
                "passed": False,
                "message": f"Features duplicadas: {', '.join(duplicated)}",
            }
        except Exception as e:
            return {
                "name": "Duplicate Features Test",
                "passed": False,
                "message": f"Erro na execução do teste: {str(e)}",
            }

    @staticmethod
    def test_duplicated_rows(data_or_path: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """Verifica se existem linhas completamente duplicadas."""
        try:
            df = DataCleaningTest._resolve_dataframe(data_or_path)
            count = df.duplicated().sum()

            if count == 0:
                return {
                    "name": "Duplicate Rows Test",
                    "passed": True,
                    "message": "Nenhuma linha duplicada encontrada",
                }
            return {
                "name": "Duplicate Rows Test",
                "passed": False,
                "message": f"{count} linhas duplicadas encontradas",
            }
        except Exception as e:
            return {
                "name": "Duplicate Rows Test",
                "passed": False,
                "message": f"Erro na execução do teste: {str(e)}",
            }

    @staticmethod
    def test_data_consistency(
        data_or_path: Union[str, pd.DataFrame],
        original_data_or_path: Union[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Verifica a consistência em relação ao dataset original."""
        try:
            cleaned = DataCleaningTest._resolve_dataframe(data_or_path)
            original = DataCleaningTest._resolve_dataframe(original_data_or_path)

            if len(cleaned) > len(original):
                return {
                    "name": "Data Consistency Test",
                    "passed": False,
                    "message": f"Dataset tratado ({len(cleaned)}) maior que original ({len(original)})",
                }

            common = set(original.columns) & set(cleaned.columns)
            if not common:
                return {
                    "name": "Data Consistency Test",
                    "passed": False,
                    "message": "Nenhuma coluna em comum com o dataset original",
                }

            return {
                "name": "Data Consistency Test",
                "passed": True,
                "message": f"Consistência validada com {len(common)} colunas correlatas",
            }
        except Exception as e:
            return {
                "name": "Data Consistency Test",
                "passed": False,
                "message": f"Erro na execução do teste: {str(e)}",
            }

    @staticmethod
    def test_data_retention(
        data_or_path: Union[str, pd.DataFrame],
        original_data_or_path: Union[str, pd.DataFrame],
        min_ratio: float = 80.0,
    ) -> Dict[str, Any]:
        """Verifica se a taxa de retenção de amostras é adequada."""
        try:
            cleaned = DataCleaningTest._resolve_dataframe(data_or_path)
            original = DataCleaningTest._resolve_dataframe(original_data_or_path)

            ratio = (len(cleaned) / len(original)) * 100.0
            if ratio >= min_ratio:
                return {
                    "name": "Data Retention Test",
                    "passed": True,
                    "message": f"Taxa de retenção: {ratio:.2f}% (limiar: >={min_ratio}%)",
                }
            return {
                "name": "Data Retention Test",
                "passed": False,
                "message": f"Taxa de retenção insuficiente: {ratio:.2f}% < {min_ratio}%",
            }
        except Exception as e:
            return {
                "name": "Data Retention Test",
                "passed": False,
                "message": f"Erro na execução do teste: {str(e)}",
            }

    @staticmethod
    def test_numeric_conversion(data_or_path: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """Verifica se todas as variáveis estão em formato numérico para modelagem."""
        try:
            df = DataCleaningTest._resolve_dataframe(data_or_path)
            non_numeric = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()

            if not non_numeric:
                return {
                    "name": "Numeric Conversion Test",
                    "passed": True,
                    "message": "Todas as variáveis são numéricas",
                }
            return {
                "name": "Numeric Conversion Test",
                "passed": False,
                "message": f"Variáveis não-numéricas remanescentes: {', '.join(non_numeric)}",
            }
        except Exception as e:
            return {
                "name": "Numeric Conversion Test",
                "passed": False,
                "message": f"Erro na execução do teste: {str(e)}",
            }


def run_data_cleaning_tests(
    data_or_path: Union[str, pd.DataFrame],
    original_data_or_path: Optional[Union[str, pd.DataFrame]] = None,
    require_all_numeric: bool = False,
    *,
    linhas_para_duplicatas: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Executa a bateria de testes de limpeza e retorna relatório estruturado."""
    tests = [
        DataCleaningTest.test_file_readable(data_or_path),
        DataCleaningTest.test_empty_dataset(data_or_path),
        DataCleaningTest.test_missing_values(data_or_path),
        DataCleaningTest.test_duplicated_features(data_or_path),
        DataCleaningTest.test_duplicated_rows(
            data_or_path if linhas_para_duplicatas is None else linhas_para_duplicatas
        ),
    ]

    if original_data_or_path is not None:
        tests.append(DataCleaningTest.test_data_consistency(data_or_path, original_data_or_path))
        tests.append(DataCleaningTest.test_data_retention(data_or_path, original_data_or_path))

    if require_all_numeric:
        tests.append(DataCleaningTest.test_numeric_conversion(data_or_path))

    passed = sum(1 for t in tests if t["passed"])
    failed = len(tests) - passed
    all_passed = failed == 0

    report_lines = [
        f"Relatório de Testes de Limpeza: {passed} aprovados, {failed} reprovados.",
    ]
    for t in tests:
        mark = "[PASS]" if t["passed"] else "[FAIL]"
        report_lines.append(f"  {mark} {t['name']}: {t['message']}")

    return {
        "tests": tests,
        "passed": passed,
        "failed": failed,
        "all_passed": all_passed,
        "report": "\n".join(report_lines),
    }
