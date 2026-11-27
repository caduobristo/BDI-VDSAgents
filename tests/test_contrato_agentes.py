"""Verifica alterações locais de cada agente pelo ponto de entrada do sistema."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from tools.preprocessing import criar_preprocessador


RAIZ = Path(__file__).resolve().parents[1]


class ContratoAgentesTest(unittest.TestCase):
    def executar(self, modulo, alteracao):
        with TemporaryDirectory() as temp:
            projeto = Path(temp)
            for pacote in ("agents", "environment", "tools"):
                shutil.copytree(RAIZ / pacote, projeto / pacote, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy2(RAIZ / "main.py", projeto)
            agente = projeto / "agents" / modulo
            agente.write_text(agente.read_text(encoding="utf8") + alteracao, encoding="utf8")
            destino = projeto / "resultados"
            processo = subprocess.run(
                [sys.executable, "-X", "utf8", str(projeto / "main.py"), "--avaliacao-externa",
                 "--saida-externa", str(destino)],
                cwd=RAIZ, capture_output=True, encoding="utf8", timeout=120,
            )
            self.assertEqual(processo.returncode, 0, processo.stdout + processo.stderr)
            pasta, = destino.iterdir()
            relatorio = json.loads((pasta / "relatorio.json").read_text(encoding="utf8"))
            predicoes = pd.read_csv(pasta / "predicoes.csv", nrows=50).set_index("id")
        return processo.stdout, relatorio, predicoes

    def test_explore_define_preparo_sem_alterar_o_ambiente(self):
        log, relatorio, _ = self.executar("explore_agent.py", """

def preparar_por_mediana(dados):
    return criar_preprocessador(dados.drop(columns="CreditScore"), robusto=False).set_params(
        numericas__strategy="median",
    )


ExploreAgent.PREPARO_PADRAO = ("mediana_sem_credit_score", preparar_por_mediana)
""")
        self.assertIn("mediana_sem_credit_score", log)
        tentativa, = relatorio["historico_modelagem"]
        self.assertNotIn("CreditScore", tentativa["variaveis"])
        self.assertIn("Balance", tentativa["variaveis"])
        self.assertGreater(relatorio["linhas_teste_externo"], 0)

    def test_model_adiciona_terceiro_candidato_e_reajusta_seus_parametros(self):
        _, relatorio, predicoes = self.executar("model_agent.py", """

candidatos_originais = ModelAgent.criar_candidatos


def criar_candidatos_da_equipe(tarefa):
    candidatos = candidatos_originais(tarefa)
    candidatos["RandomForest"].set_params(max_depth=2)
    candidatos["FlorestaDaEquipe"] = RandomForestClassifier(n_estimators=73, max_depth=6, random_state=42)
    return candidatos


ModelAgent.criar_candidatos = staticmethod(criar_candidatos_da_equipe)
""")
        pontuacoes = relatorio["selecao_interna"]["pontuacoes_por_modelo"]
        self.assertEqual(set(pontuacoes), {"RandomForest", "LogisticRegression", "FlorestaDaEquipe"})
        self.assertEqual(relatorio["modelo"], "FlorestaDaEquipe")

        dados = RAIZ / "data" / "bank_churn"
        treino = pd.read_csv(dados / "train.csv")
        teste = pd.read_csv(dados / "test.csv").set_index("id").loc[predicoes.index]
        entradas = treino.drop(columns=["id", "CustomerId", "Surname", "Exited"])
        preparo = criar_preprocessador(entradas, robusto=False)
        modelo = RandomForestClassifier(n_estimators=73, max_depth=6, random_state=42)
        modelo.fit(preparo.fit_transform(entradas), treino["Exited"])
        probabilidades = modelo.predict_proba(preparo.transform(teste[entradas.columns]))[:, 1]
        np.testing.assert_allclose(predicoes["probabilidade_classe_positiva"], probabilidades)

    def test_pcs_define_corte_e_escolha_sem_alterar_a_modelagem(self):
        _, relatorio, _ = self.executar("pcs_agent.py", """

def escolher_pela_acuracia(self, aprovados, criterio):
    return max(aprovados, key=lambda nome: aprovados[nome]["metricas"]["accuracy"], default=None)


PCSAgent.LIMIARES_PREVISIBILIDADE = {**PCSAgent.LIMIARES_PREVISIBILIDADE, "roc_auc": 0.80}
PCSAgent.escolher_modelo = escolher_pela_acuracia
""")
        tentativa, = relatorio["historico_modelagem"]
        self.assertEqual(tentativa["limiar_previsibilidade"], 0.80)
        self.assertTrue(all(modelo["aprovado"] for modelo in tentativa["modelos"].values()))
        self.assertGreater(tentativa["modelos"]["RandomForest"]["metricas"]["roc_auc"],
                           tentativa["modelos"]["LogisticRegression"]["metricas"]["roc_auc"])
        self.assertEqual(relatorio["modelo"], "LogisticRegression")
