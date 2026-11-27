"""Verifica a independência entre a seleção interna e os rótulos externos."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from tools.avaliacao_externa import pontuar_predicoes
from agents.model_agent import ModelAgent
from tools.preprocessing import criar_preprocessador


RAIZ = Path(__file__).resolve().parents[1]


class AvaliacaoExternaTest(unittest.TestCase):
    def executar(self, destino, *opcoes):
        processo = subprocess.run(
            [sys.executable, "-X", "utf8", "main.py", "--avaliacao-externa",
             "--saida-externa", str(destino), *opcoes],
            cwd=RAIZ, capture_output=True, encoding="utf8", timeout=120,
        )
        self.assertEqual(processo.returncode, 0, processo.stdout + processo.stderr)
        log = processo.stdout
        self.assertLess(log.index("Melhor Hipótese Selecionada"), log.index("predições gravadas"))
        self.assertLess(log.index("predições gravadas"), log.index("Lendo rótulos"))
        pasta, = destino.iterdir()
        relatorio = json.loads((pasta / "relatorio.json").read_text(encoding="utf8"))
        assinatura = hashlib.sha256((pasta / "predicoes.csv").read_bytes()).hexdigest()
        return relatorio, assinatura

    def test_inverter_rotulos_so_altera_metricas_externas(self):
        with TemporaryDirectory() as temp:
            pasta = Path(temp)
            original, assinatura = self.executar(pasta / "original")
            self.assertEqual(original["replanejamentos"], 0)
            selecao = original["selecao_interna"]
            self.assertEqual(selecao["criterio"], "roc_auc")
            pontuacoes = selecao["pontuacoes_por_modelo"]
            self.assertEqual(set(pontuacoes), {"RandomForest", "LogisticRegression"})
            self.assertEqual(original["modelo"], max(pontuacoes, key=pontuacoes.get))
            self.assertEqual(selecao["valor"], pontuacoes[original["modelo"]])
            rotulos = pd.read_csv(RAIZ / "data/bank_churn/test_eval.csv", usecols=["id", "Exited"])
            invertidos = pasta / "rotulos_invertidos.csv"
            rotulos["Exited"] = 1 - rotulos["Exited"]
            rotulos.sample(frac=1, random_state=42).to_csv(invertidos, index=False)
            controle, outra_assinatura = self.executar(
                pasta / "controle", "--metrica", "roc_auc", "--rotulos-externos", str(invertidos)
            )
            self.assertEqual(original["modelo"], controle["modelo"])
            self.assertEqual(original["selecao_interna"], controle["selecao_interna"])
            self.assertEqual(original["linhas_teste_externo"], len(rotulos))
            self.assertEqual(assinatura, outra_assinatura)
            for metrica in ("acuracia", "auc"):
                self.assertAlmostEqual(
                    controle["metricas_externas"][metrica], 1 - original["metricas_externas"][metrica]
                )

    def test_selecao_explicita_por_acuracia(self):
        with TemporaryDirectory() as temp:
            relatorio, _ = self.executar(Path(temp), "--metrica", "accuracy")
            selecao = relatorio["selecao_interna"]
            self.assertEqual(selecao["criterio"], "acuracia")
            self.assertEqual(selecao["valor"], selecao["acuracia"])
            self.assertEqual(selecao["valor"], max(selecao["pontuacoes_por_modelo"].values()))

    def test_replaneja_uma_vez_e_seleciona_apenas_entre_aprovados(self):
        with TemporaryDirectory() as temp:
            destino = Path(temp)
            relatorio, _ = self.executar(destino, "--limiar-estabilidade-modelo", "0.95")
            arquivo, = destino.glob("*/predicoes.csv")
            predicoes = pd.read_csv(arquivo, nrows=50).set_index("id")
        primeira, segunda = relatorio["historico_modelagem"]
        self.assertEqual(relatorio["replanejamentos"], 1)
        self.assertEqual(primeira["mensagem"], "replanejar_modelagem")
        self.assertIsNone(primeira["modelo_selecionado"])
        self.assertTrue(all(m["motivos"] and not m["aprovado"] for m in primeira["modelos"].values()))
        self.assertEqual(segunda["mensagem"], "certificacao_pcs_emitida")
        self.assertEqual(segunda["plano"], "selecao_de_variaveis")
        self.assertEqual(len(segunda["variaveis"]), len(primeira["variaveis"]) // 2)
        self.assertLess(set(segunda["variaveis"]), set(primeira["variaveis"]))
        self.assertEqual(primeira["limiar_estabilidade"], segunda["limiar_estabilidade"])
        floresta, logistica = segunda["modelos"]["RandomForest"], segunda["modelos"]["LogisticRegression"]
        self.assertGreater(floresta["metricas"]["roc_auc"], logistica["metricas"]["roc_auc"])
        self.assertFalse(floresta["aprovado"])
        self.assertTrue(logistica["aprovado"])
        self.assertEqual(relatorio["modelo"], "LogisticRegression")
        self.assertGreater(relatorio["linhas_teste_externo"], 0)
        dados = RAIZ / "data" / "bank_churn"
        treino = pd.read_csv(dados / "train.csv")
        teste = pd.read_csv(dados / "test.csv").set_index("id").loc[predicoes.index]
        entradas = treino.drop(columns=["id", "CustomerId", "Surname", "Exited"])
        preparo = criar_preprocessador(entradas, robusto=False)
        variaveis = segunda["variaveis"]
        modelo = ModelAgent.criar_candidatos("classificacao")[relatorio["modelo"]]
        modelo.fit(preparo.fit_transform(entradas)[variaveis], treino["Exited"])
        esperado = modelo.predict_proba(preparo.transform(teste[entradas.columns])[variaveis])[:, 1]
        np.testing.assert_allclose(predicoes["probabilidade_classe_positiva"], esperado)

    def test_certifica_o_estavel_sem_replanejar_quando_so_um_passa(self):
        with TemporaryDirectory() as temp:
            relatorio, _ = self.executar(Path(temp), "--metrica", "accuracy",
                                         "--limiar-estabilidade-modelo", "0.94")
        tentativa, = relatorio["historico_modelagem"]
        floresta = tentativa["modelos"]["RandomForest"]
        logistica = tentativa["modelos"]["LogisticRegression"]
        self.assertGreater(logistica["metricas"]["accuracy"], floresta["metricas"]["accuracy"])
        self.assertFalse(logistica["aprovado"])
        self.assertTrue(floresta["aprovado"])
        self.assertEqual(relatorio["modelo"], "RandomForest")
        self.assertEqual(relatorio["replanejamentos"], 0)

    def test_rejeita_ids_incompativeis(self):
        predicoes = pd.DataFrame({"id": ["1", "2"], "predicao": [0, 1]})
        with TemporaryDirectory() as temp:
            arquivo = Path(temp) / "rotulos.csv"
            for ids in ([1, 1], [1, 3], [1], [1, 2, 3]):
                with self.subTest(ids=ids):
                    pd.DataFrame({"id": ids, "Exited": [0] * len(ids)}).to_csv(arquivo, index=False)
                    with self.assertRaisesRegex(ValueError, "id"):
                        pontuar_predicoes(predicoes, arquivo, chave="id", alvo="Exited",
                                          tarefa="classificacao", classes=[0, 1])

    def test_avalia_classificacao_multiclasse_com_alvo_externo_distinto(self):
        with TemporaryDirectory() as temp:
            relatorio, _ = self.executar(
                Path(temp), "--dataset", "data/obesity_risks/train.csv",
                "--alvo", "NObeyesdad", "--alvo-externo", "True", "--limiar-estabilidade-modelo", "0.84",
            )
        metricas = relatorio["metricas_externas"]
        self.assertEqual(relatorio["alvo_externo"], "True")
        self.assertEqual(len(relatorio["classes"]), 7)
        self.assertEqual(relatorio["replanejamentos"], 1)
        self.assertEqual(relatorio["selecao_interna"]["plano"], "selecao_de_variaveis")
        self.assertNotIn("auc", metricas)
        self.assertAlmostEqual(metricas["nps"], sum(metricas[m] for m in ("acuracia", "precisao", "recall", "f1")) / 4)

    def test_avalia_regressao_com_chave_configurada(self):
        with TemporaryDirectory() as temp:
            relatorio, _ = self.executar(
                Path(temp), "--dataset", "data/ames_houses/train.csv",
                "--alvo", "SalePrice", "--tarefa", "regressao", "--id-coluna", "Order",
            )
            arquivo, = Path(temp).glob("*/predicoes.csv")
            self.assertEqual(pd.read_csv(arquivo).columns.tolist(), ["Order", "predicao"])
        self.assertEqual(relatorio["chave"], "Order")
        metricas = relatorio["metricas_externas"]
        self.assertGreater(metricas["rmse"], metricas["mae"])
        self.assertAlmostEqual(metricas["nps"], (1 / (1 + metricas["rmse"]) + 1 / (1 + metricas["mae"]) + metricas["r2"]) / 3)

    def test_avalia_csv_novo_com_arquivos_e_delimitador_informados(self):
        with TemporaryDirectory() as temp:
            pasta = Path(temp)
            dados = pd.DataFrame({
                "codigo": [f"{i:04d}" for i in range(120)],
                "sinal": list(range(60)) * 2,
                "classe": ["negativo"] * 30 + ["positivo"] * 30 + ["negativo"] * 30 + ["positivo"] * 30,
            })
            treino, teste = dados.iloc[:90], dados.iloc[90:].copy()
            teste.loc[:, "sinal"] = range(15, 45)
            teste.loc[:, "classe"] = ["negativo"] * 15 + ["positivo"] * 15
            treino.to_csv(pasta / "origem.csv", sep=";", index=False)
            teste.drop(columns="classe").to_csv(pasta / "reservado.csv", sep=";", index=False)
            teste[["codigo", "classe"]].rename(columns={"classe": "real"}).iloc[::-1].to_csv(
                pasta / "gabarito.csv", sep=";", index=False,
            )
            relatorio, _ = self.executar(
                pasta / "saida", "--dataset", str(pasta / "origem.csv"),
                "--teste-externo", str(pasta / "reservado.csv"),
                "--rotulos-externos", str(pasta / "gabarito.csv"), "--separador", ";",
                "--id-coluna", "codigo", "--alvo", "classe", "--alvo-externo", "real", "--metrica", "roc_auc",
            )
            arquivo, = (pasta / "saida").glob("*/predicoes.csv")
            predicoes = pd.read_csv(arquivo, dtype={"codigo": str})
        self.assertEqual(predicoes["codigo"].tolist(), teste["codigo"].tolist())
        self.assertEqual(relatorio["classe_positiva"], "positivo")
        self.assertGreater(relatorio["metricas_externas"]["auc"], 0.5)
