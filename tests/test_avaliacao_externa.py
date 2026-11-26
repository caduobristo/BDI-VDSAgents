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

from tools.avaliacao_externa import pontuar_bank_churn
from tools.modeling import criar_hipoteses
from tools.preprocessing import criar_preprocessador


RAIZ = Path(__file__).resolve().parents[1]


class AvaliacaoExternaTest(unittest.TestCase):
    def executar(self, destino, *opcoes):
        processo = subprocess.run(
            [sys.executable, "-X", "utf8", "main.py", "--sem-simulacao",
             "--avaliacao-externa", "--saida-externa", str(destino), *opcoes],
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
        modelo = criar_hipoteses("classificacao")[relatorio["modelo"]]
        modelo.fit(preparo.fit_transform(entradas)[variaveis], treino["Exited"])
        esperado = modelo.predict_proba(preparo.transform(teste[entradas.columns])[variaveis])[:, 1]
        np.testing.assert_allclose(predicoes["probabilidade_churn"], esperado)

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
        predicoes = pd.DataFrame({"id": [1, 2], "predicao": [0, 1], "probabilidade_churn": [0.1, 0.9]})
        with TemporaryDirectory() as temp:
            arquivo = Path(temp) / "rotulos.csv"
            for ids in ([1, 1], [1, 3], [1], [1, 2, 3]):
                with self.subTest(ids=ids):
                    pd.DataFrame({"id": ids, "Exited": [0] * len(ids)}).to_csv(arquivo, index=False)
                    with self.assertRaisesRegex(ValueError, "id"):
                        pontuar_bank_churn(predicoes, arquivo)
