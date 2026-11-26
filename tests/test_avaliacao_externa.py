"""Verifica a independência entre a seleção interna e os rótulos externos."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from tools.avaliacao_externa import pontuar_bank_churn


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

    def test_rejeita_ids_incompativeis(self):
        predicoes = pd.DataFrame({"id": [1, 2], "predicao": [0, 1], "probabilidade_churn": [0.1, 0.9]})
        with TemporaryDirectory() as temp:
            arquivo = Path(temp) / "rotulos.csv"
            for ids in ([1, 1], [1, 3], [1], [1, 2, 3]):
                with self.subTest(ids=ids):
                    pd.DataFrame({"id": ids, "Exited": [0] * len(ids)}).to_csv(arquivo, index=False)
                    with self.assertRaisesRegex(ValueError, "id"):
                        pontuar_bank_churn(predicoes, arquivo)
