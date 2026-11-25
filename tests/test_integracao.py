"""Exercita os três agentes MASPY pelo ponto de entrada do projeto."""
import subprocess
import sys
import unittest
from pathlib import Path


class FluxoAgentesTest(unittest.TestCase):
    def executar(self, *args):
        return subprocess.run(
            [sys.executable, "-X", "utf8", "main.py", *args],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            encoding="utf-8",
            timeout=60,
        )

    def verificar_entrega(self, resultado):
        log = resultado.stdout
        self.assertEqual(resultado.returncode, 0, log + resultado.stderr)
        treino = "Iniciando treinamento de hipóteses concorrentes"
        self.assertEqual(log.count(treino), 1, log)
        for evento in ("Autorização para modelagem recebida", "Matriz de features aprovadas recebida"):
            self.assertIn(evento, log)
            self.assertLess(log.index(evento), log.index(treino))
        self.assertEqual(log.count("Status da Certificação PCS:"), 1, log)
        self.assertLess(log.index(treino), log.index("Status da Certificação PCS:"))

    def test_aprovacao_direta(self):
        resultado = self.executar("--sem-simulacao")
        self.verificar_entrega(resultado)
        self.assertNotIn("Ativando Plano B", resultado.stdout)

    def test_recuperacao_pelo_plano_b(self):
        resultado = self.executar()
        self.verificar_entrega(resultado)
        self.assertEqual(resultado.stdout.count("Ativando Plano B"), 1)
        self.assertLess(resultado.stdout.index("Ativando Plano B"), resultado.stdout.index("Iniciando treinamento"))

    def test_reprovacao_das_duas_tentativas(self):
        resultado = self.executar("--limiar-estabilidade", "1")
        self.assertEqual(resultado.returncode, 1, resultado.stdout + resultado.stderr)
        self.assertIn("Pré-processamento reprovado", resultado.stderr)
        self.assertEqual(resultado.stdout.count("REJEITADO."), 2)
        self.assertNotIn("Iniciando treinamento", resultado.stdout)
        self.assertNotIn("Status da Certificação PCS:", resultado.stdout)

    def test_datasets_personalizados(self):
        casos = [
            (["--dataset", "data/online_shopping/train.csv", "--alvo", "Revenue"], "F1="),
            (["--dataset", "data/obesity_risks/train.csv", "--alvo", "NObeyesdad"], "F1="),
            (
                ["--dataset", "data/ames_houses/train.csv", "--alvo", "SalePrice",
                 "--tarefa", "regressao", "--ignorar", "Order"],
                "R²=",
            ),
        ]
        for args, metrica in casos:
            with self.subTest(dataset=args[1]):
                resultado = self.executar(*args)
                self.verificar_entrega(resultado)
                self.assertIn(metrica, resultado.stdout)

    def test_alvo_ausente_no_csv(self):
        resultado = self.executar("--dataset", "data/online_shopping/train.csv", "--alvo", "Inexistente")
        self.assertEqual(resultado.returncode, 1)
        self.assertIn("Coluna alvo 'Inexistente' ausente no CSV", resultado.stderr)
        self.assertNotIn("[ExploreAgent]", resultado.stdout)
