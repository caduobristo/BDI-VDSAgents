"""Verifica a agregação e o aceite de resultados do benchmark."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.benchmark import resumir_base, validar_saida


class BenchmarkTest(unittest.TestCase):
    def test_calcula_indicadores_com_falhas_e_cinco_saidas_validas(self):
        tentativas = [{"nps": valor} for valor in (0.2, None, 0.4, 0.6, None, 0.8, 1.0)]
        resultado = resumir_base(tentativas)
        self.assertTrue(resultado["completa"])
        self.assertEqual(resultado["validas"], 5)
        self.assertEqual(resultado["falhas"], 2)
        self.assertAlmostEqual(resultado["vs"], 5 / 7)
        self.assertAlmostEqual(resultado["anps"], 0.6)
        self.assertAlmostEqual(resultado["desvio_padrao_nps"], 0.08 ** 0.5)
        self.assertAlmostEqual(resultado["cs"], (5 / 7 + 0.6) / 2)

    def test_preserva_vs_observado_quando_a_meta_nao_e_atingida(self):
        resultado = resumir_base([{"nps": 0.6}, {"nps": None}])
        self.assertFalse(resultado["completa"])
        self.assertEqual(resultado["vs"], 0.5)
        for campo in ("anps", "desvio_padrao_nps", "cs"):
            self.assertIsNone(resultado[campo])

    def test_valida_o_nps_e_a_cobertura_das_predicoes(self):
        with TemporaryDirectory() as temp:
            relatorio = Path(temp) / "relatorio.json"
            dados = {"chave": "id", "linhas_teste_externo": 2, "metricas_externas": {"nps": 0.0}}
            relatorio.write_text(json.dumps(dados), encoding="utf-8")
            predicoes = relatorio.with_name("predicoes.csv")
            predicoes.write_text("id,predicao\n01,0\n02,1\n", encoding="utf-8")
            self.assertEqual(validar_saida(relatorio), 0.0)

            predicoes.write_text("id,predicao\n01,0\n01,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identificadores repetidos"):
                validar_saida(relatorio)

            dados["metricas_externas"]["nps"] = float("nan")
            relatorio.write_text(json.dumps(dados), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NPS finito"):
                validar_saida(relatorio)
