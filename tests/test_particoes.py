"""Verifica a origem das linhas e o isolamento do teste nos dois planos."""
import unittest

import pandas as pd

from environment.ds_environment import DataScienceEnv
from tools.modeling import PLANO_REDUZIDO


class ParticoesTest(unittest.TestCase):
    def test_plano_reduzido_seleciona_variaveis_somente_no_treino(self):
        env = DataScienceEnv("TesteSelecao", alvo="Exited", ignorar=["id", "CustomerId", "Surname"])
        env.carregar_dataset("ExploreAgent", "data/bank_churn/train.csv")
        env.executar_limpeza_padrao("ExploreAgent")
        env.gerar_perturbacoes_bootstrap("PCSAgent")
        inicial = env.treinar_e_avaliar_hipoteses("ModelAgent", PLANO_REDUZIDO)
        env.raw_test["Age"] = 0
        env.raw_test["Exited"] = 1 - env.raw_test["Exited"]
        alterado = env.treinar_e_avaliar_hipoteses("ModelAgent", PLANO_REDUZIDO)
        self.assertEqual(inicial["variaveis"], alterado["variaveis"])

    def test_teste_reservado_nos_dois_planos(self):
        env = DataScienceEnv("TesteParticoes", alvo="Exited", ignorar=["id", "CustomerId", "Surname"])
        env.carregar_dataset("ExploreAgent", "data/bank_churn/train.csv")
        treino, teste = env.raw_data.index, env.raw_test.index
        self.assertEqual((len(treino), len(teste)), (750, 250))
        self.assertTrue(treino.intersection(teste).empty)
        original = env.raw_test.copy()

        for limpar in (env.executar_limpeza_padrao, env.executar_limpeza_robusta):
            with self.subTest(plano=limpar.__name__):
                env.raw_test = original.copy()
                limpar("ExploreAgent")
                esperado = env.processed_data.copy()
                self.assertTrue(esperado.index.equals(treino))

                # Valores presentes apenas no teste não podem alterar o treino.
                env.raw_test["Balance"] = 1e12
                env.raw_test["Geography"] = "Categoria exclusiva do teste"
                limpar("ExploreAgent")
                pd.testing.assert_frame_equal(esperado, env.processed_data)
                self.assertFalse(any("exclusiva" in col for col in esperado.columns))

                env.gerar_perturbacoes_bootstrap("PCSAgent")
                self.assertEqual(len(env.perturbed_datasets), 3)
                for amostra in env.perturbed_datasets:
                    self.assertTrue(amostra.index.isin(treino).all())
                    self.assertTrue(amostra.index.intersection(teste).empty)

                resultado = env.treinar_e_avaliar_hipoteses("ModelAgent")
                self.assertEqual(resultado["test_sample_size"], len(teste))
