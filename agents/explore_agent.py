"""Agente BDI responsável pela exploração, limpeza e preparação dos dados."""
from functools import partial
from typing import Any, Optional
from maspy import Agent, Belief, Goal, achieve, gain, pl, tell, Any as MaspyAny
from tools.preprocessing import criar_preprocessador


class ExploreAgent(Agent):
    """Agente BDI que executa limpeza padrão (Plano A) ou contingência (Plano B)."""

    PREPARO_PADRAO = ("plano_a_padrao", partial(criar_preprocessador, robusto=False))
    PREPARO_CONTINGENCIA = ("plano_b_contingencia_knn_iqr", partial(criar_preprocessador, robusto=True))

    def __init__(self, name: Optional[str] = "ExploreAgent"):
        super().__init__(
            name=name,
            beliefs=[
                Belief("tentativa_atual", 1),
                Belief("dataset_pre_processado", "pendente"),
            ],
            goals=[
                Goal("dados_pre_processados_adequadamente"),
            ],
        )

    @pl(gain, Goal("dados_pre_processados_adequadamente"))
    def plano_limpeza_padrao(self, src: str) -> None:
        info_nulidades = self.verificar_nulidades()
        self.add(Belief("nulidade_variaveis", 1 if info_nulidades["has_nulls"] else 0))

        estrategia, construir = self.PREPARO_PADRAO
        print(f"\n[ExploreAgent]: Executando Plano A ({estrategia})...")
        res = self.aplicar_preparo(estrategia=estrategia, construir=construir)

        self.rm(Belief("dataset_pre_processado", MaspyAny))
        self.add(Belief("dataset_pre_processado", res))

        print("[ExploreAgent]: Plano A finalizado. Solicitando auditoria PCS...")
        self.send("PCSAgent", achieve, Goal("auditar_dados_pre_processados", res))

    @pl(gain, Goal("adotar_plano_alternativo", MaspyAny))
    def plano_contingencia(self, src: str, feedback: Any) -> None:
        print(f"\n[ExploreAgent]: Rejeição recebida de {src}. Feedback: {feedback}")
        estrategia, construir = self.PREPARO_CONTINGENCIA
        print(f"[ExploreAgent]: Ativando Plano B ({estrategia})...")

        # Atualiza a crença de tentativa para o ciclo de contingência
        self.rm(Belief("tentativa_atual", MaspyAny))
        self.add(Belief("tentativa_atual", 2))

        res = self.aplicar_preparo(estrategia=estrategia, construir=construir)

        self.rm(Belief("dataset_pre_processado", MaspyAny))
        self.add(Belief("dataset_pre_processado", res))

        print("[ExploreAgent]: Plano B finalizado. Reenviando dados pré-processados para auditoria PCS...")
        self.send("PCSAgent", achieve, Goal("auditar_dados_pre_processados", res))

    @pl(gain, Belief("pre_processamento_reprovado", MaspyAny))
    def encerrar_por_reprovacao(self, src: str, motivo: str) -> None:
        self.stop_cycle()

    @pl(gain, Belief("estabilidade_pcs_aprovada", MaspyAny))
    def plano_liberacao_features(self, src: str, status: Any) -> None:
        print("\n[ExploreAgent]: Estabilidade PCS aprovada! Executando Plano de Liberação...")
        features_info = self.obter_matriz_features_aprovadas()

        print(f"[ExploreAgent -> ModelAgent]: Enviando matriz_features_aprovadas ({features_info['n_features']} features).")
        self.send("ModelAgent", tell, Belief("matriz_features_aprovadas", features_info))
        self.stop_cycle()
