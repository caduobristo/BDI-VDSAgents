"""Agente BDI responsável pela auditoria e certificação dos princípios PCS."""
from typing import Any, Optional
from maspy import Agent, Belief, Goal, achieve, gain, pl, tell, Any as MaspyAny


class PCSAgent(Agent):
    """Agente supervisor que audita estabilidade de dados e predições sob os princípios PCS."""

    def __init__(self, name: Optional[str] = "PCSAgent", limiar_estabilidade: float = 0.75):
        super().__init__(
            name=name,
            beliefs=[
                Belief("limiar_estabilidade", limiar_estabilidade),
            ],
        )

    @pl(gain, Goal("auditar_dados_pre_processados", MaspyAny))
    def plano_auditar_pre_processamento(self, src: str, data_meta: Any) -> None:
        print(f"\n[PCSAgent]: Iniciando auditoria PCS sobre dados pré-processados recebidos de {src}...")

        # Gera k=3 amostras perturbadas via bootstrap no ambiente
        self.gerar_perturbacoes_bootstrap(k=3)

        limiar_bel = self.get(Belief("limiar_estabilidade", MaspyAny), ck_src=False)
        limiar = limiar_bel.values if limiar_bel is not None else 0.75

        score, passed, details = self.calcular_estabilidade_pcs(limiar=limiar)

        if not passed:
            print(f"[PCSAgent -> ExploreAgent]: REJEITADO (Score {score:.2f} < Limiar {limiar:.2f})")
            print(f"[PCSAgent]: Detalhes: {details}")
            self.send("ExploreAgent", achieve, Goal("adotar_plano_alternativo", details))
        else:
            print(f"[PCSAgent -> ExploreAgent]: APROVADO (Score {score:.2f} >= Limiar {limiar:.2f})")
            print(f"[PCSAgent -> ModelAgent]: APROVADO. Autorização para Modelagem concedida.")
            self.send("ExploreAgent", tell, Belief("estabilidade_pcs_aprovada", "aprovado"))
            self.send("ModelAgent", achieve, Goal("autorizacao_para_modelagem", "concedida"))

    @pl(gain, Goal("auditar_predicoes_modelos", MaspyAny))
    def plano_auditar_predicoes_modelos(self, src: str, predictions_data: Any) -> None:
        print(f"\n[PCSAgent]: Auditando variância final e estabilidade preditiva recebidas de {src}...")

        certificate = self.auditar_predicoes_pcs(predictions_data)

        print(f"[PCSAgent]: Emitindo Certificado PCS - Status: {certificate['status']}")
        print(f"  • Predictability: Acurácia={certificate['predictability']['accuracy']:.4f}, F1={certificate['predictability']['f1_score']:.4f} (Validado: {certificate['predictability']['verified']})")
        print(f"  • Computability: Tempo={certificate['computability']['execution_time_seconds']}s (Validado: {certificate['computability']['verified']})")
        print(f"  • Stability: Estabilidade Bootstrap={certificate['stability']['mean_perturbation_stability']:.4f}, Concordância={certificate['stability']['inter_model_agreement']:.4f} (Validado: {certificate['stability']['verified']})")

        self.send("ModelAgent", tell, Belief("certificacao_pcs_emitida", certificate))
        self.stop_cycle()
