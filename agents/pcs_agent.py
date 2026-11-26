"""Agente BDI responsável pela auditoria e certificação dos princípios PCS."""
from typing import Any, Optional
from maspy import Agent, Belief, Goal, achieve, gain, pl, tell, Any as MaspyAny
from tools.modeling import NOMES_METRICAS


class PCSAgent(Agent):
    """Agente supervisor que audita estabilidade de dados e predições sob os princípios PCS."""

    def __init__(self, name: Optional[str] = "PCSAgent", limiar_estabilidade: float = 0.75,
                 limiar_estabilidade_modelo: float = 0.80):
        super().__init__(
            name=name,
            beliefs=[
                Belief("limiar_estabilidade", limiar_estabilidade),
                Belief("limiar_estabilidade_modelo", limiar_estabilidade_modelo),
            ],
        )
        self.historico_modelagem = []

    @pl(gain, Goal("auditar_dados_pre_processados", MaspyAny))
    def plano_auditar_pre_processamento(self, src: str, data_meta: Any) -> None:
        print(f"\n[PCSAgent]: Iniciando auditoria PCS sobre dados pré-processados recebidos de {src}...")

        if data_meta["tests_passed"]:
            self.gerar_perturbacoes_bootstrap(k=3)
            limiar_bel = self.get(Belief("limiar_estabilidade", MaspyAny), ck_src=False)
            limiar = limiar_bel.values if limiar_bel is not None else 0.75
            _, passed, details = self.calcular_estabilidade_pcs(limiar=limiar)
        else:
            passed, details = False, data_meta["report"]

        if not passed:
            print(f"[PCSAgent -> ExploreAgent]: REJEITADO. {details}")
            if data_meta["attempt"] == 1:
                self.send("ExploreAgent", achieve, Goal("adotar_plano_alternativo", details))
            else:
                for destino in ("ExploreAgent", "ModelAgent"):
                    self.send(destino, tell, Belief("pre_processamento_reprovado", details))
                self.stop_cycle()
        else:
            print(f"[PCSAgent -> ExploreAgent]: APROVADO. {details}")
            print(f"[PCSAgent -> ModelAgent]: APROVADO. Autorização para Modelagem concedida.")
            self.send("ExploreAgent", tell, Belief("estabilidade_pcs_aprovada", "aprovado"))
            self.send("ModelAgent", achieve, Goal("autorizacao_para_modelagem", "concedida"))

    @pl(gain, Goal("auditar_predicoes_modelos", MaspyAny))
    def plano_auditar_predicoes_modelos(self, src: str, predictions_data: Any) -> None:
        print(f"\n[PCSAgent]: Auditando variância final e estabilidade preditiva recebidas de {src}...")

        limiar = self.get(Belief("limiar_estabilidade_modelo", MaspyAny), ck_src=False).values
        certificate = self.auditar_predicoes_pcs(predictions_data, limiar_estabilidade=limiar)
        tentativa = certificate["auditoria"]
        criterio = tentativa["criterio"]
        for nome, resultado in tentativa["modelos"].items():
            decisao = "APROVADO" if resultado["aprovado"] else "REPROVADO"
            motivo = " | ".join(resultado["motivos"])
            print(f"[PCSAgent] {nome} {decisao} | {NOMES_METRICAS[criterio]}={resultado['metricas'][criterio]:.4f} "
                  f"| estabilidade={resultado['estabilidade']:.4f}" + (f" | {motivo}" if motivo else ""))

        sem_modelo = certificate["best_hypothesis"] is None
        replanejar = sem_modelo and predictions_data["pode_replanejar"]
        tentativa["tentativa"] = len(self.historico_modelagem) + 1
        tentativa["crenca"] = "nenhum_modelo_aprovado" if sem_modelo else "modelo_aprovado"
        tentativa["mensagem"] = "replanejar_modelagem" if replanejar else "certificacao_pcs_emitida"
        self.historico_modelagem.append(tentativa)
        if replanejar:
            print("[PCSAgent -> ModelAgent] Nenhum candidato aprovado. Solicitando replanejamento.")
            self.send("ModelAgent", achieve, Goal("replanejar_modelagem", tentativa))
            return

        certificate["historico_modelagem"] = self.historico_modelagem.copy()
        print(f"[PCSAgent] Certificação final: {certificate['status']}.")
        self.send("ModelAgent", tell, Belief("certificacao_pcs_emitida", certificate))
        self.stop_cycle()
