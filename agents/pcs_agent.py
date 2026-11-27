"""Agente BDI responsável pela auditoria e certificação dos princípios PCS."""
from statistics import mean
from typing import Any, Optional
from maspy import Agent, Belief, Goal, achieve, gain, pl, tell, Any as MaspyAny
from tools.modeling import NOMES_METRICAS


class PCSAgent(Agent):
    """Agente supervisor que audita estabilidade de dados e predições sob os princípios PCS."""

    LIMIARES_PREVISIBILIDADE = {"accuracy": 0.70, "roc_auc": 0.50, "r2": 0.0}
    TEMPO_MAXIMO_MODELO = 30.0

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
            score, problema = self.medir_estabilidade_limpeza()
            passed = problema is None and score >= limiar
            details = problema or f"Estabilidade bootstrap {score:.4f}, limiar {limiar:.4f}."
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

    def escolher_modelo(self, aprovados, criterio):
        return max(aprovados, key=lambda nome: aprovados[nome]["metricas"][criterio], default=None)

    def auditar_predicoes_pcs(self, predictions_data, limiar_estabilidade):
        """Emite o certificado final PCS com base nas métricas da tarefa."""
        modelos = predictions_data["models"]
        criterio = predictions_data["primary_metric"]
        limiar_previsibilidade = self.LIMIARES_PREVISIBILIDADE[criterio]
        computability_passed = predictions_data["elapsed_time_seconds"] < self.TEMPO_MAXIMO_MODELO
        auditorias = {}
        for nome, dados in modelos.items():
            valor = dados["metricas"][criterio]
            estabilidade = dados["perturbation_stability"]
            motivos = []
            if not valor > limiar_previsibilidade:
                motivos.append(f"{criterio}={valor:.4f} precisa superar {limiar_previsibilidade:.2f}")
            if not estabilidade >= limiar_estabilidade:
                motivos.append(f"estabilidade abaixo do limiar {limiar_estabilidade:g}")
            if not computability_passed:
                motivos.append(f"tempo da tentativa atingiu o limite de {self.TEMPO_MAXIMO_MODELO:g} s")
            auditorias[nome] = {
                "metricas": dados["metricas"], "estabilidade": estabilidade,
                "aprovado": not motivos, "motivos": motivos,
            }

        aprovados = {nome: auditoria for nome, auditoria in auditorias.items() if auditoria["aprovado"]}
        melhor = self.escolher_modelo(aprovados, criterio)
        mean_stability = mean(m["perturbation_stability"] for m in modelos.values())
        concordancia = predictions_data["inter_model_agreement"]
        return {
            "status": "CERTIFICADO" if melhor is not None else "REPROVADO",
            "best_hypothesis": melhor,
            "auditoria": {
                "plano": predictions_data["plano"], "variaveis": predictions_data["variaveis"],
                "criterio": criterio, "limiar_previsibilidade": limiar_previsibilidade,
                "limiar_estabilidade": limiar_estabilidade, "modelos": auditorias,
                "tempo_segundos": predictions_data["elapsed_time_seconds"],
                "modelo_selecionado": melhor,
            },
            "primary_metric": criterio,
            "validation_scores": {nome: dados["metricas"][criterio] for nome, dados in modelos.items()},
            "predictability": {
                **(modelos[melhor]["metricas"] if melhor is not None else {}),
                "verified": melhor is not None,
            },
            "computability": {
                "execution_time_seconds": round(predictions_data["elapsed_time_seconds"], 3),
                "verified": computability_passed,
            },
            "stability": {
                "mean_perturbation_stability": round(mean_stability, 4),
                "selected_perturbation_stability": modelos[melhor]["perturbation_stability"] if melhor is not None else None,
                "inter_model_agreement": round(concordancia, 4) if concordancia is not None else None,
                "verified": melhor is not None,
            },
        }

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
