"""Agente BDI responsável pela construção, inferência e validação de modelos concorrentes."""
from typing import Any, Optional
from maspy import Agent, Belief, Goal, achieve, gain, pl, tell, Any as MaspyAny
from tools.modeling import formatar_metricas


class ModelAgent(Agent):
    """Agente BDI que treina hipóteses concorrentes e consolida predições após validação PCS."""

    def __init__(self, name: Optional[str] = "ModelAgent"):
        # As duas mensagens de liberação podem chegar no mesmo ciclo.
        super().__init__(name=name, max_intentions=1)
        self.autorizacao_recebida: bool = False
        self.features_recebidas: bool = False
        self.modelagem_iniciada = False
        self.certificado_final = None
        self.motivo_reprovacao = None

    @pl(gain, Goal("autorizacao_para_modelagem", MaspyAny))
    def ao_receber_autorizacao(self, src: str, payload: Any) -> None:
        print(f"\n[ModelAgent]: Autorização para modelagem recebida de {src}.")
        self.autorizacao_recebida = True
        self.verificar_e_executar_modelagem()

    @pl(gain, Belief("matriz_features_aprovadas", MaspyAny))
    def ao_receber_matriz_features(self, src: str, features_info: Any) -> None:
        print(f"\n[ModelAgent]: Matriz de features aprovadas recebida de {src}.")
        self.features_recebidas = True
        self.verificar_e_executar_modelagem()

    def verificar_e_executar_modelagem(self) -> None:
        # Só inicia o treino quando possui autorização para modelagem E matriz de features aprovadas
        if self.autorizacao_recebida and self.features_recebidas and not self.modelagem_iniciada:
            self.modelagem_iniciada = True
            self.executar_modelagem_concorrente()

    def executar_modelagem_concorrente(self) -> None:
        print("\n[ModelAgent]: Condições BDI satisfeitas (Autorização PCS + Features Aprovadas).")
        print("[ModelAgent]: Iniciando treinamento de hipóteses concorrentes...")

        predictions_data = self.treinar_e_avaliar_hipoteses()

        print(f"[ModelAgent]: Treinamento finalizado em {predictions_data['elapsed_time_seconds']:.3f}s.")
        for nome, resultado in predictions_data["models"].items():
            metricas = formatar_metricas(resultado["metricas"])
            print(f"  {nome}: {metricas}, estabilidade={resultado['perturbation_stability']:.4f}")
        print(f"  • Concordância entre Hipóteses: {predictions_data['inter_model_agreement']:.4f}")

        print("[ModelAgent]: Submetendo predições perturbadas para auditoria final do PCSAgent...")
        self.send("PCSAgent", achieve, Goal("auditar_predicoes_modelos", predictions_data))

    @pl(gain, Belief("pre_processamento_reprovado", MaspyAny))
    def encerrar_por_reprovacao(self, src: str, motivo: str) -> None:
        self.motivo_reprovacao = motivo
        self.stop_cycle()

    @pl(gain, Belief("certificacao_pcs_emitida", MaspyAny))
    def entregar_relatorio_final(self, src: str, certificate: Any) -> None:
        print("\n" + "=" * 75)
        print("[ModelAgent -> Cientista de Dados]: RELATÓRIO ANALÍTICO CONSOLIDADO")
        print("=" * 75)
        print(f"Status da Certificação PCS: {certificate['status']}")
        print(f"Melhor Hipótese Selecionada: {certificate['best_hypothesis']}")
        metricas = {chave: valor for chave, valor in certificate["predictability"].items() if chave != "verified"}
        print(f"P - Previsibilidade: {formatar_metricas(metricas)}")
        print(f"C - Computabilidade (Tempo de Execução): {certificate['computability']['execution_time_seconds']}s")
        print(f"S - Estabilidade (Bootstrap): {certificate['stability']['mean_perturbation_stability']:.4f}")
        print("=" * 75)
        print("[ModelAgent]: Ciclo deliberativo concluído com sucesso. Entregando resultados.")
        self.certificado_final = certificate
        self.stop_cycle()
