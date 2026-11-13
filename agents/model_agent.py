"""Agente BDI responsável pela construção, inferência e validação de modelos concorrentes."""
from typing import Any, Optional
from maspy import Agent, Belief, Goal, achieve, gain, pl, tell, Any as MaspyAny


class ModelAgent(Agent):
    """Agente BDI que treina hipóteses concorrentes e consolida predições após validação PCS."""

    def __init__(self, name: Optional[str] = "ModelAgent"):
        super().__init__(name=name)
        self.autorizacao_recebida: bool = False
        self.features_recebidas: bool = False
        self.modelagem_concluida: bool = False

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
        if self.autorizacao_recebida and self.features_recebidas and not self.modelagem_concluida:
            self.modelagem_concluida = True
            self.executar_modelagem_concorrente()

    def executar_modelagem_concorrente(self) -> None:
        print("\n[ModelAgent]: Condições BDI satisfeitas (Autorização PCS + Features Aprovadas).")
        print("[ModelAgent]: Iniciando treinamento de hipóteses concorrentes (RandomForest vs LogisticRegression)...")

        predictions_data = self.treinar_e_avaliar_hipoteses()

        rf = predictions_data["random_forest"]
        lr = predictions_data["logistic_regression"]
        print(f"[ModelAgent]: Treinamento finalizado em {predictions_data['elapsed_time_seconds']}s.")
        print(f"  • Random Forest: Acurácia={rf['accuracy']:.4f}, F1={rf['f1_score']:.4f}, Estabilidade={rf['perturbation_stability']:.4f}")
        print(f"  • Logistic Regression: Acurácia={lr['accuracy']:.4f}, F1={lr['f1_score']:.4f}, Estabilidade={lr['perturbation_stability']:.4f}")
        print(f"  • Concordância entre Hipóteses: {predictions_data['inter_model_agreement']:.4f}")

        print("[ModelAgent]: Submetendo predições perturbadas para auditoria final do PCSAgent...")
        self.send("PCSAgent", achieve, Goal("auditar_predicoes_modelos", predictions_data))

    @pl(gain, Belief("certificacao_pcs_emitida", MaspyAny))
    def entregar_relatorio_final(self, src: str, certificate: Any) -> None:
        print("\n" + "=" * 75)
        print("[ModelAgent -> Cientista de Dados]: RELATÓRIO ANALÍTICO CONSOLIDADO")
        print("=" * 75)
        print(f"Status da Certificação PCS: {certificate['status']}")
        print(f"Melhor Hipótese Selecionada: {certificate['best_hypothesis']}")
        print(f"P - Previsibilidade (Acurácia / F1): {certificate['predictability']['accuracy']:.4f} / {certificate['predictability']['f1_score']:.4f}")
        print(f"C - Computabilidade (Tempo de Execução): {certificate['computability']['execution_time_seconds']}s")
        print(f"S - Estabilidade (Bootstrap): {certificate['stability']['mean_perturbation_stability']:.4f}")
        print("=" * 75)
        print("[ModelAgent]: Ciclo deliberativo concluído com sucesso. Entregando resultados.")

        self.stop_cycle()
