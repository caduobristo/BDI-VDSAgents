"""Agente BDI responsável pela construção, inferência e validação de modelos concorrentes."""
from typing import Any, Optional
from maspy import Agent, Belief, Goal, achieve, gain, pl, tell, Any as MaspyAny
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tools.modeling import formatar_metricas


PLANO_COMPLETO = "todas_as_variaveis"
PLANO_REDUZIDO = "selecao_de_variaveis"


class ModelAgent(Agent):
    """Agente BDI que treina hipóteses concorrentes e consolida predições após validação PCS."""

    def __init__(self, name: Optional[str] = "ModelAgent"):
        # As duas mensagens de liberação podem chegar no mesmo ciclo.
        super().__init__(name=name, max_intentions=1)
        self.autorizacao_recebida: bool = False
        self.features_recebidas: bool = False
        self.modelagem_iniciada = False
        self.plano = PLANO_COMPLETO
        self.tarefa = None
        self.permite_replanejamento = False
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
        self.tarefa = features_info["tarefa"]
        self.permite_replanejamento = features_info["n_features"] > 1
        self.verificar_e_executar_modelagem()

    def verificar_e_executar_modelagem(self) -> None:
        # Só inicia o treino quando possui autorização para modelagem E matriz de features aprovadas
        if self.autorizacao_recebida and self.features_recebidas and not self.modelagem_iniciada:
            self.modelagem_iniciada = True
            print("\n[ModelAgent]: Condições BDI satisfeitas (Autorização PCS + Features Aprovadas).")
            self.executar_modelagem_concorrente()

    @staticmethod
    def criar_candidatos(tarefa):
        if tarefa == "regressao":
            floresta = RandomForestRegressor
            linear = Ridge()
        else:
            floresta = RandomForestClassifier
            linear = LogisticRegression(max_iter=1000, random_state=42)
        return {
            "RandomForest": floresta(n_estimators=50, max_depth=6, random_state=42),
            type(linear).__name__: make_pipeline(StandardScaler(), linear),
        }

    @staticmethod
    def selecionar_variaveis(tarefa, treino, alvo):
        pontuar = f_regression if tarefa == "regressao" else f_classif
        seletor = SelectKBest(pontuar, k=treino.shape[1] // 2).fit(treino, alvo)
        return list(seletor.get_feature_names_out())

    def executar_modelagem_concorrente(self) -> None:
        print(f"[ModelAgent]: Iniciando treinamento de hipóteses concorrentes. Plano: {self.plano}.")

        seletor = self.selecionar_variaveis if self.plano == PLANO_REDUZIDO else None
        predictions_data = self.treinar_e_avaliar_hipoteses(
            plano=self.plano, candidatos=self.criar_candidatos(self.tarefa), selecionar=seletor,
        )
        predictions_data["pode_replanejar"] = self.permite_replanejamento and self.plano == PLANO_COMPLETO
        print(f"[ModelAgent] Variáveis da tentativa: {', '.join(predictions_data['variaveis'])}.")

        print(f"[ModelAgent]: Treinamento finalizado em {predictions_data['elapsed_time_seconds']:.3f}s.")
        for nome, resultado in predictions_data["models"].items():
            metricas = formatar_metricas(resultado["metricas"])
            print(f"  {nome}: {metricas}, estabilidade={resultado['perturbation_stability']:.4f}")
        concordancia = predictions_data["inter_model_agreement"]
        if concordancia is not None:
            print(f"  • Concordância entre Hipóteses: {concordancia:.4f}")

        print("[ModelAgent]: Submetendo predições perturbadas para auditoria final do PCSAgent...")
        self.send("PCSAgent", achieve, Goal("auditar_predicoes_modelos", predictions_data))

    @pl(gain, Goal("replanejar_modelagem", MaspyAny))
    def replanejar_modelagem(self, src: str, auditoria: Any) -> None:
        if self.plano != PLANO_COMPLETO or not self.permite_replanejamento:
            return
        self.add(Belief("nenhum_modelo_aprovado", auditoria))
        self.plano = PLANO_REDUZIDO
        print(f"[ModelAgent] Troca de plano: {auditoria['plano']} -> {self.plano}.")
        self.executar_modelagem_concorrente()

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
        self.certificado_final = certificate
        if certificate["best_hypothesis"] is None:
            print("[ModelAgent] Planos esgotados. Encerrando sem modelo aprovado.")
            self.stop_cycle()
            return
        print(f"Melhor Hipótese Selecionada: {certificate['best_hypothesis']}")
        metricas = {chave: valor for chave, valor in certificate["predictability"].items() if chave != "verified"}
        print(f"P - Previsibilidade: {formatar_metricas(metricas)}")
        print(f"C - Computabilidade (Tempo de Execução): {certificate['computability']['execution_time_seconds']}s")
        print(f"S - Estabilidade do modelo selecionado: {certificate['stability']['selected_perturbation_stability']:.4f}")
        print("=" * 75)
        print("[ModelAgent]: Ciclo deliberativo concluído com sucesso. Entregando resultados.")
        self.stop_cycle()
