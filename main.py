"""Ponto de entrada principal para execução do sistema deliberativo VDS-BDI em MASPY."""
from maspy import Admin
from environment.ds_environment import DataScienceEnv
from agents.explore_agent import ExploreAgent
from agents.pcs_agent import PCSAgent
from agents.model_agent import ModelAgent


def main():
    print("=" * 75)
    print(" SISTEMA MULTIAGENTE DELIBERATIVO VDS-BDI (Predictability, Computability, Stability)")
    print(" Framework: MASPY | Domínio: Bank Churn (Data Science Life Cycle)")
    print("=" * 75)

    admin = Admin()
    # Desativa rastreamento verboso de baixo nível para manter logs limpos e legíveis
    admin.console_settings(
        show_exec=False,
        show_cycle=False,
        show_prct=False,
        show_slct=False,
    )

    env = DataScienceEnv("DataScienceEnv")
    explore = ExploreAgent("ExploreAgent")
    pcs = PCSAgent("PCSAgent", limiar_estabilidade=0.75)
    model = ModelAgent("ModelAgent")

    # Conecta os agentes BDI ao ambiente de dados
    admin.connect_to([explore, pcs, model], env)

    # Dispara o ciclo de raciocínio de todos os agentes
    admin.start_system()

    if not model.modelagem_concluida:
        raise SystemExit("Pipeline encerrado antes da modelagem. Consulte a auditoria PCS.")
    print("\n[VDS-BDI]: Execução do pipeline multiagente concluída.")


if __name__ == "__main__":
    main()
