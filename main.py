"""Ponto de entrada principal para execução do sistema deliberativo VDS-BDI em MASPY."""
import argparse

from maspy import Admin
from environment.ds_environment import DataScienceEnv
from agents.explore_agent import ExploreAgent
from agents.pcs_agent import PCSAgent
from agents.model_agent import ModelAgent


def main(*, simular_falhas: bool = True, limiar_estabilidade: float = 0.75):
    if not 0 <= limiar_estabilidade <= 1:
        raise SystemExit("O limiar de estabilidade deve estar entre 0 e 1.")
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

    env = DataScienceEnv("DataScienceEnv", simular_falhas=simular_falhas)
    explore = ExploreAgent("ExploreAgent")
    pcs = PCSAgent("PCSAgent", limiar_estabilidade=limiar_estabilidade)
    model = ModelAgent("ModelAgent")

    # Conecta os agentes BDI ao ambiente de dados
    admin.connect_to([explore, pcs, model], env)

    # Dispara o ciclo de raciocínio de todos os agentes
    admin.start_system()

    if model.motivo_reprovacao is not None:
        raise SystemExit(f"Pré-processamento reprovado: {model.motivo_reprovacao}")
    if model.certificado_final is None:
        raise SystemExit("Pipeline encerrado sem entrega do relatório PCS.")
    print("\n[VDS-BDI]: Execução do pipeline multiagente concluída.")
    return model.certificado_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sem-simulacao", action="store_true", help="Usa os dados sem injetar nulos e outliers.")
    parser.add_argument("--limiar-estabilidade", type=float, default=0.75, help="Limiar da auditoria de limpeza (0 a 1).")
    args = parser.parse_args()
    main(simular_falhas=not args.sem_simulacao, limiar_estabilidade=args.limiar_estabilidade)
