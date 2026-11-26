"""Ponto de entrada principal para execução do sistema deliberativo VDS-BDI em MASPY."""
import argparse
from pathlib import Path

from maspy import Admin
from environment.ds_environment import BANK_CHURN, DataScienceEnv
from agents.explore_agent import ExploreAgent
from agents.pcs_agent import PCSAgent
from agents.model_agent import ModelAgent
from tools.avaliacao_externa import avaliar_bank_churn


def main(
    *, simular_falhas: bool = True, limiar_estabilidade: float = 0.75,
    dataset: str = BANK_CHURN, alvo: str | None = None,
    tarefa: str = "classificacao", ignorar: list[str] | None = None,
    tamanho_amostra: int = 1000, separador: str = ",",
    avaliacao_externa: bool = False, rotulos_externos: Path | None = None,
    saida_externa: Path | None = None,
):
    if not 0 <= limiar_estabilidade <= 1:
        raise SystemExit("O limiar de estabilidade deve estar entre 0 e 1.")
    if alvo is None:
        if Path(dataset).resolve() != Path(BANK_CHURN).resolve():
            raise SystemExit("Informe a coluna alvo com --alvo ao usar outro dataset.")
        alvo = "Exited"

    if avaliacao_externa and (
        Path(dataset).resolve() != Path(BANK_CHURN).resolve()
        or alvo != "Exited" or tarefa != "classificacao"
    ):
        raise SystemExit("A avaliação externa está disponível para Bank Churn com alvo Exited em classificação.")
    if not avaliacao_externa and (rotulos_externos or saida_externa):
        raise SystemExit("Use --avaliacao-externa para configurar rótulos ou saída externa.")

    env = DataScienceEnv(
        "DataScienceEnv", simular_falhas=simular_falhas,
        alvo=alvo, tarefa=tarefa, ignorar=ignorar,
    )
    try:
        env.carregar_dataset("main", path=dataset, sample_size=tamanho_amostra, separador=separador)
    except (OSError, ValueError) as erro:
        raise SystemExit(f"Não foi possível carregar o dataset: {erro}") from erro

    print("=" * 75)
    print(" SISTEMA MULTIAGENTE DELIBERATIVO VDS-BDI (Predictability, Computability, Stability)")
    print(f" MASPY | Dataset: {dataset} | Alvo: {alvo} | Tarefa: {tarefa}")
    print("=" * 75)

    admin = Admin()
    # Desativa rastreamento verboso de baixo nível para manter logs limpos e legíveis
    admin.console_settings(
        show_exec=False,
        show_cycle=False,
        show_prct=False,
        show_slct=False,
    )

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
    if avaliacao_externa:
        treino = Path(dataset)
        try:
            model.certificado_final["avaliacao_externa"] = avaliar_bank_churn(
                treino, env.preprocessador, model.certificado_final,
                rotulos_externos or treino.with_name("test_eval.csv"),
                saida_externa or Path("resultados/avaliacao-externa"),
                len(env.raw_data), len(env.raw_test),
            )
        except (ValueError, OSError) as erro:
            raise SystemExit(f"Falha na avaliação externa: {erro}") from erro
    print("\n[VDS-BDI]: Execução do pipeline multiagente concluída.")
    return model.certificado_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=BANK_CHURN, help="Caminho do CSV com dados rotulados.")
    parser.add_argument("--alvo", help="Coluna a prever. Obrigatória para CSVs personalizados.")
    parser.add_argument("--tarefa", choices=("classificacao", "regressao"), default="classificacao")
    parser.add_argument("--ignorar", action="append", help="Coluna de entrada a excluir. Pode ser repetido.")
    parser.add_argument("--tamanho-amostra", type=int, default=1000, help="Limite de linhas. Use 0 para ler todas.")
    parser.add_argument("--separador", default=",", help="Separador de campos do CSV.")
    parser.add_argument("--sem-simulacao", action="store_false", dest="simular_falhas", help="Desativa a simulação de falhas de Bank Churn.")
    parser.add_argument("--limiar-estabilidade", type=float, default=0.75, help="Limiar da auditoria de limpeza (0 a 1).")
    parser.add_argument("--avaliacao-externa", action="store_true", help="Avalia o modelo escolhido no teste externo de Bank Churn.")
    parser.add_argument("--rotulos-externos", type=Path, help="CSV com id e Exited para pontuação. Padrão: test_eval.csv.")
    parser.add_argument("--saida-externa", type=Path, help="Diretório para os resultados de cada execução.")
    main(**vars(parser.parse_args()))
