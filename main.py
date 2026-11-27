"""Ponto de entrada principal para execução do sistema deliberativo VDS-BDI em MASPY."""
import argparse
from pathlib import Path

from maspy import Admin
from environment.ds_environment import DataScienceEnv
from agents.explore_agent import ExploreAgent
from agents.pcs_agent import PCSAgent
from agents.model_agent import ModelAgent
from tools.avaliacao_externa import avaliar_externamente
from tools.modeling import METRICAS_SELECAO, NOMES_METRICAS


def main(
    *, simular_falhas: bool = False, limiar_estabilidade: float = 0.75,
    dataset: str | None = None, alvo: str | None = None,
    tarefa: str = "classificacao", ignorar: list[str] | None = None,
    tamanho_amostra: int = 1000, separador: str = ",",
    avaliacao_externa: bool = False, rotulos_externos: Path | None = None,
    saida_externa: Path | None = None, metrica: str | None = None,
    limiar_estabilidade_modelo: float = 0.80,
    teste_externo: Path | None = None, id_coluna: str | None = None, alvo_externo: str | None = None,
):
    if not 0 <= limiar_estabilidade <= 1:
        raise SystemExit("O limiar de estabilidade deve estar entre 0 e 1.")
    if not 0 <= limiar_estabilidade_modelo <= 1:
        raise SystemExit("O limiar de estabilidade do modelo deve estar entre 0 e 1.")
    if dataset is None:
        dataset = "data/bank_churn/train.csv"
        alvo = alvo or "Exited"
        if ignorar is None:
            ignorar = ["CustomerId", "Surname", "id"]
        if tarefa == "classificacao" and alvo == "Exited":
            metrica = metrica or "roc_auc"
    elif alvo is None:
        raise SystemExit("Informe a coluna alvo com --alvo ao passar --dataset.")

    if not avaliacao_externa and (teste_externo or rotulos_externos or saida_externa or alvo_externo):
        raise SystemExit("Use --avaliacao-externa para configurar os arquivos ou o alvo externo.")
    if avaliacao_externa or id_coluna is not None:
        id_coluna = id_coluna or "id"
        ignorar = list(dict.fromkeys([*(ignorar or []), id_coluna]))

    env = DataScienceEnv(
        "DataScienceEnv", simular_falhas=simular_falhas,
        alvo=alvo, tarefa=tarefa, ignorar=ignorar,
    )
    try:
        env.carregar_dataset("main", path=dataset, sample_size=tamanho_amostra, separador=separador, metrica=metrica)
    except (OSError, ValueError) as erro:
        raise SystemExit(f"Não foi possível carregar o dataset: {erro}") from erro

    print("=" * 75)
    print(" SISTEMA MULTIAGENTE DELIBERATIVO VDS-BDI (Predictability, Computability, Stability)")
    print(f" MASPY | Dataset: {dataset} | Alvo: {alvo} | Tarefa: {tarefa}")
    print(f" Métrica de seleção na validação interna: {NOMES_METRICAS[env.metrica_principal]}")
    print(f" Estabilidade mínima por modelo: {limiar_estabilidade_modelo:g}")
    print(f" Injeção de falhas no treino: {'ativada' if simular_falhas else 'desativada'}")
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
    pcs = PCSAgent("PCSAgent", limiar_estabilidade=limiar_estabilidade,
                   limiar_estabilidade_modelo=limiar_estabilidade_modelo)
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
            model.certificado_final["avaliacao_externa"] = avaliar_externamente(
                treino, env.preprocessador, model.certificado_final,
                candidato=env.candidatos_avaliados.get(model.certificado_final["best_hypothesis"]),
                colunas_entrada=env.colunas_entrada, variaveis_selecionadas=env.variaveis_selecionadas,
                arquivo_teste=teste_externo or treino.with_name("test.csv"),
                arquivo_rotulos=rotulos_externos or treino.with_name("test_eval.csv"),
                destino=saida_externa or Path("resultados/avaliacao-externa"),
                chave=id_coluna, alvo=alvo, alvo_externo=alvo_externo or alvo,
                tarefa=tarefa, separador=separador,
                n_treino_interno=len(env.raw_data), n_validacao=len(env.raw_test),
            )
        except (ValueError, OSError) as erro:
            raise SystemExit(f"Falha na avaliação externa: {erro}") from erro
        if model.certificado_final["status"] != "CERTIFICADO":
            raise SystemExit("Modelagem reprovada. Teste externo não utilizado.")
    print("\n[VDS-BDI]: Execução do pipeline multiagente concluída.")
    return model.certificado_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Caminho do CSV com dados rotulados. Se omitido, usa o exemplo Bank Churn.")
    parser.add_argument("--alvo", help="Coluna a prever. Obrigatória para CSVs personalizados.")
    parser.add_argument("--tarefa", choices=("classificacao", "regressao"), default="classificacao")
    parser.add_argument("--metrica", choices=[m for opcoes in METRICAS_SELECAO.values() for m in opcoes],
                        help="Métrica de seleção interna. Padrões: accuracy em classificação, r2 em regressão e roc_auc no exemplo Bank Churn.")
    parser.add_argument("--ignorar", action="append", help="Coluna de entrada a excluir. Pode ser repetido.")
    parser.add_argument("--tamanho-amostra", type=int, default=1000, help="Limite de linhas. Use 0 para ler todas.")
    parser.add_argument("--separador", default=",", help="Separador de campos do CSV.")
    parser.add_argument("--simular-falhas", action="store_true",
                        help="Injeta falhas no treino para demonstrar a contingência de limpeza.")
    parser.add_argument("--limiar-estabilidade", type=float, default=0.75, help="Limiar da auditoria de limpeza (0 a 1).")
    parser.add_argument("--limiar-estabilidade-modelo", type=float, default=0.80,
                        help="Concordância bootstrap mínima para aprovar cada modelo (0 a 1).")
    parser.add_argument("--avaliacao-externa", action="store_true", help="Avalia o modelo aprovado em um teste reservado.")
    parser.add_argument("--teste-externo", type=Path, help="CSV com chaves e variáveis. Padrão: test.csv ao lado do treino.")
    parser.add_argument("--rotulos-externos", type=Path, help="CSV com chaves e rótulos. Padrão: test_eval.csv ao lado do treino.")
    parser.add_argument("--id-coluna", help="Chave usada para alinhar as predições e excluída do treino. Padrão na avaliação externa: id.")
    parser.add_argument("--alvo-externo", help="Nome do rótulo no CSV externo. Padrão: o mesmo de --alvo.")
    parser.add_argument("--saida-externa", type=Path, help="Diretório para os resultados de cada execução.")
    main(**vars(parser.parse_args()))
