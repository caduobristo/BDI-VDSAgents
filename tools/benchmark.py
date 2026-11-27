"""Executa o fluxo completo até obter cinco saídas válidas por base."""

import argparse
import csv
import json
from math import isfinite
from pathlib import Path
from statistics import mean, pstdev
import subprocess
import sys
from tempfile import mkdtemp
from time import perf_counter


RAIZ = Path(__file__).resolve().parents[1]
META_VALIDAS = 5
BASES = {
    "bank_churn": (),
    "adult": ("--alvo", "income"),
    "ames_houses": ("--alvo", "SalePrice", "--tarefa", "regressao", "--id-coluna", "Order"),
    "In-Vehicle_Coupon_Recommendation": ("--alvo", "Y"),
    "obesity_risks": ("--alvo", "NObeyesdad", "--alvo-externo", "True"),
    "online_shopping": ("--alvo", "Revenue", "--id-coluna", "ID"),
    "parkinsons": ("--alvo", "motor_UPDRS", "--tarefa", "regressao", "--ignorar", "total_UPDRS",
                   "--ignorar", "subject#"),
    "Seoul Bike Sharing Demand": ("--alvo", "Rented Bike Count", "--tarefa", "regressao", "--id-coluna", "ID"),
    "titanic": ("--alvo", "Survived", "--id-coluna", "PassengerId"),
}


def resumir_base(tentativas):
    nps = [tentativa["nps"] for tentativa in tentativas if tentativa["nps"] is not None]
    validas = len(nps)
    vs = validas / len(tentativas)
    completa = validas == META_VALIDAS
    anps = mean(nps) if completa else None
    return {
        "completa": completa, "tentativas": len(tentativas), "validas": validas,
        "falhas": len(tentativas) - validas, "vs": vs, "anps": anps,
        "desvio_padrao_nps": pstdev(nps) if completa else None,
        "cs": (vs + anps) / 2 if completa else None, "execucoes": tentativas,
    }


def validar_saida(arquivo):
    relatorio = json.loads(arquivo.read_text(encoding="utf-8"))
    nps = (relatorio.get("metricas_externas") or {}).get("nps")
    if type(nps) not in (int, float) or not isfinite(nps):
        raise ValueError("O relatório externo não contém um NPS finito.")

    with arquivo.with_name("predicoes.csv").open(encoding="utf-8", newline="") as entrada:
        predicoes = csv.DictReader(entrada)
        chave = relatorio["chave"]
        if not {chave, "predicao"}.issubset(predicoes.fieldnames or ()):
            raise ValueError("O CSV de predições não contém a chave e a predição.")
        ids = set()
        for linha in predicoes:
            identificador = linha[chave]
            if not identificador or identificador in ids or not linha["predicao"]:
                raise ValueError("O CSV contém identificadores ou predições vazias, ou identificadores repetidos.")
            ids.add(identificador)
    if not ids or len(ids) != relatorio["linhas_teste_externo"]:
        raise ValueError("O número de predições difere do total de linhas do teste externo.")
    return nps


def executar_tentativa(base, parametros, pasta, timeout):
    pasta.mkdir(parents=True)
    comando = [sys.executable, "-X", "utf8", "-u", "main.py"]
    if parametros:
        comando.extend(("--dataset", f"data/{base}/train.csv", *parametros))
    comando.extend(("--avaliacao-externa", "--saida-externa", str(pasta)))
    log = pasta / "terminal.log"
    inicio = perf_counter()
    codigo, motivo = None, None
    with log.open("wb") as saida:
        try:
            processo = subprocess.run(comando, cwd=RAIZ, stdout=saida, stderr=subprocess.STDOUT, timeout=timeout)
            codigo = processo.returncode
        except subprocess.TimeoutExpired:
            motivo = f"Execução excedeu o limite de {timeout:g} s."
        except OSError as erro:
            motivo = f"Falha ao iniciar o fluxo: {erro}"

    arquivos = list(pasta.glob("*/relatorio.json"))
    arquivo = arquivos[0] if len(arquivos) == 1 else None
    nps = None
    if codigo == 0:
        try:
            if arquivo is None:
                raise ValueError("A execução precisa gerar um único relatório externo.")
            nps = validar_saida(arquivo)
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as erro:
            motivo = f"Saída inválida: {erro}"
    elif motivo is None:
        linhas = log.read_text(encoding="utf-8", errors="replace").splitlines()
        motivo = next((linha for linha in reversed(linhas) if linha.strip()), f"Código de saída {codigo}.")
    return {
        "codigo_saida": codigo, "nps": nps, "motivo_falha": motivo,
        "tempo_segundos": round(perf_counter() - inicio, 3),
        "comando": comando, "log": str(log), "relatorio": str(arquivo) if arquivo else None,
    }


def executar_benchmark(destino, max_tentativas, timeout):
    destino.mkdir(parents=True, exist_ok=True)
    pasta = Path(mkdtemp(prefix="benchmark-9-", dir=destino.resolve()))
    arquivo = pasta / "resumo.json"
    resumo = {
        "meta_validas_por_base": META_VALIDAS, "max_tentativas_por_base": max_tentativas,
        "timeout_segundos": timeout, "bases": {},
    }
    print(f"Relatório do benchmark: {arquivo}", flush=True)
    for base, parametros in BASES.items():
        tentativas = []
        for numero in range(1, max_tentativas + 1):
            tentativa = executar_tentativa(base, parametros, pasta / base / f"{numero:02d}", timeout)
            tentativas.append(tentativa)
            resultado = resumir_base(tentativas)
            resumo["bases"][base] = resultado
            temporario = arquivo.with_suffix(".tmp")
            temporario.write_text(json.dumps(resumo, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                  encoding="utf-8")
            temporario.replace(arquivo)
            nps = tentativa["nps"]
            detalhe = f"NPS={nps:.4f}" if nps is not None else tentativa["motivo_falha"]
            print(f"{base} | tentativa {numero}/{max_tentativas} | válidas {resultado['validas']}/{META_VALIDAS}"
                  f" | {detalhe}", flush=True)
            if resultado["completa"]:
                break
        if resultado["completa"]:
            print(f"{base} | VS={resultado['vs']:.4f} | ANPS={resultado['anps']:.4f}"
                  f" ± {resultado['desvio_padrao_nps']:.4f} | CS={resultado['cs']:.4f}", flush=True)
        else:
            print(f"{base} | incompleta | VS observado={resultado['vs']:.4f}", flush=True)
    return arquivo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tentativas", type=int, default=8, help="Limite por base. Padrão: 8.")
    parser.add_argument("--timeout", type=float, default=180, help="Tempo limite por execução, em segundos. Padrão: 180.")
    parser.add_argument("--saida", type=Path, default=RAIZ / "resultados/avaliacao-externa",
                        help="Diretório onde será criada a pasta do benchmark.")
    args = parser.parse_args()
    if args.max_tentativas < META_VALIDAS:
        parser.error(f"--max-tentativas deve ser pelo menos {META_VALIDAS}.")
    if not isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout deve ser positivo e finito.")
    executar_benchmark(args.saida, args.max_tentativas, args.timeout)
