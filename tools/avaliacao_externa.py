"""Avaliação externa de uma hipótese já selecionada para Bank Churn."""

import json
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from tools.modeling import criar_hipoteses


def validar_ids(dados, origem):
    if "id" not in dados or dados["id"].isna().any() or dados["id"].duplicated().any():
        raise ValueError(f"{origem} precisa de um id único e preenchido por linha.")


def prever_bank_churn(treino, teste, preprocessador, nome_modelo):
    validar_ids(treino, "train.csv")
    validar_ids(teste, "test.csv")
    if "Exited" in teste:
        raise ValueError("test.csv deve conter somente identificadores e features.")
    if teste["id"].isin(treino["id"]).any():
        raise ValueError("Há ids compartilhados entre treino e teste externo.")

    colunas = list(preprocessador.feature_names_in_)
    faltantes = set(colunas) - set(teste.columns)
    if faltantes:
        raise ValueError(f"Features ausentes em test.csv: {', '.join(sorted(faltantes))}.")
    if set(treino["Exited"].unique()) != {0, 1}:
        raise ValueError("O treino de Bank Churn precisa das classes 0 e 1 em Exited.")

    preparo_final = clone(preprocessador)
    X_treino = preparo_final.fit_transform(treino[colunas])
    modelo = criar_hipoteses("classificacao")[nome_modelo]
    modelo.fit(X_treino, treino["Exited"])
    X_teste = preparo_final.transform(teste[colunas])
    classe_positiva = list(modelo.classes_).index(1)
    return pd.DataFrame({
        "id": teste["id"],
        "predicao": modelo.predict(X_teste),
        "probabilidade_churn": modelo.predict_proba(X_teste)[:, classe_positiva],
    })


def pontuar_bank_churn(predicoes, arquivo_rotulos):
    rotulos = pd.read_csv(arquivo_rotulos, usecols=["id", "Exited"])
    validar_ids(rotulos, "Arquivo de rótulos")
    validar_ids(predicoes, "Predições")
    if set(rotulos["id"]) != set(predicoes["id"]):
        raise ValueError("Os ids dos rótulos externos e das predições devem coincidir.")
    y = rotulos.set_index("id").loc[predicoes["id"], "Exited"]
    if set(y.unique()) != {0, 1}:
        raise ValueError("A avaliação com AUC exige rótulos externos 0 e 1.")
    previsto = predicoes["predicao"]
    return {
        "acuracia": accuracy_score(y, previsto),
        "precisao": precision_score(y, previsto, zero_division=0),
        "recall": recall_score(y, previsto, zero_division=0),
        "f1": f1_score(y, previsto, zero_division=0),
        "auc": roc_auc_score(y, predicoes["probabilidade_churn"]),
    }


def avaliar_bank_churn(arquivo_treino, preprocessador, certificado, arquivo_rotulos, destino,
                      n_treino_interno, n_validacao):
    nome = certificado["best_hypothesis"]
    criterio = certificado["primary_metric"]
    print(f"[Avaliação externa] Hipótese fixada pela validação interna: {nome}.")
    treino = pd.read_csv(arquivo_treino)
    teste = pd.read_csv(arquivo_treino.with_name("test.csv"))
    predicoes = prever_bank_churn(treino, teste, preprocessador, nome)
    destino.mkdir(parents=True, exist_ok=True)
    pasta = Path(mkdtemp(prefix="bank_churn-", dir=destino))
    predicoes.to_csv(pasta / "predicoes.csv", index=False)
    print(f"[Avaliação externa] {len(predicoes)} predições gravadas em {pasta / 'predicoes.csv'}.")
    print(f"[Avaliação externa] Lendo rótulos de {arquivo_rotulos} para pontuação.")
    metricas = pontuar_bank_churn(predicoes, arquivo_rotulos)
    relatorio = {
        "modelo": nome,
        "selecao_interna": {
            "criterio": "acuracia" if criterio == "accuracy" else criterio,
            "valor": certificado["predictability"][criterio],
            "pontuacoes_por_modelo": certificado["validation_scores"],
            "acuracia": certificado["predictability"]["accuracy"],
            "status_pcs": certificado["status"],
            "linhas_treino": n_treino_interno,
            "linhas_validacao": n_validacao,
        },
        "linhas_reajuste": len(treino),
        "linhas_teste_externo": len(teste),
        "metricas_externas": metricas,
    }
    (pasta / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for chave, valor in metricas.items():
        print(f"  {chave}: {valor:.4f}")
    return relatorio
