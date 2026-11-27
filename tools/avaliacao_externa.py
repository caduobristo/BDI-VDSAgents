"""Avaliação externa de uma hipótese já selecionada pelo PCS."""

import json
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score, roc_auc_score,
)

def validar_ids(dados, origem, chave):
    if chave not in dados or dados[chave].isna().any() or dados[chave].duplicated().any():
        raise ValueError(f"{origem} precisa de uma chave '{chave}' única e preenchida por linha.")


def gerar_predicoes(treino, teste, preprocessador, candidato, colunas, variaveis, *, chave, alvo, tarefa):
    validar_ids(treino, "O treino", chave)
    validar_ids(teste, "O teste externo", chave)
    if chave in ("predicao", "probabilidade_classe_positiva"):
        raise ValueError(f"A chave '{chave}' conflita com uma coluna de saída.")
    if alvo in teste:
        raise ValueError("O teste externo deve conter somente identificadores e variáveis de entrada.")
    if teste[chave].isin(treino[chave]).any():
        raise ValueError("Há ids compartilhados entre treino e teste externo.")

    faltantes = set(colunas) - set(teste.columns)
    if faltantes:
        raise ValueError(f"Features ausentes em test.csv: {', '.join(sorted(faltantes))}.")
    preparo_final = clone(preprocessador)
    X_treino = preparo_final.fit_transform(treino[colunas].replace([np.inf, -np.inf], np.nan), treino[alvo])
    X_teste = preparo_final.transform(teste[colunas].replace([np.inf, -np.inf], np.nan))
    if variaveis is not None:
        if set(variaveis) - set(X_treino.columns):
            raise ValueError("O reajuste não produziu todas as variáveis certificadas.")
        X_treino = X_treino[variaveis]
        X_teste = X_teste[variaveis]
    modelo = clone(candidato).fit(X_treino, treino[alvo])
    predicoes = pd.DataFrame({
        chave: teste[chave],
        "predicao": modelo.predict(X_teste),
    })
    classes = list(modelo.classes_) if tarefa == "classificacao" else None
    if classes is not None and len(classes) == 2 and hasattr(modelo, "predict_proba"):
        predicoes["probabilidade_classe_positiva"] = modelo.predict_proba(X_teste)[:, 1]
    return predicoes, np.asarray(classes).tolist() if classes is not None else None


def pontuar_predicoes(predicoes, arquivo_rotulos, *, chave, alvo, tarefa, classes=None, separador=","):
    rotulos = pd.read_csv(arquivo_rotulos, sep=separador, usecols=[chave, alvo], dtype={chave: "string"})
    validar_ids(rotulos, "O arquivo de rótulos", chave)
    validar_ids(predicoes, "As predições", chave)
    if set(rotulos[chave]) != set(predicoes[chave]):
        raise ValueError("Os ids dos rótulos externos e das predições devem coincidir.")
    y = rotulos.set_index(chave).loc[predicoes[chave], alvo]
    previsto = predicoes["predicao"]
    if y.isna().any():
        raise ValueError("Os rótulos externos precisam estar preenchidos.")
    if tarefa == "regressao":
        rmse = float(np.sqrt(mean_squared_error(y, previsto)))
        mae = float(mean_absolute_error(y, previsto))
        r2 = float(r2_score(y, previsto)) if len(y) >= 2 else None
        return {"rmse": rmse, "mae": mae, "r2": r2,
                "nps": (1 / (1 + rmse) + 1 / (1 + mae) + r2) / 3 if r2 is not None else None}

    if not set(y).issubset(classes):
        raise ValueError("Os rótulos externos contêm classes ausentes do treino.")
    opcoes = {"average": "macro", "labels": classes, "zero_division": 0}
    if len(classes) == 2:
        opcoes = {"average": "binary", "pos_label": classes[1], "zero_division": 0}
    metricas = {
        "acuracia": accuracy_score(y, previsto),
        "precisao": precision_score(y, previsto, **opcoes),
        "recall": recall_score(y, previsto, **opcoes),
        "f1": f1_score(y, previsto, **opcoes),
    }
    metricas["nps"] = sum(metricas.values()) / 4
    if len(classes) == 2 and y.nunique() == 2 and "probabilidade_classe_positiva" in predicoes:
        metricas["auc"] = roc_auc_score(y == classes[1], predicoes["probabilidade_classe_positiva"])
    return metricas


def avaliar_externamente(arquivo_treino, preprocessador, certificado, *, candidato, colunas_entrada,
                        variaveis_selecionadas, arquivo_teste, arquivo_rotulos,
                        destino, chave, alvo, alvo_externo, tarefa, separador, n_treino_interno, n_validacao):
    nome = certificado["best_hypothesis"]
    criterio = certificado["primary_metric"]
    destino.mkdir(parents=True, exist_ok=True)
    pasta = Path(mkdtemp(prefix=f"{arquivo_treino.parent.name}-", dir=destino))
    relatorio = {
        "dataset": str(arquivo_treino),
        "tarefa": tarefa,
        "chave": chave,
        "alvo": alvo,
        "alvo_externo": alvo_externo,
        "modelo": nome,
        "historico_modelagem": certificado["historico_modelagem"],
        "replanejamentos": len(certificado["historico_modelagem"]) - 1,
        "selecao_interna": {
            "plano": certificado["auditoria"]["plano"],
            "criterio": "acuracia" if criterio == "accuracy" else criterio,
            "valor": certificado["predictability"].get(criterio),
            "pontuacoes_por_modelo": certificado["validation_scores"],
            "acuracia": certificado["predictability"].get("accuracy"),
            "status_pcs": certificado["status"],
            "linhas_treino": n_treino_interno,
            "linhas_validacao": n_validacao,
        },
        "linhas_reajuste": 0,
        "linhas_teste_externo": 0,
        "metricas_externas": None,
    }
    if nome is not None and certificado["status"] == "CERTIFICADO":
        if candidato is None:
            raise ValueError(f"A hipótese certificada '{nome}' está ausente da última tentativa.")
        print(f"[Avaliação externa] Hipótese fixada pela validação interna: {nome}.")
        treino = pd.read_csv(arquivo_treino, sep=separador, dtype={chave: "string"})
        teste = pd.read_csv(arquivo_teste, sep=separador, dtype={chave: "string"})
        predicoes, classes = gerar_predicoes(
            treino, teste, preprocessador, candidato, colunas_entrada, variaveis_selecionadas,
            chave=chave, alvo=alvo, tarefa=tarefa,
        )
        predicoes.to_csv(pasta / "predicoes.csv", index=False)
        print(f"[Avaliação externa] {len(predicoes)} predições gravadas em {pasta / 'predicoes.csv'}.")
        print(f"[Avaliação externa] Lendo rótulos de {arquivo_rotulos} para pontuação.")
        metricas = pontuar_predicoes(predicoes, arquivo_rotulos, chave=chave, alvo=alvo_externo,
                                    tarefa=tarefa, classes=classes, separador=separador)
        if classes is not None:
            relatorio["classes"] = classes
            if len(classes) == 2:
                relatorio["classe_positiva"] = classes[1]
        relatorio.update(linhas_reajuste=len(treino), linhas_teste_externo=len(teste), metricas_externas=metricas)
        for metrica, valor in metricas.items():
            print(f"  {metrica}: {valor:.4f}" if valor is not None else f"  {metrica}: indisponível")
    (pasta / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Histórico de modelagem gravado em {pasta / 'relatorio.json'}.")
    return relatorio
