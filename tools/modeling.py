"""Compara modelos de classificação ou regressão no mesmo conjunto de teste."""

from itertools import combinations
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


NOMES_METRICAS = {"accuracy": "Acurácia", "roc_auc": "ROC AUC", "f1_score": "F1", "mae": "MAE", "r2": "R²"}
METRICAS_SELECAO = {"classificacao": ("accuracy", "roc_auc"), "regressao": ("r2",)}


def formatar_metricas(metricas):
    return ", ".join(f"{NOMES_METRICAS[chave]}={valor:.4f}" for chave, valor in metricas.items())


def criar_hipoteses(tarefa):
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


def avaliar_modelos(tarefa, X_treino, y_treino, X_teste, y_teste, perturbacoes, alvo, metrica_principal=None):
    regressao = tarefa == "regressao"
    if metrica_principal is None:
        metrica_principal = METRICAS_SELECAO[tarefa][0]
    if regressao:
        escala = float(np.mean(np.abs(y_treino - y_treino.mean()))) or 1.0
    else:
        media_f1 = "binary" if set(y_treino.unique()) == {0, 1} else "macro"
        rotulos = LabelEncoder().fit(y_treino)
        y_treino = rotulos.transform(y_treino)
        y_teste = rotulos.transform(y_teste)

    def concordancia(a, b):
        if regressao:
            return float(1 / (1 + np.mean(np.abs(a - b)) / escala))
        return float(np.mean(a == b))

    resultados = {}
    predicoes = []
    inicio = perf_counter()
    for nome, modelo in criar_hipoteses(tarefa).items():
        pred = modelo.fit(X_treino, y_treino).predict(X_teste)
        predicoes.append(pred)
        if regressao:
            metricas = {"mae": mean_absolute_error(y_teste, pred), "r2": r2_score(y_teste, pred)}
        else:
            metricas = {
                "accuracy": accuracy_score(y_teste, pred),
                "f1_score": f1_score(y_teste, pred, average=media_f1, zero_division=0),
            }
            if metrica_principal == "roc_auc":
                prob_positiva = modelo.predict_proba(X_teste)[:, 1]
                metricas["roc_auc"] = roc_auc_score(y_teste, prob_positiva)

        predicoes_bootstrap = [
            modelo.fit(
                amostra.drop(columns=alvo),
                amostra[alvo] if regressao else rotulos.transform(amostra[alvo]),
            ).predict(X_teste)
            for amostra in perturbacoes
        ]
        pares = [concordancia(a, b) for a, b in combinations(predicoes_bootstrap, 2)]
        resultados[nome] = {
            "metricas": metricas,
            "perturbation_stability": float(np.mean(pares)) if pares else 0.0,
        }

    return {
        "models": resultados,
        "primary_metric": metrica_principal,
        "elapsed_time_seconds": perf_counter() - inicio,
        "inter_model_agreement": concordancia(*predicoes),
        "test_sample_size": len(y_teste),
    }
