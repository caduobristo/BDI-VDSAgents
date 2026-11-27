# BDI-VDSAgents

O projeto usa o MASPY para coordenar três agentes em tarefas supervisionadas com dados tabulares.
Na raiz do clone, com Python 3.12 no Windows, execute:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### Como usar outro dataset

Passe um CSV com cabeçalho e rótulos em `--dataset` e informe a coluna
a prever em `--alvo`. A tarefa padrão é classificação. Para prever
valores numéricos contínuos, passe `--tarefa regressao`.

```powershell
python main.py --dataset data/online_shopping/train.csv --alvo Revenue
python main.py --dataset data/obesity_risks/train.csv --alvo NObeyesdad
python main.py --dataset data/ames_houses/train.csv --alvo SalePrice --tarefa regressao --ignorar Order
```

- `--ignorar` exclui uma coluna da modelagem e pode ser repetido. Sem essa
  opção, o ambiente ignora `id` (sem distinguir maiúsculas de minúsculas)
  ou os identificadores configurados no exemplo de Bank Churn.
- A seleção interna usa todo o CSV de treino por padrão. Para uma rodada
  rápida, passe `--tamanho-amostra 1000`. O valor `0` também usa todas as
  linhas. `--separador` aceita outro delimitador, como `";"`.
- `--metrica` define a métrica de seleção antes do treino. Use `roc_auc` para
  classificação binária, `accuracy` para classificação ou `r2` para regressão.
  O exemplo executado sem `--dataset` usa `roc_auc`. Para um CSV informado,
  os padrões são `accuracy` em classificação e `r2` em regressão.
- O ambiente remove apenas linhas de origem
  inteiramente iguais antes da divisão. Identificadores distintos preservam
  observações que aparentam ser iguais depois do pré-processamento.

Para classificação, o relatório traz acurácia e F1. Com rótulos 0 e 1,
o F1 é da classe 1. Nos demais casos, é a média simples do F1 entre classes.
Para regressão, o ambiente compara Random Forest e Ridge por R², informa MAE
e mede estabilidade pela diferença média entre predições de pares de modelos.
Essa diferença é normalizada pelo desvio absoluto médio do alvo no treino
e convertida para a escala de 0 a 1.

A divisão é aleatória. Para séries temporais ou medições repetidas do
mesmo indivíduo, use um protocolo de validação adequado ao problema
antes de interpretar os resultados como desempenho em dados novos.

### Contrato entre os agentes

O ambiente separa os dados em 75% para treino e 25% para validação interna,
com semente 42 e estratificação pelo alvo em classificação. Ambos os planos de
limpeza usam o mesmo treino. Os dados simulados de contingência, o ajuste
da imputação e da codificação, a auditoria da limpeza e as três amostras
bootstrap usam apenas essa partição. Na modelagem, o ambiente aplica o
pré-processamento aprovado à validação e avalia todos os modelos nas mesmas linhas.

Os dados tabulares ficam no ambiente compartilhado. Os agentes trocam metas,
decisões e metadados pelas mensagens abaixo.

| Origem → destino | Mensagem | Conteúdo e efeito |
| --- | --- | --- |
| Explore → PCS | `auditar_dados_pre_processados` | A tentativa, a estratégia, as dimensões e o resultado das validações (`attempt`, `strategy`, `shape`, `tests_passed`, `report`) seguem para auditoria. |
| PCS → Explore | `adotar_plano_alternativo` | O motivo da primeira reprovação aciona o Plano B. |
| PCS → Explore e Model | `pre_processamento_reprovado` | Uma segunda reprovação encerra a execução antes do treino. |
| PCS → Explore / PCS → Model | `estabilidade_pcs_aprovada` / `autorizacao_para_modelagem` | O Explore pode liberar as features e o Model recebe autorização para treinar. |
| Explore → Model | `matriz_features_aprovadas` | Identifica as features, o alvo, a tarefa e as dimensões. O Model aguarda essa mensagem e a autorização do PCS. |
| Model → PCS / PCS → Model | `auditar_predicoes_modelos` / `certificacao_pcs_emitida` | O Model envia as métricas dos candidatos, o tempo e a concordância. O PCS devolve o certificado usado no relatório. |
| PCS → Model | `replanejar_modelagem` | A reprovação de todos os candidatos pode acionar uma segunda tentativa com menos variáveis. A mensagem inclui os motivos por candidato. |

O PCS verifica o relatório de limpeza antes de avaliar as três amostras
bootstrap. A estabilidade é 1 menos a metade da diferença absoluta média entre
as matrizes de correlação. O limiar padrão é 0,75.

### Onde alterar cada agente

- **ExploreAgent** em [agents/explore_agent.py](agents/explore_agent.py).
  Defina a estratégia em `PREPARO_PADRAO` ou `PREPARO_CONTINGENCIA`, ou
  escolha a estratégia no plano e passe-a a `aplicar_preparo`. A função de
  construção recebe as variáveis do treino e retorna um transformador clonável
  pelo scikit-learn. O transformador recebe `X` e `y` no ajuste, funciona
  com `X` na transformação e retorna um DataFrame com índice preservado.
- **ModelAgent** em [agents/model_agent.py](agents/model_agent.py).
  `criar_candidatos(tarefa)` retorna um dicionário de nomes distintos para
  estimadores clonáveis. Para seleção por ROC AUC, os estimadores precisam
  oferecer `predict_proba`. O seletor recebe tarefa, variáveis e alvo do
  treino e retorna nomes de colunas. O ambiente guarda a configuração da
  tentativa e o avaliador reajusta o vencedor com o preparo e a seleção
  correspondentes, inclusive para candidatos com nomes ou parâmetros novos.
- **PCSAgent** em [agents/pcs_agent.py](agents/pcs_agent.py).
  Defina os cortes e a política em `auditar_predicoes_pcs` e a seleção final
  em `escolher_modelo`, que recebe os candidatos aprovados. A decisão sobre a
  limpeza fica em `plano_auditar_pre_processamento`. O ambiente fornece
  medidas, e o agente decide sobre aprovação e replanejamento.

Os testes em `tests/test_contrato_agentes.py` executam o programa em cópias
isoladas e alteram um único arquivo de agente por cenário.

### Auditoria por modelo e replanejamento

O PCS verifica a previsibilidade e a estabilidade de cada candidato e escolhe
o de maior pontuação entre os aprovados. O tempo de modelagem da tentativa
deve ficar abaixo de 30 segundos. Para classificação, a estabilidade individual
é a fração média de classes preditas iguais entre pares de ajustes bootstrap
nas mesmas linhas de validação. O corte padrão é 0,80.

Quando todos falham e há pelo menos duas variáveis, o ModelAgent
tenta mais um plano. Ele mantém a metade das variáveis pré-processadas, arredondada
para baixo, com maior estatística F no treino interno. O tipo de tarefa
determina se a estatística é calculada para classificação ou regressão.
Essa lista fica fixa nos bootstraps, na validação e no reajuste externo. Os dois
planos usam os mesmos clientes, amostras bootstrap, algoritmos e limiares.

Para observar o replanejamento com dados originais e o corte experimental de 0,95, execute

```powershell
py -3.12 -X utf8 main.py --tamanho-amostra 1000 --metrica roc_auc --limiar-estabilidade-modelo 0.95 --avaliacao-externa
```

Com avaliação externa habilitada, o JSON guarda cada tentativa de modelagem, as variáveis,
as métricas, os motivos de reprovação e a mensagem do PCS. Se os dois planos
falharem, o programa grava o histórico, encerra com código 1 e preserva o teste
externo sem leitura ou geração de predições.

### Verificação da integração

O padrão usa os dados originais.
Use `python main.py --tamanho-amostra 1000 --simular-falhas` para demonstrar a contingência de
limpeza. O ambiente insere, em memória, valores ausentes e um valor extremo
em uma variável numérica do treino interno. Os CSVs e as partições de
validação e teste são preservados. Os bootstraps do PCS fazem parte da
auditoria em ambos os modos.
O parâmetro `--limiar-estabilidade` controla a auditoria de limpeza. Com o
valor 1 e `--tamanho-amostra 1000` no exemplo de Bank Churn, as duas tentativas são reprovadas e o programa
encerra com código 1 e o motivo informado pelo PCS.

Execute `python -m unittest discover -s tests -v` para verificar as partições,
a aprovação direta, a recuperação pelo Plano B e o encerramento por reprovação.

### Avaliação externa nas três bases

```powershell
py -3.12 -X utf8 main.py --avaliacao-externa
py -3.12 -X utf8 main.py --dataset data/obesity_risks/train.csv --alvo NObeyesdad --alvo-externo True --avaliacao-externa
py -3.12 -X utf8 main.py --dataset data/ames_houses/train.csv --alvo SalePrice --tarefa regressao --id-coluna Order --avaliacao-externa
```

Os agentes dividem `train.csv` entre treino interno e validação para escolher
o modelo. Após a certificação, o programa reajusta o preparo e o vencedor
com o mesmo `train.csv` completo, inclusive quando a seleção usou
`--tamanho-amostra`. Ele mantém as colunas escolhidas quando o plano reduzido
vence. Em seguida, gera as predições a partir de `test.csv`, grava
`predicoes.csv` e só então lê `test_eval.csv` para medir o desempenho.

Por padrão, o programa busca `test.csv` e `test_eval.csv` ao lado do treino.
Use `--teste-externo` e `--rotulos-externos` para outros caminhos e
`--alvo-externo` quando a coluna do gabarito tiver outro nome. Os três CSVs
devem trazer a chave configurada em `--id-coluna` (padrão `id`), que é
excluída das variáveis do modelo. O teste fornece as variáveis sem o alvo.
As chaves do teste devem coincidir com as dos rótulos e ser distintas das do treino.

O relatório usa precisão, recall e F1 binários para duas classes, com a
segunda classe na ordenação do modelo como positiva. Nesse caso, o CSV de
predições inclui `probabilidade_classe_positiva` e o avaliador calcula AUC
quando as duas classes aparecem no gabarito. Em multiclasse, usa média macro.
Em regressão, calcula RMSE, MAE e R². O NPS individual é gravado no
`relatorio.json`. As saídas ficam em `resultados/avaliacao-externa/`, com opção
de alterar o destino via `--saida-externa`.

### Benchmark dos nove datasets

```powershell
py -3.12 -X utf8 -m tools.benchmark --max-tentativas 8
```

O executor chama `main.py --avaliacao-externa` para cada tentativa,
com dados de treino completos, sem simulação e com as configurações padrão
de seleção e auditoria. Cada base para ao obter cinco saídas válidas ou
atingir o limite de tentativas. O timeout padrão é de 180 segundos por
execução e pode ser ajustado com `--timeout`.

Uma saída válida precisa de código de saída zero, NPS externo finito e um
CSV com uma predição por registro do teste. Uma troca de plano conta dentro
da mesma tentativa. Falhas entram no denominador de VS. ANPS é a média
dos cinco NPS válidos, acompanhada pelo desvio-padrão populacional. CS
é a média de VS e ANPS. Para uma base incompleta, o resumo mostra o VS
observado e deixa ANPS, desvio-padrão e CS sem valor.

A pasta `resultados/avaliacao-externa/benchmark-9-*` contém o log e os
artefatos de cada tentativa. O arquivo `resumo.json` é atualizado ao fim
de cada execução. As rodadas reutilizam as sementes atuais do pipeline.
O desvio-padrão descreve a variação observada nessas repetições.

[Protocolo experimental do VDSAgents](https://arxiv.org/html/2510.24339v2#S4)
