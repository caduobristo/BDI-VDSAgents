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
- `--tamanho-amostra` limita a entrada a 1.000 linhas por padrão. Use `0`
  para processar todo o CSV. `--separador` aceita outro delimitador, como `";"`.
- Em CSVs personalizados, o ambiente remove apenas linhas de origem
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

O ambiente separa a amostra em 75% para treino e 25% para
teste, com semente 42 e estratificação pelo alvo em classificação. Ambos os planos de
limpeza usam o mesmo treino. Os dados simulados de contingência, o ajuste
da imputação e da codificação, a auditoria da limpeza e as três amostras
bootstrap usam apenas essa partição. Na modelagem, o ambiente aplica ao
teste o pré-processamento aprovado e avalia todos os modelos nas mesmas linhas.

Os dados tabulares ficam no ambiente compartilhado. Os agentes trocam metas,
decisões e metadados pelas mensagens abaixo.

| Origem → destino | Mensagem | Conteúdo e efeito |
| --- | --- | --- |
| Explore → PCS | `auditar_dados_pre_processados` | A tentativa, a estratégia, as dimensões e o resultado das validações (`attempt`, `strategy`, `shape`, `tests_passed`, `report`) seguem para auditoria. |
| PCS → Explore | `adotar_plano_alternativo` | O motivo da primeira reprovação aciona o Plano B. |
| PCS → Explore e Model | `pre_processamento_reprovado` | Uma segunda reprovação encerra a execução antes do treino. |
| PCS → Explore / PCS → Model | `estabilidade_pcs_aprovada` / `autorizacao_para_modelagem` | O Explore pode liberar as features e o Model recebe autorização para treinar. |
| Explore → Model | `matriz_features_aprovadas` | Identifica as features, o alvo e as dimensões. O Model aguarda tanto essa mensagem quanto a autorização do PCS. |
| Model → PCS / PCS → Model | `auditar_predicoes_modelos` / `certificacao_pcs_emitida` | O Model envia as métricas das duas hipóteses, o tempo e a concordância. O PCS devolve o certificado usado no relatório. |

O PCS verifica o relatório de limpeza antes de avaliar as três amostras
bootstrap. A estabilidade é 1 menos a metade da diferença absoluta média entre
as matrizes de correlação. O limiar padrão é 0,75. Para Bank Churn, um máximo
superior ao dobro do percentil 99 em `Balance` ou `EstimatedSalary` também
reprova o pré-processamento.

### Verificação da integração

A execução padrão de Bank Churn usa falhas simuladas no treino para exercitar a contingência.
Use `python main.py --sem-simulacao` para executar com a amostra original.
O parâmetro `--limiar-estabilidade` controla a auditoria de limpeza. Com o
valor 1 e a amostra padrão, as duas tentativas são reprovadas e o programa
encerra com código 1 e o motivo informado pelo PCS.

Execute `python -m unittest discover -s tests -v` para verificar as partições,
a aprovação direta, a recuperação pelo Plano B e o encerramento por reprovação.

### Avaliação externa de Bank Churn

```powershell
py -3.12 -X utf8 main.py --sem-simulacao --avaliacao-externa
```

Os agentes usam somente `train.csv` para aprovar o plano de limpeza e
selecionar o modelo por acurácia na validação interna. A amostra padrão
contém 750 linhas para treino e 250 para validação. A opção
`--tamanho-amostra 0` usa o arquivo inteiro nessa seleção.

Após a entrega do certificado PCS, o programa reajusta o pré-processamento
aprovado e a hipótese vencedora com **todo** o `train.csv`. Ele prevê a
classe e a probabilidade de `Exited = 1` para cada `id` de `test.csv` e
grava `predicoes.csv`. Só então lê `id` e `Exited` de `test_eval.csv`.
O avaliador alinha os rótulos por `id` e exige uma correspondência de um
para um para calcular acurácia, precisão, recall, F1 e ROC AUC.

Cada execução cria uma pasta em `resultados/avaliacao-externa/` com as
predições e o `relatorio.json`. O relatório identifica o modelo, o critério
de seleção, a acurácia de validação, o status do PCS, as quantidades de
linhas e as métricas externas. Use `--saida-externa` para escolher o diretório
e `--rotulos-externos` para indicar outro arquivo de rótulos por id.
