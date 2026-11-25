# BDI-VDSAgents

O exemplo de Bank Churn usa o MASPY para coordenar três agentes.
Na raiz do clone, com Python 3.12 no Windows, execute:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### Contrato entre os agentes

O ambiente separa a amostra de Bank Churn em 75% para treino e 25% para
teste, com estratificação por `Exited` e semente 42. Ambos os planos de
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

A execução padrão usa falhas simuladas no treino para exercitar a contingência.
Use `python main.py --sem-simulacao` para executar com a amostra original.
O parâmetro `--limiar-estabilidade` controla a auditoria de limpeza. Com o
valor 1 e a amostra padrão, as duas tentativas são reprovadas e o programa
encerra com código 1 e o motivo informado pelo PCS.

Execute `python -m unittest discover -s tests -v` para verificar as partições,
a aprovação direta, a recuperação pelo Plano B e o encerramento por reprovação.
