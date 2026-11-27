# Classificação de Intenções Bancárias com NLP e LLMs

Como um banco que recebe milhares de mensagens de clientes por dia pode encaminhar cada uma automaticamente para a equipe certa, de maneira efetiva? E quando vale a pena pagar por uma LLM em vez de usar um modelo clássico de Machine Learning?

Neste projeto pessoal, construo um classificador de intenções em Python com Machine Learning e o comparo com classificações feitas por LLMs, usando o dataset **Banking77 em português** para responder a essas perguntas. São cerca de 13 mil mensagens de clientes em **77 categorias de atendimento**, como `chegada_do_cartão`, `alterar_pin` e `cancelar_transferência`.

> **Status:** em andamento. A parte clássica (modelagem e avaliação do melhor modelo) está concluída; a próxima etapa é a classificação com LLMs.

## Resultado até agora

O melhor modelo clássico é o **SVM (kernel RBF) sobre os embeddings do `multilingual-e5-large`**, com **0,9307 de F1-macro** e 0,9286 de acurácia no teste, **+6,53 pp** de F1-macro sobre o baseline (Regressão Logística sem ajuste sobre TF-IDF).

Os erros são poucos (7,1% das mensagens) e se concentram em pares de categorias parecidas. Com só 30 mensagens rotuladas por categoria, esse modelo já supera o melhor modelo com TF-IDF treinado com todos os dados.

## A ideia

O projeto compara duas formas de resolver o mesmo problema:

| | Machine Learning clássico | LLMs via API |
|---|---|---|
| **Como funciona** | um modelo treinado com as mensagens rotuladas | a LLM recebe a mensagem e a lista de categorias num prompt |
| **Variações testadas** | Regressão Logística, SVM, Random Forest e XGBoost, sobre TF-IDF e sobre embeddings de sentenças | prompts zero-shot (só as categorias) e few-shot (com exemplos) |
| **Precisa de dados rotulados?** | sim, para o treino | não (zero-shot) ou poucos exemplos (few-shot) |
| **Custo** | treino uma vez; prever é praticamente grátis | paga por mensagem classificada |

Primeiro, os modelos clássicos são comparados entre si, para encontrar a melhor combinação de representação de texto e modelo. Depois, o melhor deles é comparado com as LLMs, usando:

- **F1-macro** (métrica principal) e **acurácia**;
- **custo por mil mensagens**;
- **tempo de resposta**;
- **análise dos erros**, principalmente entre categorias parecidas (por exemplo, os três rótulos de taxa de câmbio);
- **curva de aprendizado**: quantas mensagens rotuladas o modelo clássico precisa para alcançar o desempenho de uma LLM.

No fim, os resultados são traduzidos para a linguagem de negócio, apontando a opção com melhor custo-benefício para cada cenário.

## Etapas

- [x] **Pré-processamento** dos textos ([`01_pre_processamento.ipynb`](notebooks/01_pre_processamento.ipynb))
- [x] **Análise exploratória** com visualização de dados e técnicas de PLN: tokenização, stopwords e TF-IDF com n-gramas ([`02_analise_exploratoria.ipynb`](notebooks/02_analise_exploratoria.ipynb))
- [x] **Modelagem com TF-IDF**: baseline, Regressão Logística, SVM, Random Forest e XGBoost ([`03_modelagem.ipynb`](notebooks/03_modelagem.ipynb))
- [x] **Modelagem com embeddings** de sentenças: Regressão Logística e SVM sobre três encoders ([`03_modelagem.ipynb`](notebooks/03_modelagem.ipynb))
- [x] **Avaliação do melhor modelo**: métricas por classe, análise de erros entre categorias parecidas, curva de aprendizado e projeção com PCA ([`04_avaliacao.ipynb`](notebooks/04_avaliacao.ipynb))
- [ ] **Classificação com LLMs via API**, com prompts zero-shot e few-shot
- [ ] **Comparação final** por F1-macro, acurácia, custo, tempo de resposta e curva de aprendizado
- [ ] **Recomendação de negócio**

## Dataset

[`c2d-usp/banking77-pt-br`](https://huggingface.co/datasets/c2d-usp/banking77-pt-br): tradução para o português do Brasil do dataset Banking77, feita com o ChatGPT (`gpt-3.5-turbo` e `gpt-4o-mini`) e depois revisada.

| | |
|---|---|
| Mensagens no dataset original | 13.085 |
| Mensagens após a limpeza | 12.530 |
| Categorias | 77 |
| Mensagens por categoria | de 71 a 225 (razão de 3,2) |
| Tamanho mediano das mensagens | 9 palavras |

## O que já foi feito

### Pré-processamento

Como o dataset foi traduzido automaticamente, boa parte do trabalho foi encontrar e corrigir problemas da tradução:

- remoção das colunas sem utilidade (indicador de revisão e modelo usado na tradução) e das linhas vazias;
- remoção de linhas que não eram mensagens de clientes (respostas do próprio GPT, comentários do revisor e respostas padrão do banco, encontradas entre os *outliers* de tamanho);
- remoção de anotações do revisor coladas no fim das frases e preenchimento de marcadores que ficaram no texto, como `[país]`;
- correção de um erro sistemático: *top up* (recarregar o cartão) foi traduzido como "abastecer", o que gerou frases sobre gasolina e tanque de combustível nas categorias de recarga;
- remoção de mensagens duplicadas e de mensagens que aparecem com mais de uma categoria, também ignorando diferenças de maiúsculas e pontuação.

### Análise exploratória

- O desbalanceamento entre categorias é moderado (de 71 a 225 mensagens por categoria). Por isso, a divisão entre treino e teste é estratificada e a métrica principal é o **F1-macro**, que dá o mesmo peso a todas as categorias.
- As mensagens são curtas (mediana de 9 palavras), e o tamanho varia conforme a categoria: pedidos simples são curtos, e relatos de problemas, longos.
- As negações (`não`, `nem`, `sem`) foram mantidas fora da lista de stopwords: elas mudam o sentido das frases e fazem parte do nome de vários rótulos, como `cartão_não_funcionando`.
- Categorias do mesmo tema (câmbio, identidade, transferências, recarga) compartilham as palavras mais frequentes e distintivas; a hipótese era que bigramas e trigramas ajudariam a separar parte delas (a modelagem mostrou que não). Essas categorias foram apontadas como as mais difíceis, o que a análise de erros confirmou depois.

### Modelagem

É um problema de **classificação supervisionada multiclasse**: todas as mensagens têm rótulo, o alvo é uma categoria e há 77 classes.

Divisão estratificada 80/20 (10.024 mensagens de treino e 2.506 de teste). Os hiperparâmetros são escolhidos só no treino, com validação cruzada estratificada de 5 folds (os mesmos para todos os modelos) e F1-macro como critério: `GridSearchCV` na maioria dos modelos e `Optuna` no XGBoost. O teste é usado uma única vez, no fim, e o melhor modelo é escolhido pelo F1-macro da validação cruzada, não pelo do teste.

Representações comparadas:

| Representação | Tipo | O que a comparação mostra |
|---|---|---|
| TF-IDF | esparsa, contagem de palavras | efeito do ajuste do vetorizador (n-gramas, stopwords, `min_df`, `sublinear_tf`) |
| [Serafim 335m](https://huggingface.co/PORTULAN/serafim-335m-portuguese-pt-sentence-encoder) | embedding, português | modelo de português vs. multilíngue |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | embedding, multilíngue | referência de tamanho para o e5-large |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | embedding, multilíngue | efeito do tamanho do modelo |

#### Resultados no teste

| Modelo | Representação | F1-macro (CV) | Acurácia | F1-macro | Ganho sobre o baseline |
|---|---|---|---|---|---|
| **SVM ajustado (RBF)** | **e5-large** | **0,9286** | **0,9286** | **0,9307** | **+6,53 pp** |
| Regressão Logística ajustada | e5-large | 0,9237 | 0,9238 | 0,9264 | +6,10 pp |
| SVM ajustado (polinomial) | e5-base | 0,9208 | 0,9226 | 0,9245 | +5,91 pp |
| Regressão Logística ajustada | e5-base | 0,9167 | 0,9210 | 0,9231 | +5,77 pp |
| SVM ajustado (RBF) | Serafim 335m | 0,9092 | 0,9170 | 0,9187 | +5,32 pp |
| Regressão Logística ajustada | Serafim 335m | 0,9034 | 0,9110 | 0,9119 | +4,64 pp |
| SVM ajustado (polinomial) | TF-IDF | 0,8830 | 0,8923 | 0,8922 | +2,67 pp |
| Regressão Logística ajustada | TF-IDF | 0,8756 | 0,8915 | 0,8906 | +2,52 pp |
| Random Forest ajustado | TF-IDF | 0,8487 | 0,8579 | 0,8558 | −0,97 pp |
| Regressão Logística (baseline, sem ajuste) | TF-IDF | 0,8441 | 0,8671 | 0,8654 | — |
| XGBoost ajustado (Optuna) | TF-IDF | 0,8228 | 0,8312 | 0,8313 | −3,42 pp |

#### Principais conclusões

- **A representação pesou mais que o classificador.** Os 6 modelos com embeddings ficaram acima de todos os modelos com TF-IDF: o pior deles (Regressão Logística + Serafim, 0,9119) supera o melhor com TF-IDF (SVM, 0,8922). Entre os encoders, o e5-large foi o melhor nos dois classificadores, e o modelo multilíngue superou o treinado só em português.
- **O ajuste de hiperparâmetros foi indispensável, principalmente nos embeddings.** O `C` foi o hiperparâmetro de maior efeito em todos os modelos. Sem ajuste, a Regressão Logística com os embeddings do e5 ficava *abaixo* do baseline com TF-IDF (−4,85 pp no e5-base); com ajuste, o ganho chegou a +12 pp.
- **No TF-IDF, mais simples foi melhor.** Unigramas venceram bigramas e trigramas (trigramas custam de 1,6 a 2,4 pp), e **remover stopwords piorou** os modelos em 1,3 a 1,5 pp, em 5 de 5 folds: palavras como "por", "que" e "minha" ajudam a separar intenções parecidas, como `por_que_verificar_identidade` e `verificar_minha_identidade`. `min_df`, `sublinear_tf` e o balanceamento de classes mudaram o F1-macro em 0,3 pp ou menos.
- **Os modelos de árvore não funcionaram bem com TF-IDF.** Random Forest e XGBoost ficaram abaixo do baseline, como esperado: as árvores dividem os dados uma palavra por vez e, com 77 classes, as folhas ficam com poucos exemplos.
- **O `StandardScaler` não ajudou nos embeddings** (−1,65 pp na Regressão Logística com e5-large), e no SVM os kernels RBF, polinomial e linear ficaram a menos de 0,5 pp um do outro.
- **A validação cruzada foi uma boa estimativa do teste.** O F1-macro do teste ficou um pouco acima do da validação cruzada em todos os modelos, e a ordem entre eles se manteve (só o Random Forest e o baseline trocam de posição).

### Avaliação do melhor modelo

O SVM + e5-large é comparado em detalhe com a Regressão Logística + e5-large (segundo melhor) e com o SVM + TF-IDF (melhor com TF-IDF).

- **Por categoria:** das 77 categorias, 11 não têm nenhum erro, 45 têm F1 de 0,90 ou mais e só 3 ficam abaixo de 0,80. A especificidade fica entre 0,996 e 1 em todas, então não separa as boas das ruins; precision, recall e F1 são as métricas que mostram os erros.
- **Desbalanceamento:** não prejudica o modelo. As médias macro e weighted são quase iguais, e o tamanho da categoria não tem relação com o F1 (correlação de −0,19).
- **Erros:** se concentram em **pares de categorias parecidas** (transferências, recarga, câmbio e verificação de identidade). Os 10% de pares mais parecidos nos embeddings (similaridade do cosseno entre os centroides) concentram 73,7% dos erros. Parte deles vem de mensagens ambíguas ou com rótulo duvidoso, que nenhum modelo resolveria só pelo texto.
- **Curva de aprendizado:** com só 30 mensagens por categoria, o SVM + e5-large já passa o SVM + TF-IDF treinado com o treino inteiro (130 por categoria, em média). A partir de cerca de 40 por categoria, rotular mais traz pouco ganho. Essa curva vai servir de referência na comparação com as LLMs: se uma LLM *zero-shot* chegar a 0,87 de F1-macro, por exemplo, ela equivale ao e5-large treinado com só 20 mensagens por categoria.
- **Projeção (PCA):** 161 componentes explicam 90% da variância dos embeddings (1024 dimensões), mas reduzir a dimensão não melhorou o modelo. Com 200 componentes, a perda é de só 0,4 pp, com treino mais rápido.

## Estrutura

```
├── data/
│   ├── raw/             # dataset original (treino e teste do Hugging Face)
│   ├── processed/       # df_01.csv (após a limpeza) e df_02.csv (após a análise exploratória)
│   ├── embeddings/      # embeddings gerados pelos notebooks (não versionados)
│   └── buscas/          # cache das buscas de hiperparâmetros (não versionado)
├── notebooks/
│   ├── 01_pre_processamento.ipynb
│   ├── 02_analise_exploratoria.ipynb
│   ├── 03_modelagem.ipynb
│   └── 04_avaliacao.ipynb
└── requirements.txt
```

## Como rodar

```bash
pip install -r requirements.txt
jupyter lab
```

Rode os notebooks em ordem, a partir da pasta `notebooks/`. Na primeira execução, o notebook de modelagem:

- baixa os três modelos de embedding do Hugging Face (cerca de 4 GB) e salva os embeddings em `data/embeddings/`;
- roda as buscas de hiperparâmetros e salva cada uma em `data/buscas/`. Algumas levam horas (principalmente o SVM e o XGBoost); a busca do XGBoost salva o progresso a cada tentativa e pode ser interrompida e retomada.

Nas execuções seguintes, embeddings e buscas são carregados do disco. O notebook de avaliação usa os embeddings e as buscas salvos pela modelagem, então precisa que o `03_modelagem.ipynb` tenha sido rodado antes.

Todos os notebooks estão salvos com as saídas, então dá para ler sem rodar.

## Ferramentas

Python, pandas, scikit-learn, XGBoost, Optuna, NLTK, Matplotlib, Seaborn e sentence-transformers. Na próxima etapa, APIs de LLMs.
