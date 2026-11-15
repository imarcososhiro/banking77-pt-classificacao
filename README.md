# Classificação de Intenções Bancárias com NLP e LLMs

Como um banco que recebe milhares de mensagens de clientes por dia pode encaminhar cada uma automaticamente para a equipe certa, de maneira efetiva? E quando vale a pena pagar por uma LLM em vez de usar um modelo clássico de Machine Learning?

Neste projeto pessoal, construo um classificador de intenções em Python com Machine Learning e o comparo com classificações feitas por LLMs, usando o dataset **Banking77 em português** para responder a essas perguntas. São cerca de 13 mil mensagens de clientes em **77 categorias de atendimento**, como `chegada_do_cartão`, `alterar_pin` e `cancelar_transferência`.

> **Status:** em andamento.

## A ideia

O projeto compara duas formas de resolver o mesmo problema:

| | Machine Learning clássico | LLMs via API |
|---|---|---|
| **Como funciona** | um modelo treinado com as mensagens rotuladas | a LLM recebe a mensagem e a lista de categorias num prompt |
| **Variações testadas** | Regressão Logística e SVM Linear, sobre TF-IDF e sobre embeddings de sentenças | prompts zero-shot (só as categorias) e few-shot (com exemplos) |
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

- [x] **Pré-processamento** dos textos ([`01_processamento.ipynb`](notebooks/01_processamento.ipynb))
- [x] **Análise exploratória** com visualização de dados e técnicas de PLN: tokenização, stopwords e TF-IDF com n-gramas ([`02_analise_exploratoria.ipynb`](notebooks/02_analise_exploratoria.ipynb))
- [x] **Baseline e Regressão Logística sobre TF-IDF** ([`03_modelagem.ipynb`](notebooks/03_modelagem.ipynb))
- [ ] **Regressão Logística e SVM Linear sobre embeddings** de sentenças (os embeddings já estão gerados)
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

- remoção de linhas que não eram mensagens de clientes (respostas do próprio GPT e comentários do revisor);
- remoção de anotações do revisor coladas no fim das frases e preenchimento de marcadores que ficaram no texto, como `[país]`;
- correção de um erro sistemático: *top up* (recarregar o cartão) foi traduzido como "abastecer", o que gerou frases sobre gasolina e tanque de combustível nas categorias de recarga;
- remoção de mensagens duplicadas e de mensagens que aparecem com mais de uma categoria, também ignorando diferenças de maiúsculas e pontuação.

### Análise exploratória

- O desbalanceamento entre categorias é moderado. Por isso, a divisão entre treino e teste é estratificada e a métrica principal é o **F1-macro**, que dá o mesmo peso a todas as categorias.
- As negações (`não`, `nem`, `sem`) foram mantidas fora da lista de stopwords: elas mudam o sentido das frases e fazem parte do nome de vários rótulos, como `cartão_não_funcionando`.
- Categorias do mesmo tema compartilham as palavras mais frequentes; bigramas e trigramas ajudam a separar parte delas.

### Modelagem com TF-IDF

Divisão estratificada 80/20 (10.024 mensagens de treino e 2.506 de teste). Os hiperparâmetros são escolhidos com `GridSearchCV` e validação cruzada estratificada de 5 folds, feitos só no treino; o teste é usado uma única vez, no fim.

| Modelo | Representação | Acurácia | F1-macro | Ganho sobre o baseline |
|---|---|---|---|---|
| Regressão Logística ajustada | TF-IDF com stopwords | 0,8899 | **0,8892** | +2,37 pp |
| Regressão Logística ajustada | TF-IDF sem stopwords | 0,8783 | 0,8761 | +1,07 pp |
| Regressão Logística (baseline, sem ajuste) | TF-IDF com stopwords | 0,8671 | 0,8654 | — |

**Remover stopwords piorou o modelo** em cerca de 1,3 pp, tanto na validação cruzada quanto no teste. Palavras como "por", "que" e "minha" ajudam a separar intenções parecidas, como `por_que_verificar_identidade` e `verificar_minha_identidade`.

### Embeddings de sentenças

Os próximos modelos clássicos usam embeddings, que representam o significado da frase inteira, e não só as palavras que ela contém:

| Modelo de embedding | Tipo | O que a comparação mostra |
|---|---|---|
| [Serafim 335m](https://huggingface.co/PORTULAN/serafim-335m-portuguese-pt-sentence-encoder) | português | modelo de português vs. multilíngue |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | multilíngue | referência de tamanho para o e5-large |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | multilíngue | efeito do tamanho do modelo |

## Estrutura

```
├── data/
│   ├── raw/             # dataset original (treino e teste do Hugging Face)
│   ├── processed/       # df_01.csv (após a limpeza) e df_02.csv (após a análise exploratória)
│   └── embeddings/      # embeddings gerados pelos notebooks (não versionados)
├── notebooks/
│   ├── 01_processamento.ipynb
│   ├── 02_analise_exploratoria.ipynb
│   └── 03_modelagem.ipynb
└── requirements.txt
```

## Como rodar

```bash
pip install -r requirements.txt
jupyter lab
```

Rode os notebooks em ordem, a partir da pasta `notebooks/`. Na primeira execução, o notebook de modelagem baixa os três modelos de embedding do Hugging Face (cerca de 4 GB) e salva os embeddings em `data/embeddings/`; nas execuções seguintes eles são carregados do disco.

## Ferramentas

Python, pandas, scikit-learn, NLTK, Matplotlib, Seaborn, sentence-transformers e APIs de LLMs.
