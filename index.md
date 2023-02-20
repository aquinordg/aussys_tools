# Package mltools

Python package [mltools](https://github.com/aquinordg/mltools): Machine Learning Tools.

<div style="text-align: justify"> Essa ferramenta auxilia a análise e tomada de decisão em cenários de classificação binária, especialmente em aplicações de detecção de objetos a deriva no mar. </div>

## Instalação

```markdown
pip install git+https://github.com/aquinordg/mltools.git
```

## Funções

### Relatório baseado em limiar de probabilidade

<div style="text-align: justify"> Esta função informa a quantidade de imagens 'no sea' identificadas de forma equivocadas e despercebidas, dado determinado limiar de probabilidade estabelecido como parâmetro. Para isso, a função utiliza uma matriz de probabilidade de pertencimento e os valores esperados, dada determinada classe, além de algumas informações da missão. Os resultados podem ser mostrados por relatório impresso em tela ou adquiridos de forma direta. </div>

#### Chamada

```markdown
from mltools import aussys_rb_thres
```

#### Aplicação

```markdown

aussys_rb_thres(predict_proba,
                expected,
                mission_duration,
                captures_per_second,
                n_sea_exp,
                threshold,
                print_mode=True)

```

#### Parâmetros

* **predict_proba**: _array-like of float in range [0.0, 1.0] and shape m_<br/>
> Probabilidades dos exemplos pertencerem à classe alvo.

* **expected**: _array-like of bool and shape m_<br/>
> Vetor booleano de dimensão m: classes esperadas de cada um dos exemplos.

* **mission_duration**: _int_<br/>
> Duração da missão em segundos.

* **captures_per_second**: _int_<br/>
> Número de capturas por segundo.

* **n_sea_exp**: _int_<br/>
> Número esperado de imagens 'sea' para 1 'nosea'. <br/>
> **Ex.:** 1('nosea') : _n_sea_exp_.

* **threshold**: _float in range [0.0, 1.0]_<br/>
> Limiar de probabilidade de pertencimento definido pelo usuário.

* **print_mode**: _bool_<br/>
> Relatório em tela (True) ou valores diretos (False).

#### _Retornos_

* **sea_fpr**: _int_<br/>
> Quantidade de imagens 'no sea' identificadas de forma equivocada.

* **nosea_fnr**: _int_<br/>
> Total de imagens 'no sea' que deverão passar despercebidas.

### Relatório baseado na quantidade de imagens

<div style="text-align: justify"> Esta função informa quantidades atualizadas de imagens 'nosea' identificadas de forma equivocada ou passadas despercebidas consideradas aceitáveis e o limiar adequado, conforme a entrada escolhida. Assim como a função anterior, é utilizada a matriz de probabilidade de pertencimento e os valores esperados, dada determinada classe, além de algumas informações da missão. O método encontra o novo limiar por meio de uma busca gulosa em todos os cenários possíveis a partir de um parâmetro sensibilidade. Os resultados podem ser mostrados por relatório impresso em tela ou adquiridos de forma direta. </div>

#### Chamada

```markdown
from mltools import aussys_rb_images
```

#### Aplicação
```markdown

aussys_rb_images(predict_proba,
                 expected,
                 mission_duration,
                 captures_per_second,
                 n_sea_exp,
                 sea_fpr=None,
                 nosea_fnr=None,
                 print_mode=True)

```
#### Parâmetros

**predict_proba**: _array-like of float in range [0.0, 1.0] and shape m_<br/>
Probabilidades dos exemplos pertencerem à classe alvo.

**expected**: _array-like of bool and shape m_<br/>
Vetor booleano de dimensão m: classes esperadas de cada um dos exemplos.

**mission_duration**: _int_<br/>
Duração da missão em segundos.

**captures_per_second**: _int_<br/>
Número de capturas por segundo.

**n_sea_exp**: _int_<br/>
Número esperado de imagens 'sea' para 1 'nosea'.
Ex.: 1(nosea):n_sea_exp.

**sen**: _float in range [0.0, 1.0]_<br/>
Sensibilidade, diretamente proporcional à precisão do método e tempo de processamento.

**sea_fpr**: _int_<br/>
Quantidade de imagens ‘no sea’ em que é aceitável serem identificadas de forma equivocada.

**nosea_fnr**: _int_<br/>
Total de imagens ‘no sea’ em que é aceitável passarem despercebidas.

**print_mode**: _bool_<br/>
Relatório em tela (True) ou valores diretos (False).

#### _Retornos_

**new_sea_fpr, threshold**: _array-like_<br/>
Quantidade de imagens ‘no sea’ identificadas de forma equivocada, dado o valor ‘nosea_fnr’ indicado e o respectivo valor de limiar adequado.

**new_nosea_fnr, threshold**: _array-like_<br/>
Total de imagens ‘no sea’ que deverão passar despercebidas, dado o valor de ‘sea_fpr’ indicado e o limiar adequado.
