# Trabalho de Conclusão de Curso - Bacharelado em Matemática Aplicada e Computacional
**Habilitação em Estatística Econômica, IME-USP**

Este projeto explora a aplicação introdutória de modelos de séries temporais, combinando abordagens clássicas (econométricas) e modernas (redes neurais), focado em dados econômicos. O trabalho inclui a preparação de dados, aplicação e ajuste de modelos de séries temporais, e uma avaliação crítica dos resultados obtidos.

## Modelos Utilizados:
- **VAR (Vector Autoregression)**
- **VARMA (Vector Autoregressive Moving-Average)**
- **MLP (Multilayer Perceptron)**

O contexto econômico escolhido envolve dados de ações da bolsa de valores, agrupados em índices setoriais. Embora a aplicação destes modelos em dados econômicos seja intrigante, teorias econômicas estabelecidas alertam para as limitações de previsão de dados futuros baseando-se exclusivamente em informações de ações e modelos simplificados. Os resultados, portanto, demonstram limitações significativas para predições de mercado, servindo mais como um estudo introdutório sobre a aplicabilidade de modelos de séries temporais na economia.

## Dependências

Este projeto foi desenvolvido com as seguintes versões de ferramentas e bibliotecas, garantindo compatibilidade e estabilidade:

- Python: 3.11.3
- pandas: 1.5.3
- numpy: 1.26.4
- yfinance: 0.2.32
- yahoo_fin: 0.8.9.1
- keras: 2.14.0
- statsmodels: 0.14.0

Para instalar todas as dependências necessárias, execute o seguinte comando no seu ambiente Python:

```bash
pip install pandas==1.5.3 numpy==1.26.4 yfinance==0.2.32 yahoo_fin==0.8.9.1 keras==2.14.0 statsmodels==0.14.0
