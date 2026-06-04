# Preditor Geral de Fenomenos

Preditor espacial de ocorrencias de fenomenos baseado em caracteristicas gravimetricas e geologicas para a regiao Centro-Sul do Brasil.

## Como usar

1. Instale as dependencias:
```bash
pip install streamlit geopandas matplotlib scipy scikit-learn
```

2. Execute o app:
```bash
streamlit run app.py
```

3. Carregue um arquivo CSV/TXT com pares de coordenadas (latitude, longitude) em WGS84:
```
-26.700800, -49.831017
-22.973167, -49.800199
```

4. Clique em "Executar Modelo" e aguarde o resultado.

## Dados

A pasta `data/` contem:
- `grid_data.csv` - Dados tabulados por celula (variaveis gravimetricas + geologia textual)
- `grid_cells.geojson` - Geometrias das celulas hexagonais do grid
- `geo_binary_sparse.npz` - Features geologicas binarias (esparsas)
- `geo_columns.json` - Nomes das colunas geologicas
- `unit_mapping.json` / `symb_mapping.json` - Mapeamentos de categorias
- `metadata.json` - Metadados do dataset

## Modelo

O algoritmo MaxEnt (Entropia Maxima) ajusta uma distribuicao de Gibbs:
```
p(x) = exp(lambda . f(x)) / Z
```
onde f(x) sao as features (gravimetricas contínuas + geologicas categoricas) e lambda sao os parametros aprendidos.

Otimizado via L-BFGS-B com regularizacao L1. Recomendado no minimo 5 pontos de ocorrencia para resultados significativos.
