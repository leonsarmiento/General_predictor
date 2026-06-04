"""
Preditor Geral de Fenômenos baseado em Entropia Máxima (MaxEnt)
================================================================
Aplicativo Streamlit para predição espacial de ocorrências de fenômenos
utilizando características gravimétricas e geológicas da região Centro-Sul do Brasil.

O modelo MaxEnt (Entropia Máxima) é um algoritmo de aprendizado de máquina que
estima a distribuição de probabilidade de ocorrência a partir de dados de presença,
combinando variáveis contínuas (gravimetria) e categóricas (geologia).

Referência: Phillips et al. (2006) - Maximum entropy modeling of species geographic distributions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from shapely.geometry import Point
import json
import os
import io
import tempfile

# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Preditor Geral de Fenômenos - MaxEnt",
    page_icon="🌍",
    layout="wide",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
REPO_URL = 'https://github.com/leonsarmiento/General_predictor'

# ══════════════════════════════════════════════════════════════
# CLASSE DO MODELO MAXENT
# ══════════════════════════════════════════════════════════════

class MaxEntModel:
    """
    Implementação do algoritmo de Entropia Máxima (MaxEnt) para modelagem
    preditiva de ocorrências com variáveis contínuas e categóricas.
    
    A distribuição de Gibbs: p(x) = exp(lambda . f(x)) / Z
    Otimizada via L-BFGS-B com regularização L1.
    """
    
    def __init__(self, beta=2.0, tol=1e-6, max_iter=500):
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.lambdas = None
        self.feature_importance = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None):
        """Ajusta o modelo MaxEnt aos dados."""
        X_pres = X[y == 1]
        X_bg = X[y == 0]
        n_features = X.shape[1]
        self.feature_names = feature_names or [f'f{i}' for i in range(n_features)]
        
        lambdas0 = np.random.randn(n_features) * 0.01
        
        result = minimize(
            fun=self._neg_log_likelihood,
            x0=lambdas0,
            args=(X_pres, X_bg),
            method='L-BFGS-B',
            jac=self._gradient,
            options={'maxiter': self.max_iter, 'ftol': self.tol, 'disp': False}
        )
        
        self.lambdas = result.x
        raw = np.abs(self.lambdas)
        self.feature_importance = raw / raw.sum() * 100.0
        self.converged = result.success
        self.nit = result.nit
        return self
    
    def _neg_log_likelihood(self, lambdas, X_pres, X_bg):
        X_all = np.vstack([X_pres, X_bg])
        linear_all = X_all @ lambdas
        log_z = logsumexp(linear_all)
        linear_pres = X_pres @ lambdas
        pres_ll = np.sum(linear_pres - log_z)
        l1 = self.beta * np.sum(np.abs(lambdas))
        return -pres_ll + l1
    
    def _gradient(self, lambdas, X_pres, X_bg):
        X_all = np.vstack([X_pres, X_bg])
        linear_all = X_all @ lambdas
        log_z = logsumexp(linear_all)
        probs = np.exp(linear_all - log_z)
        expected = probs @ X_all
        observed = X_pres.sum(axis=0)
        grad = -(observed - len(X_pres) * expected.ravel())
        grad += self.beta * np.sign(lambdas)
        return grad
    
    def predict_proba(self, X):
        """Retorna probabilidade normalizada [0, 1]."""
        linear = X @ self.lambdas
        log_z = logsumexp(linear)
        log_probs = linear - log_z
        probs = np.exp(log_probs)
        return probs / probs.max()


# ══════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Carrega dados do grid e mapeamentos."""
    # Grid CSV
    df = pd.read_csv(os.path.join(DATA_DIR, 'grid_data.csv'))
    
    # GeoJSON
    grid_gdf = gpd.read_file(os.path.join(DATA_DIR, 'grid_cells.geojson'))
    
    # Binary geo features (sparse)
    sparse_data = np.load(os.path.join(DATA_DIR, 'geo_binary_sparse.npz'),
                          allow_pickle=True)
    geo_sparse = csr_matrix(
        (sparse_data['data'], sparse_data['indices'], sparse_data['indptr']),
        shape=tuple(sparse_data['shape'])
    )
    sparse_ids = sparse_data['ids']
    
    # Column names
    with open(os.path.join(DATA_DIR, 'geo_columns.json'), 'r') as f:
        geo_columns = json.load(f)
    
    # Mappings
    with open(os.path.join(DATA_DIR, 'unit_mapping.json'), 'r', encoding='utf-8') as f:
        unit_map = json.load(f)
    with open(os.path.join(DATA_DIR, 'symb_mapping.json'), 'r', encoding='utf-8') as f:
        symb_map = json.load(f)
    
    # Metadata
    with open(os.path.join(DATA_DIR, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    
    return df, grid_gdf, geo_sparse, sparse_ids, geo_columns, unit_map, symb_map, meta


def parse_uploaded_points(uploaded_file):
    """Analisa arquivo CSV/TXT com coordenadas lat,lon."""
    content = uploaded_file.getvalue().decode('utf-8')
    lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
    
    points = []
    for line in lines:
        # Skip header
        if 'lat' in line.lower() or 'lon' in line.lower():
            continue
        parts = line.replace('\t', ',').split(',')
        if len(parts) >= 2:
            try:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                points.append((lat, lon))
            except ValueError:
                continue
    return points


def match_points_to_grid(points, grid_gdf):
    """Encontra a célula do grid que contém cada ponto.
    
    Retorna:
        matched_ids: lista de IDs das células correspondentes (apenas pontos dentro do grid)
        discarded: lista de tuplas (indice, lat, lon, motivo) dos pontos descartados
    """
    matched_ids = []
    discarded = []
    
    pts_gdf = gpd.GeoDataFrame(
        {'lat': [p[0] for p in points], 'lon': [p[1] for p in points]},
        geometry=[Point(p[1], p[0]) for p in points],
        crs="EPSG:4326"
    )
    
    # Verificar bounds do grid
    grid_bounds = grid_gdf.total_bounds  # minx, miny, maxx, maxy
    
    joined = gpd.sjoin(pts_gdf, grid_gdf[['id', 'geometry']], how='left', predicate='within')
    
    for i, row in joined.iterrows():
        lat, lon = row['lat'], row['lon']
        
        # Verificar se está dentro dos bounds do grid
        if (lon < grid_bounds[0] or lon > grid_bounds[2] or
            lat < grid_bounds[1] or lat > grid_bounds[3]):
            discarded.append((i, lat, lon, "Coordenada fora da area de cobertura do grid"))
            continue
        
        if pd.notna(row['id']):
            matched_ids.append(int(row['id']))
        else:
            # Ponto dentro dos bounds mas nao caiu em nenhuma celula (gap no grid)
            discarded.append((i, lat, lon, "Ponto nao interceptou nenhuma celula do grid"))
    
    return matched_ids, discarded


def build_feature_matrix(df, geo_sparse, sparse_ids, matched_ids):
    """Constrói a matriz de features para o modelo."""
    continuous_cols = [
        'gravity_min', 'gravity_max', 'gravity_mean',
        'freeair_min', 'freeair_max', 'freeair_mean',
        'bouguer_min', 'bouguer_max', 'bouguer_mean',
    ]
    
    # Filter to cells with data
    df_valid = df.dropna(subset=['gravity_mean']).copy()
    valid_ids = df_valid['id'].values
    
    # Continuous features
    X_cont = df_valid[continuous_cols].values.astype(np.float64)
    
    # Scale
    scaler = StandardScaler()
    X_cont = scaler.fit_transform(X_cont)
    
    # Match sparse geo rows to valid df rows
    id_to_sparse_idx = {int(sid): i for i, sid in enumerate(sparse_ids)}
    valid_sparse_idx = [id_to_sparse_idx.get(int(vid), -1) for vid in valid_ids]
    
    # Build geo matrix for valid rows
    geo_rows = []
    for idx in valid_sparse_idx:
        if idx >= 0:
            geo_rows.append(geo_sparse[idx].toarray().flatten())
        else:
            geo_rows.append(np.zeros(geo_sparse.shape[1]))
    X_geo = np.array(geo_rows, dtype=np.float64)
    
    # Combine
    X = np.hstack([X_cont, X_geo])
    
    # Presence labels - check that matched cells actually have data
    valid_id_set = set(df_valid['id'].values)
    data_missing_ids = [mid for mid in matched_ids if mid not in valid_id_set]
    y = np.zeros(len(df_valid), dtype=np.float64)
    for mid in matched_ids:
        if mid in valid_id_set:
            mask = df_valid['id'].values == mid
            y[mask] = 1.0
    
    feature_names = continuous_cols + [f'geo_{i}' for i in range(X_geo.shape[1])]
    
    return X, y, df_valid, scaler, feature_names, continuous_cols, data_missing_ids


def create_prediction_map(grid_gdf, df_valid, probs, points):
    """Cria o mapa de predição."""
    import math
    
    fig, ax = plt.subplots(figsize=(30, 30), dpi=100)
    fig.patch.set_facecolor('#f5f5f0')
    ax.set_facecolor('#f5f5f0')
    
    # Merge predictions to grid
    pred_df = df_valid[['id']].copy()
    pred_df['prob'] = probs
    grid_pred = grid_gdf.merge(pred_df, on='id', how='left')
    grid_pred = grid_pred.to_crs(epsg=3857)
    
    # Bounds
    valid = grid_pred[grid_pred['prob'].notna()]
    bounds = valid.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.05
    pad_y = (bounds[3] - bounds[1]) * 0.05
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    
    # Basemap
    try:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=8)
    except:
        pass
    
    # Colormap
    cmap = LinearSegmentedColormap.from_list('suitability', [
        '#1a1a2e', '#1b4965', '#5ab77e', '#f0e448', '#f5a623', '#d62839'
    ], N=256)
    
    low = valid[valid['prob'] < 0.5]
    high = valid[valid['prob'] >= 0.5]
    
    if len(low) > 0:
        low.plot(column='prob', cmap=cmap, linewidth=0.0, edgecolor='none',
                 alpha=0.15, ax=ax, zorder=3, vmin=0, vmax=1)
    if len(high) > 0:
        high.plot(column='prob', cmap=cmap, linewidth=0.2, edgecolor='#333',
                  alpha=0.55, ax=ax, zorder=4, vmin=0, vmax=1)
    
    # Plot input points
    def lonlat_to_webmerc(lat, lon):
        x = lon * 20037508.34 / 180.0
        y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
        return x, y * 20037508.34 / 180.0
    
    for i, (lat, lon) in enumerate(points):
        x, y = lonlat_to_webmerc(lat, lon)
        ax.scatter(x, y, c='#FF6D00', marker='*', s=300, zorder=7,
                   edgecolors='white', linewidths=2.0)
        ax.annotate(f"Ponto {i+1}", xy=(x, y), xytext=(15, 15),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#333', alpha=0.9),
                    fontsize=9, fontweight='bold', color='#333', zorder=10)
    
    ax.set_title("Mapa de Suscetibilidade - Modelo MaxEnt\nVariaveis Gravimetricas e Geologicas",
                 fontsize=16, fontweight='bold', color='#333', pad=15)
    
    # Repo URL below title, above legend
    ax.text(0.5, 1.005, REPO_URL,
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9, color='#666', style='italic',
            fontfamily='monospace')
    
    # Legend with repo info
    legend_elements = [
        plt.Line2D([0],[0], marker='o', color='w', label='Ponto de ocorrencia',
            markerfacecolor='#FF6D00', markersize=10, linestyle='None', markeredgecolor='white'),
        mpatches.Patch(facecolor='#d62839', edgecolor='#333', alpha=0.6, label='Alta suscetibilidade'),
        mpatches.Patch(facecolor='#5ab77e', edgecolor='#333', alpha=0.4, label='Media suscetibilidade'),
        mpatches.Patch(facecolor='#1b4965', edgecolor='none', alpha=0.2, label='Baixa suscetibilidade'),
    ]
    legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=10, facecolor='white',
        edgecolor='#999', labelcolor='#333', framealpha=0.92,
        title=f"Fonte: {REPO_URL}", title_fontsize=8)
    legend.get_title().set_color('#666')
    
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#f5f5f0', edgecolor='none')
    plt.close()
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════

st.title("Preditor Geral de Fenômenos")
st.subheader("Modelo de Entropia Maxima baseado em Caracteristicas Gravimetricas e Geologicas")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Configuracoes")
    beta = st.slider("Regularizacao (beta)", 0.5, 5.0, 2.0, 0.5,
                     help="Controla a complexidade do modelo. Valores maiores = mais regularizacao.")
    max_iter = st.selectbox("Iteracoes maximas", [200, 500, 1000], index=1)
    
    st.markdown("---")
    st.markdown("""
    **Sobre o modelo:**
    
    Este aplicativo utiliza um modelo de Entropia Maxima (MaxEnt) para 
    prever areas de suscetibilidade a fenomenos com base em:
    - **9 variaveis gravimetricas** (Gravidade, FreeAir, Bouguer: min, max, media)
    - **~2000 variaveis geologicas** (unidades litoestratigraficas e simbolos)
    
    O modelo aprende com os pontos de ocorrencia fornecidos e estima 
    a probabilidade de ocorrencia em toda a area de estudo.
    
    **Area de estudo:** Regiao Centro-Sul do Brasil
    (~20 de latitude x ~21 de longitude)
    """)

# Main content
st.markdown("""
### Upload de Dados

Carregue um arquivo CSV ou TXT com pares de coordenadas (latitude, longitude) em WGS84 (EPSG:4326).

**Formato esperado:** Uma coordenada por linha, separada por virgula:
""")

st.code("-26.700800, -49.831017\n-22.973167, -49.800199\n-18.227172, -43.264358", language="plaintext")

# ── Botão de instruções ──
if st.button("Como usar este aplicativo", use_container_width=True):
    st.markdown("""
    ---
    ## Instruções de Uso
    
    ### 1. Formato do arquivo de entrada
    Prepare um arquivo **CSV** ou **TXT** contendo pares de coordenadas (latitude, longitude),
    um por linha, separados por vírgula. O sistema de coordenadas deve ser **WGS84 (EPSG:4326)**.
    
    **Exemplo de arquivo válido:**
    ```
    -26.700800, -49.831017
    -22.973167, -49.800199
    -18.227172, -43.264358
    -25.344483, -49.733521
    -14.066863, -47.484732
    ```
    
    **Regras:**
    - Latitude primeiro, longitude depois
    - Separador: vírgula
    - Coordenadas em graus decimais (ex: `-22.973`, não `22°58'23"S`)
    - Sem cabeçalho obrigatório (se houver, será ignorado)
    - **Mínimo recomendado:** 5 pontos para resultados significativos
    
    ### 2. Área de cobertura
    O grid cobre a região Centro-Sul do Brasil:
    - **Latitude:** -30.3° a -10.5°
    - **Longitude:** -57.0° a -36.0°
    
    Pontos fora desta área não serão processados.
    
    ### 3. Execução
    1. Carregue o arquivo de coordenadas no campo acima
    2. Verifique se os pontos foram lidos corretamente (expanda "Visualizar pontos carregados")
    3. Ajuste os parâmetros na barra lateral se desejar
    4. Clique em **"Executar Modelo"**
    5. Aguarde o processamento (pode levar alguns minutos)
    
    ### 4. Resultados
    - **AUC:** métrica de qualidade do modelo (0.5 = aleatório, 1.0 = perfeito)
    - **Importância das variáveis:** quais features mais contribuíram para a predição
    - **Mapa de suscetibilidade:** visualização espacial das probabilidades
    - **Download CSV:** tabela com probabilidade por célula do grid
    
    ### 5. Parâmetros ajustáveis (barra lateral)
    - **Regularização (beta):** controla a complexidade do modelo. Valores maiores = modelo mais conservador. Padrão: 2.0
    - **Iterações máximas:** mais iterações = melhor convergência, mas mais lento. Padrão: 500
    
    ### 6. Variáveis utilizadas pelo modelo
    - **Gravimétricas (9):** Gravidade, FreeAir, Bouguer (mínimo, máximo, média por célula)
    - **Geológicas (~2000):** Unidades litoestratigráficas e símbolos geológicos (presença/ausência por célula)
    
    ---
    """)

uploaded_file = st.file_uploader(
    "Selecione o arquivo com coordenadas (WGS84 obrigatorio)",
    type=['csv', 'txt'],
    help="Formato: latitude, longitude por linha"
)

# Load data button
col1, col2 = st.columns([1, 3])

with col1:
    run_model = st.button("Executar Modelo", type="primary", use_container_width=True)

if uploaded_file is not None:
    # Parse points
    points = parse_uploaded_points(uploaded_file)
    
    if len(points) == 0:
        st.error("Nenhum ponto valido encontrado. Verifique o formato do arquivo.")
    else:
        st.success(f"**{len(points)} pontos** carregados com sucesso.")
        
        # Show parsed points
        with st.expander("Visualizar pontos carregados"):
            pts_df = pd.DataFrame(points, columns=['Latitude', 'Longitude'])
            st.dataframe(pts_df, use_container_width=True)
        
        if run_model:
            with st.spinner("Carregando dados do grid..."):
                df, grid_gdf, geo_sparse, sparse_ids, geo_columns, unit_map, symb_map, meta = load_data()
            
            with st.spinner("Matchando pontos ao grid..."):
                matched_ids, discarded = match_points_to_grid(points, grid_gdf)
                
                # Report discarded points
                if discarded:
                    st.warning(f"**{len(discarded)} ponto(s) descartado(s):**")
                    for idx, lat, lon, reason in discarded:
                        st.warning(f"  Ponto {idx+1} ({lat:.4f}, {lon:.4f}): {reason}")
                
                if len(matched_ids) == 0:
                    st.error("Nenhum ponto valido restante apos filtragem. Todos os pontos estao fora da area de cobertura do grid. O modelo nao pode ser executado.")
                    st.stop()
                
                st.info(f"**{len(matched_ids)} de {len(points)} pontos** validos dentro do grid.")
            
            with st.spinner("Construindo matriz de features..."):
                X, y, df_valid, scaler, feature_names, continuous_cols, data_missing_ids = build_feature_matrix(
                    df, geo_sparse, sparse_ids, matched_ids
                )
                
                # Report cells matched but without predictor data
                if data_missing_ids:
                    st.warning(f"**{len(data_missing_ids)} ponto(s)** caiu em celulas sem dados gravimetricos e foram descartados.")
                
                n_presence = int(y.sum())
                n_background = int((y == 0).sum())
            
            # Abort if no presence points remain
            if n_presence == 0:
                st.error("Nenhum ponto de ocorrencia com dados completos restante. Verifique se os pontos estao dentro da area com cobertura de dados.")
                st.stop()
            
            st.markdown("---")
            st.subheader("Resultado do Modelo")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Pontos de ocorrencia", n_presence)
            with col_b:
                st.metric("Celulas de fundo", n_background)
            with col_c:
                st.metric("Variaveis", X.shape[1])
            
            with st.spinner("Ajustando modelo MaxEnt... Isso pode levar alguns minutos."):
                model = MaxEntModel(beta=beta, tol=1e-6, max_iter=max_iter)
                model.fit(X, y, feature_names=feature_names)
                probs = model.predict_proba(X)
            
            # AUC
            if n_presence > 0 and n_background > 0:
                auc = roc_auc_score(y, probs)
                st.metric("AUC (treino)", f"{auc:.4f}")
            
            if model.converged:
                st.success(f"Modelo convergiu em {model.nit} iteracoes.")
            else:
                st.warning("Modelo nao convergiu completamente. Considere aumentar o numero de iteracoes.")
            
            # Feature importance (continuous only)
            st.subheader("Importancia das Variaveis Gravimetricas")
            n_cont = len(continuous_cols)
            imp_cont = pd.DataFrame({
                'Variavel': continuous_cols,
                'Lambda': model.lambdas[:n_cont],
                'Contribuicao (%)': model.feature_importance[:n_cont],
            }).sort_values('Contribuicao (%)', ascending=False)
            
            st.dataframe(imp_cont, use_container_width=True)
            
            # Top 10 geo features
            st.subheader("Top 10 Variaveis Geologicas")
            geo_imp = pd.DataFrame({
                'Variavel': geo_columns,
                'Lambda': model.lambdas[n_cont:],
                'Contribuicao (%)': model.feature_importance[n_cont:],
            }).sort_values('Contribuicao (%)', ascending=False).head(10)
            
            st.dataframe(geo_imp, use_container_width=True)
            
            # Map
            st.subheader("Mapa de Suscetibilidade")
            with st.spinner("Gerando mapa..."):
                map_buf = create_prediction_map(grid_gdf, df_valid, probs, points)
                st.image(map_buf, use_container_width=True)
            
            # Download predictions
            st.subheader("Download dos Resultados")
            pred_download = df_valid[['id', 'row_index', 'col_index']].copy()
            pred_download['probabilidade'] = probs
            pred_download['latitude_centroide'] = df_valid.apply(
                lambda r: grid_gdf[grid_gdf['id'] == r['id']].geometry.centroid.y.values[0]
                if len(grid_gdf[grid_gdf['id'] == r['id']]) > 0 else None, axis=1
            )
            pred_download['longitude_centroide'] = df_valid.apply(
                lambda r: grid_gdf[grid_gdf['id'] == r['id']].geometry.centroid.x.values[0]
                if len(grid_gdf[grid_gdf['id'] == r['id']]) > 0 else None, axis=1
            )
            
            csv_buf = io.StringIO()
            pred_download.to_csv(csv_buf, index=False)
            
            st.download_button(
                label="Baixar predicoes (CSV)",
                data=csv_buf.getvalue(),
                file_name="predicoes_maxent.csv",
                mime="text/csv"
            )
else:
    st.info("Carregue um arquivo com coordenadas para iniciar a analise.")
    st.markdown("""
    **Formato do arquivo:**
    - CSV ou TXT
    - Uma coordenada por linha
    - Colunas: `latitude, longitude`
    - Sistema de coordenadas: **WGS84 (EPSG:4326)** obrigatorio
    
    **Exemplo:**
    ```
    -26.700800, -49.831017
    -22.973167, -49.800199
    -18.227172, -43.264358
    ```
    """)
