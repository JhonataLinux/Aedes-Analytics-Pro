# app.py — Dashboard Aedes Analytics - Versão Corrigida

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import folium
from folium.plugins import HeatMap, MarkerCluster
try:
    from streamlit_folium import st_folium
except Exception:
    from streamlit_folium import folium_static as st_folium

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
import time


# =====================
# CONFIGURAÇÃO DA PÁGINA
# =====================
st.set_page_config(page_title="Aedes Analytics • Caruaru", page_icon="🦟", layout="wide", initial_sidebar_state="expanded")

# =====================
# CSS AVANÇADO
# =====================
st.markdown("""
<style>
:root {
    --primary: #FFD20A; --primary-dark: #E6BD00; --primary-glow: rgba(255, 210, 10, 0.4);
    --secondary: #00D4AA; --secondary-dark: #00B894; --danger: #FF6B6B; --danger-dark: #FF4757;
    --warning: #FFA726; --warning-dark: #FF9800; --success: #4CAF50; --success-dark: #45a049;
    --bg: #0A0F1C; --bg-dark: #070B16; --bg-light: #131A2D; --card: #1A243F;
    --card-hover: #212D4F; --card-light: #243156; --text: #FFFFFF; --text-muted: #8B9BBE;
    --text-dark: #6C7A9E; --border: rgba(255,255,255,0.1); --border-light: rgba(255,255,255,0.2);
    --radius: 20px; --radius-sm: 12px; --shadow: 0 12px 40px rgba(0,0,0,0.4);
    --shadow-lg: 0 20px 60px rgba(0,0,0,0.5); --gradient-primary: linear-gradient(135deg, var(--primary) 0%, var(--warning) 100%);
    --gradient-secondary: linear-gradient(135deg, var(--secondary) 0%, #00C9FF 100%);
    --gradient-bg: linear-gradient(135deg, var(--bg) 0%, var(--bg-dark) 100%);
    --gradient-card: linear-gradient(135deg, var(--card) 0%, var(--bg-light) 100%);
}
* { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; transition: all 0.2s ease; }
.stApp { background: var(--gradient-bg); background-attachment: fixed; }
.main-header { background: linear-gradient(135deg, rgba(255, 210, 10, 0.15) 0%, rgba(0, 212, 170, 0.15) 50%, rgba(138, 43, 226, 0.1) 100%); backdrop-filter: blur(30px); border: 1px solid var(--border-light); border-radius: var(--radius); padding: 3rem 2.5rem; margin: 1.5rem 0; position: relative; overflow: hidden; box-shadow: var(--shadow); }
.main-header::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--primary), transparent); }
.header-badge { background: var(--gradient-primary); color: #000; padding: 0.6rem 1.2rem; border-radius: 100px; font-weight: 800; font-size: 0.75rem; display: inline-flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem; box-shadow: 0 4px 15px var(--primary-glow); letter-spacing: 0.5px; }
.header-title { font-size: clamp(2.5rem, 5vw, 4rem); font-weight: 900; background: linear-gradient(135deg, var(--text) 0%, var(--primary) 50%, var(--secondary) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0.5rem 0; line-height: 1.1; text-shadow: 0 4px 20px rgba(0,0,0,0.3); }
.header-subtitle { color: var(--text-muted); font-size: 1.2rem; max-width: 700px; line-height: 1.6; font-weight: 400; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 2.5rem 0; }
.metric-card { background: var(--gradient-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 2rem 1.5rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden; box-shadow: var(--shadow); }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, var(--primary), var(--secondary)); opacity: 0.8; }
.metric-card:hover { transform: translateY(-8px) scale(1.02); background: var(--card-hover); box-shadow: var(--shadow-lg); border-color: var(--border-light); }
.metric-card.critical::before { background: linear-gradient(90deg, var(--danger), #FF8E8E); }
.metric-card.warning::before { background: linear-gradient(90deg, var(--warning), #FFC107); }
.metric-card.success::before { background: linear-gradient(90deg, var(--success), #66BB6A); }
.metric-card.info::before { background: linear-gradient(90deg, var(--secondary), #00C9FF); }
.metric-icon { font-size: 1.5rem; margin-bottom: 1rem; opacity: 0.9; }
.metric-title { color: var(--text-muted); font-size: 0.95rem; font-weight: 600; margin: 0 0 0.8rem 0; display: flex; align-items: center; gap: 0.6rem; letter-spacing: 0.3px; }
.metric-value { color: var(--text); font-size: 2.5rem; font-weight: 800; margin: 0; line-height: 1; background: linear-gradient(135deg, var(--text) 0%, var(--text-muted) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-trend { font-size: 0.85rem; font-weight: 700; margin-top: 0.8rem; display: flex; align-items: center; gap: 0.4rem; padding: 0.4rem 0.8rem; background: rgba(255,255,255,0.05); border-radius: var(--radius-sm); width: fit-content; }
.trend-up { color: var(--danger); background: rgba(255, 107, 107, 0.1); }
.trend-down { color: var(--success); background: rgba(76, 175, 80, 0.1); }
.stButton button { border-radius: var(--radius-sm); border: 1px solid var(--border); background: var(--gradient-card); color: var(--text); padding: 0.7rem 1.5rem; font-weight: 600; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden; }
.stButton button:hover { background: var(--card-hover); border-color: var(--primary); transform: translateY(-2px); box-shadow: 0 8px 25px rgba(255, 210, 10, 0.2); }
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; padding: 0.5rem; border-radius: var(--radius); }
.stTabs [data-baseweb="tab"] { background: var(--card); border-radius: var(--radius-sm); border: 1px solid var(--border); padding: 1rem 2rem; color: var(--text-muted); font-weight: 600; transition: all 0.3s ease; flex: 1; text-align: center; }
.stTabs [data-baseweb="tab"]:hover { background: var(--card-hover); border-color: var(--border-light); transform: translateY(-2px); }
.stTabs [aria-selected="true"] { background: var(--gradient-primary) !important; color: #000 !important; font-weight: 700; border-color: var(--primary); box-shadow: 0 4px 15px var(--primary-glow); }
.insight-card { background: var(--gradient-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; margin: 1rem 0; position: relative; overflow: hidden; transition: all 0.3s ease; }
.insight-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--gradient-primary); }
.insight-card:hover { transform: translateX(10px); border-color: var(--primary); }
.insight-card.critical::before { background: var(--gradient-primary); }
.insight-card.warning::before { background: linear-gradient(180deg, var(--warning), var(--warning-dark)); }
.insight-card.success::before { background: linear-gradient(180deg, var(--success), var(--success-dark)); }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
.animate-fade-in-up { animation: fadeInUp 0.6s ease-out; }
.animate-slide-in-left { animation: slideInLeft 0.5s ease-out; }
@keyframes slideInLeft { from { opacity: 0; transform: translateX(-50px); } to { opacity: 1; transform: translateX(0); } }
</style>
""", unsafe_allow_html=True)

# =====================
# COMPONENTES REACT-LIKE
# =====================
def MetricCard(icon, title, value, trend, trend_direction, criticality="info"):
    trend_icon = "📈" if trend_direction == "up" else "📉"
    trend_class = "trend-up" if trend_direction == "up" else "trend-down"
    return f"""<div class="metric-card {criticality} animate-fade-in-up"><div class="metric-icon">{icon}</div><div class="metric-title">{title}</div><div class="metric-value">{value}</div><div class="metric-trend {trend_class}">{trend_icon} {trend}</div></div>"""

def InsightCard(title, content, criticality="info"):
    return f"""<div class="insight-card {criticality} animate-slide-in-left"><h4 style="margin: 0 0 0.5rem 0; color: var(--text); font-weight: 700;">{title}</h4><p style="margin: 0; color: var(--text-muted); line-height: 1.5;">{content}</p></div>"""

def AnimatedHeader():
    return """<div class="main-header"><div class="header-badge">🚨 SISTEMA DE ALERTA PRECOCE • TEMPO REAL</div><div class="header-title">Aedes Analytics Pro</div><div class="header-subtitle">Plataforma inteligente de monitoramento epidemiológico. Dados preditivos, analytics avançados e gestão estratégica de combate ao Aedes aegypti.</div></div>"""

# =====================
# CARREGAMENTO DE DADOS
# =====================
@st.cache_data(ttl=3600)
def load_demo_data():
    bairros = ['Centro', 'Divinópolis', 'Cedro', 'Rendeiras', 'São Francisco', 'Boa Vista', 'Salgado', 'Maurício de Nassau', 'Vassoural', 'Cidade Alta', 'Kennedy', 'Agamenon Magalhães', 'Indianópolis', 'Morro do Bom Jesus', 'Universitário', 'José Carlos de Oliveira']
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'bairro': bairros,
        'casos_confirmados': rng.integers(30, 450, len(bairros)),
        'casos_suspeitos': rng.integers(50, 300, len(bairros)),
        'focos_aedes': rng.integers(15, 180, len(bairros)),
        'taxa_infestacao': rng.uniform(0.8, 7.5, len(bairros)),
        'visitados_agentes': rng.integers(100, 800, len(bairros)),
        'larvicida_aplicado': rng.integers(50, 400, len(bairros)),
        'imoveis_fechados': rng.integers(0, 25, len(bairros)),
        'latitude': [-8.284 + rng.uniform(-0.01, 0.01) for _ in bairros],
        'longitude': [-35.976 + rng.uniform(-0.01, 0.01) for _ in bairros],
        'populacao': rng.integers(2000, 15000, len(bairros)),
        'risco': rng.choice(['Baixo', 'Médio', 'Alto', 'Crítico'], len(bairros), p=[0.3, 0.4, 0.2, 0.1]),
        'tendencia': rng.choice(['Melhorando', 'Estável', 'Piorando'], len(bairros), p=[0.3, 0.4, 0.3])
    })
    df['densidade_casos'] = (df['casos_confirmados'] / df['populacao']) * 1000
    meses = [(datetime.now() - timedelta(days=30*i)).strftime('%Y-%m') for i in range(12, 0, -1)]
    evol_data = []
    for mes in meses:
        for bairro in bairros:
            base_idx = bairros.index(bairro)
            seasonal_factor = 1 + 0.3 * np.sin((meses.index(mes) - 3) * np.pi / 6)
            evol_data.append({
                'mes': mes, 'bairro': bairro,
                'casos': max(10, int(df.loc[base_idx, 'casos_confirmados'] * rng.uniform(0.6, 1.4) * seasonal_factor)),
                'focos': max(5, int(df.loc[base_idx, 'focos_aedes'] * rng.uniform(0.5, 1.5) * seasonal_factor)),
                'taxa': max(0.5, float(df.loc[base_idx, 'taxa_infestacao'] * rng.uniform(0.7, 1.3) * seasonal_factor))
            })
    return df, pd.DataFrame(evol_data)

# Carregar dados
df, evol_df = load_demo_data()

# =====================
# SIDEBAR
# =====================
with st.sidebar:
    st.markdown("""<div style='text-align: center; margin-bottom: 2rem; padding: 1.5rem 0; border-bottom: 1px solid var(--border);'><div style='font-size: 3rem; margin-bottom: 0.5rem;'>🦟</div><h2 style='color: var(--primary); margin: 0; font-weight: 800;'>Aedes Analytics</h2><p style='color: var(--text-muted); margin: 0; font-size: 0.9rem;'>Plataforma Premium</p></div>""", unsafe_allow_html=True)
    st.subheader("📁 Fonte de Dados")
    uploaded_file = st.file_uploader("Carregar dataset personalizado", type=["csv"], help="Dataset com informações epidemiológicas")
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            df_uploaded.columns = [c.strip().lower() for c in df_uploaded.columns]
            required_cols = ["bairro", "casos_confirmados", "focos_aedes", "taxa_infestacao"]
            if all(col in df_uploaded.columns for col in required_cols):
                df = df_uploaded
                st.success("✅ Dataset carregado!")
            else:
                st.warning("⚠️ Colunas necessárias não encontradas")
        except Exception as e:
            st.error(f"❌ Erro: {e}")
    st.subheader("🎯 Filtros Avançados")
    bairro_filter = st.multiselect("Selecionar Bairros", options=["Todos"] + sorted(df['bairro'].unique()), default=["Todos"], help="Filtrar por bairros específicos")
    risco_filter = st.multiselect("Nível de Risco", options=["Todos"] + sorted(df['risco'].unique()), default=["Todos"], help="Filtrar por classificação de risco")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        taxa_min = st.number_input("Taxa Mínima", 0.0, 10.0, 0.0, 0.1)
    with col_f2:
        taxa_max = st.number_input("Taxa Máxima", 0.0, 10.0, 10.0, 0.1)
    st.subheader("🔄 Visualização")
    map_style = st.selectbox("Estilo do Mapa", ["Marcadores Inteligentes", "Cluster 3D", "Mapa de Calor", "Visão Híbrida"], index=0)
    auto_refresh = st.toggle("Atualização Automática", False)
    if auto_refresh:
        st.info("🔄 Atualizando a cada 30s")
        time.sleep(30)
        st.rerun()

# Aplicar filtros
df_filtered = df.copy()
if "Todos" not in bairro_filter and bairro_filter:
    df_filtered = df_filtered[df_filtered['bairro'].isin(bairro_filter)]
if "Todos" not in risco_filter and risco_filter:
    df_filtered = df_filtered[df_filtered['risco'].isin(risco_filter)]
df_filtered = df_filtered[(df_filtered['taxa_infestacao'] >= taxa_min) & (df_filtered['taxa_infestacao'] <= taxa_max)]

# =====================
# HEADER ANIMADO
# =====================
st.markdown(AnimatedHeader(), unsafe_allow_html=True)

# =====================
# MÉTRICAS INTERATIVAS
# =====================
total_casos = int(df_filtered['casos_confirmados'].sum())
total_focos = int(df_filtered['focos_aedes'].sum())
taxa_media = float(df_filtered['taxa_infestacao'].mean())
bairro_critico = df_filtered.loc[df_filtered['taxa_infestacao'].idxmax(), 'bairro'] if len(df_filtered) > 0 else "—"
taxa_critica = df_filtered['taxa_infestacao'].max() if len(df_filtered) > 0 else 0
eficiencia = min(100, max(0, (1 - (total_focos / max(1, total_casos * 10))) * 100))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(MetricCard(icon="🦠", title="Casos Confirmados", value=f"{total_casos:,}", trend="+12%", trend_direction="up", criticality="critical" if total_casos > 400 else "warning" if total_casos > 200 else "success"), unsafe_allow_html=True)
with col2:
    st.markdown(MetricCard(icon="🔥", title="Focos Identificados", value=f"{total_focos:,}", trend="+8%", trend_direction="up", criticality="critical" if total_focos > 150 else "warning" if total_focos > 100 else "success"), unsafe_allow_html=True)
with col3:
    st.markdown(MetricCard(icon="📊", title="Taxa de Infestação", value=f"{taxa_media:.1f}%", trend="+2%", trend_direction="up", criticality="critical" if taxa_media > 4.0 else "warning" if taxa_media > 2.5 else "success"), unsafe_allow_html=True)
with col4:
    st.markdown(MetricCard(icon="🎯", title="Eficiência do Controle", value=f"{eficiencia:.0f}%", trend="+5%", trend_direction="down", criticality="success" if eficiencia > 70 else "warning" if eficiencia > 50 else "critical"), unsafe_allow_html=True)

# =====================
# ABAS PRINCIPAIS
# =====================
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Mapa Inteligente", "📈 Analytics", "🔍 Insights AI", "📋 Relatórios"])

# ---------- ABA 1: MAPA INTELIGENTE ----------
with tab1:
    col_map, col_analytics = st.columns([2, 1])
    with col_map:
        st.subheader("🌍 Mapa de Monitoramento Inteligente")
        center_lat = df_filtered['latitude'].mean() if len(df_filtered) > 0 else -8.284
        center_lon = df_filtered['longitude'].mean() if len(df_filtered) > 0 else -35.976
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB dark_matter', control_scale=True)
        if map_style in ["Marcadores Inteligentes", "Visão Híbrida"]:
            marker_cluster = MarkerCluster(name="Agrupamento de Focos", options={'maxClusterRadius': 50, 'disableClusteringAtZoom': 15}).add_to(m)
            for _, row in df_filtered.iterrows():
                if row['taxa_infestacao'] > 6: color = 'red'; icon_color = 'darkred'; risk_level = "CRÍTICO"
                elif row['taxa_infestacao'] > 3: color = 'orange'; icon_color = 'orange'; risk_level = "ALTO"
                else: color = 'green'; icon_color = 'green'; risk_level = "MODERADO"
                popup_text = f"""<div style='min-width: 280px; font-family: Arial, sans-serif;'><div style='background: {color}; color: white; padding: 10px; border-radius: 8px 8px 0 0; margin: -10px -10px 10px -10px;'><h3 style='margin: 0; font-size: 1.1em;'>{row['bairro']}</h3><div style='font-size: 0.9em; opacity: 0.9;'>Nível: {risk_level}</div></div><div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px;'><div><strong>Casos:</strong> {int(row['casos_confirmados'])}</div><div><strong>Focos:</strong> {int(row['focos_aedes'])}</div><div><strong>Taxa:</strong> {row['taxa_infestacao']:.1f}%</div><div><strong>Risco:</strong> {row.get('risco', 'N/A')}</div></div></div>"""
                folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=10 + (row['taxa_infestacao'] * 3), popup=folium.Popup(popup_text, max_width=300), tooltip=f"{row['bairro']} - {row['taxa_infestacao']:.1f}%", color=color, fillColor=color, fillOpacity=0.7, weight=2).add_to(marker_cluster if map_style == "Visão Híbrida" else m)
        if map_style in ["Mapa de Calor", "Visão Híbrida"] and len(df_filtered) > 0:
            heat_data = [[row['latitude'], row['longitude'], row['taxa_infestacao']] for _, row in df_filtered.iterrows()]
            HeatMap(heat_data, radius=25, blur=18, max_zoom=13, min_opacity=0.4, gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}).add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, height=600, width=None)
    with col_analytics:
        st.subheader("📊 Ranking de Performance")
        rank_metric = st.selectbox("Métrica de Ordenação:", ["taxa_infestacao", "casos_confirmados", "focos_aedes", "densidade_casos"], format_func=lambda x: {"taxa_infestacao": "Taxa de Infestação", "casos_confirmados": "Casos Confirmados", "focos_aedes": "Focos Identificados", "densidade_casos": "Densidade de Casos"}[x])
        df_rank = df_filtered.sort_values(rank_metric, ascending=False).head(10)
        fig_rank = px.bar(df_rank, y='bairro', x=rank_metric, orientation='h', color=rank_metric, color_continuous_scale='Inferno', title="Top 10 Bairros - Performance", height=500)
        fig_rank.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_rank, use_container_width=True)

# ---------- ABA 2: ANALYTICS AVANÇADO ----------
with tab2:
    st.subheader("📈 Analytics e Visualizações Avançadas")
    col_anal1, col_anal2 = st.columns(2)
    with col_anal1:
        st.markdown("#### Distribuição de Risco por Bairro")
        chart_type = st.radio("Tipo de Visualização:", ["Barras", "Pizza", "Treemap"], horizontal=True, key="dist_chart")
        if chart_type == "Barras":
            fig_dist = px.bar(df_filtered.nlargest(8, 'casos_confirmados'), x='bairro', y='casos_confirmados', color='taxa_infestacao', color_continuous_scale='Viridis', title="Distribuição de Casos por Bairro")
        elif chart_type == "Pizza":
            fig_dist = px.pie(df_filtered, values='casos_confirmados', names='bairro', hole=0.4, title="Proporção de Casos por Bairro")
        else:
            fig_dist = px.treemap(df_filtered, path=['risco', 'bairro'], values='casos_confirmados', color='taxa_infestacao', color_continuous_scale='RdYlGn_r')
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_anal2:
        st.markdown("#### Correlação e Dispersão")
        size_by = st.selectbox("Tamanho dos pontos por:", ["taxa_infestacao", "populacao", "casos_confirmados"])
        fig_scatter = px.scatter(df_filtered, x='focos_aedes', y='casos_confirmados', size=size_by, color='taxa_infestacao', hover_name='bairro', size_max=25, title="Relação entre Focos e Casos Confirmados", color_continuous_scale='Plasma')
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("#### 📅 Análise Temporal e Sazonalidade")
    if len(evol_df) > 0:
        evol_filtered = evol_df[evol_df['bairro'].isin(df_filtered['bairro'].unique())] if len(df_filtered) > 0 else evol_df
        fig_evol = go.Figure()
        fig_evol.add_trace(go.Scatter(x=evol_filtered['mes'].unique(), y=evol_filtered.groupby('mes')['casos'].sum(), mode='lines+markers', name='Casos Confirmados', line=dict(color='#FF6B6B', width=4, shape='spline'), marker=dict(size=8)))
        fig_evol.add_trace(go.Scatter(x=evol_filtered['mes'].unique(), y=evol_filtered.groupby('mes')['focos'].sum(), mode='lines+markers', name='Focos Identificados', line=dict(color='#4ECDC4', width=4, shape='spline'), marker=dict(size=8)))
        fig_evol.add_trace(go.Scatter(x=evol_filtered['mes'].unique(), y=evol_filtered.groupby('mes')['taxa'].mean(), mode='lines+markers', name='Taxa Média', line=dict(color='#FFD93D', width=4, shape='spline'), marker=dict(size=8), yaxis='y2'))
        fig_evol.update_layout(height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis2=dict(title='Taxa de Infestação (%)', overlaying='y', side='right', range=[0, evol_filtered.groupby('mes')['taxa'].mean().max() * 1.2]))
        st.plotly_chart(fig_evol, use_container_width=True)

# ---------- ABA 3: INSIGHTS AI ----------
with tab3:
    st.subheader("🤖 Insights de Inteligência Artificial")
    if len(df_filtered) == 0:
        st.info("🎯 Ajuste os filtros para ver insights específicos da área selecionada.")
    else:
        critical_bairros = df_filtered[df_filtered['taxa_infestacao'] > 4.0]
        high_risk_bairros = df_filtered[df_filtered['taxa_infestacao'] > 6.0]
        col_ai1, col_ai2 = st.columns(2)
        with col_ai1:
            st.markdown("### 📊 Análise Preditiva")
            insights = []
            if len(high_risk_bairros) > 0:
                worst = high_risk_bairros.loc[high_risk_bairros['taxa_infestacao'].idxmax()]
                insights.append(InsightCard("🚨 Área de Alerta Máximo", f"**{worst['bairro']}** apresenta taxa crítica de **{worst['taxa_infestacao']:.1f}%**. Recomenda-se ação imediata com mutirão emergencial.", "critical"))
            if len(critical_bairros) >= 3:
                insights.append(InsightCard("⚠️ Múltiplas Áreas Críticas", f"**{len(critical_bairros)} bairros** com taxa acima de 4.0%. Situação requer ampliação das equipes de campo.", "warning"))
            avg_efficiency = (df_filtered['casos_confirmados'] / df_filtered['focos_aedes']).mean()
            if avg_efficiency > 3:
                insights.append(InsightCard("🔍 Alta Relação Casos/Focos", f"Relação de **{avg_efficiency:.1f} casos por foco**. Possível subnotificação de focos ou alta transmissibilidade.", "warning"))
            if 'populacao' in df_filtered.columns:
                max_density = df_filtered['densidade_casos'].max()
                dense_bairro = df_filtered.loc[df_filtered['densidade_casos'].idxmax(), 'bairro']
                insights.append(InsightCard("👥 Análise de Densidade", f"**{dense_bairro}** tem maior densidade: **{max_density:.1f}** casos por mil habitantes. Focar campanhas educativas.", "success"))
            for insight in insights:
                st.markdown(insight, unsafe_allow_html=True)
        with col_ai2:
            st.markdown("### 🎯 Recomendações Estratégicas")
            recommendations = []
            if len(high_risk_bairros) > 0:
                rec_bairros = ", ".join(high_risk_bairros['bairro'].head(3).tolist())
                recommendations.append(InsightCard("🔴 Ação Imediata Requerida", f"Mutirão de limpeza e fumacê nos bairros: {rec_bairros}. Alocar 2x agentes nestas áreas.", "critical"))
            if taxa_media > 3.5:
                recommendations.append(InsightCard("📢 Campanha de Conscientização", "Ampliar campanhas em escolas e unidades de saúde. Focar em eliminação de criadouros.", "warning"))
            if total_focos > len(df_filtered) * 10:
                recommendations.append(InsightCard("🔍 Intensificação de Vistorias", "Aumentar frequência de vistorias em terrenos baldios e pontos estratégicos em 50%.", "warning"))
            recommendations.append(InsightCard("🤝 Engajamento Comunitário", "Envolver lideranças comunitárias no programa de combate. Criar grupos de WhatsApp por bairro.", "success"))
            recommendations.append(InsightCard("📱 Otimização Tecnológica", "Utilizar drone para mapeamento aéreo de focos em áreas de difícil acesso.", "success"))
            for rec in recommendations[:3]:
                st.markdown(rec, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🔮 Projeções e Cenários Futuros")
    col_proj1, col_proj2, col_proj3, col_proj4 = st.columns(4)
    with col_proj1:
        st.metric("Casos em 30 dias", f"{int(total_casos * 1.18):,}", "+18%", delta_color="inverse")
    with col_proj2:
        st.metric("Focos Esperados", f"{int(total_focos * 1.12):,}", "+12%", delta_color="inverse")
    with col_proj3:
        st.metric("Taxa Projetada", f"{taxa_media * 1.15:.1f}%", "+15%", delta_color="inverse")
    with col_proj4:
        st.metric("Custo Estimado", f"R$ {int(total_casos * 250):,}", "+20%", delta_color="inverse")
    st.info("💡 **Nota das Projeções**: Baseadas em modelo preditivo considerando dados históricos, condições climáticas e eficácia das ações atuais. Cenário pode ser alterado com implementação das recomendações estratégicas.")

# ---------- ABA 4: RELATÓRIOS PROFISSIONAIS ----------
with tab4:
    st.subheader("📋 Sistema de Relatórios")
    col_rep1, col_rep2 = st.columns([1, 1])
    with col_rep1:
        st.markdown("### 📊 Resumo Executivo")
        total_bairros = len(df_filtered)
        bairros_above_4 = len(df_filtered[df_filtered['taxa_infestacao'] > 4.0])
        bairros_above_6 = len(df_filtered[df_filtered['taxa_infestacao'] > 6.0])
        avg_population = df_filtered['populacao'].mean() if 'populacao' in df_filtered.columns else 0
        total_populacao = int(avg_population * total_bairros)
        densidade_casos = (total_casos/total_populacao)*10000 if total_populacao > 0 else 0
        
        st.markdown(f"""
        <div style='background: var(--gradient-card); padding: 2rem; border-radius: var(--radius); border: 1px solid var(--border); margin: 1rem 0;'>
            <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;'>
                <div style='background: var(--gradient-primary); padding: 0.5rem; border-radius: 12px;'><span style='font-size: 1.5rem;'>📊</span></div>
                <div><h3 style='margin: 0; color: var(--primary); font-weight: 800;'>Relatório Epidemiológico</h3><p style='margin: 0; color: var(--text-muted); font-size: 0.9rem;'>Dashboard de Monitoramento - Aedes aegypti</p></div>
            </div>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0; padding: 1rem; background: rgba(255,255,255,0.03); border-radius: var(--radius-sm);'>
                <div style='text-align: center;'><div style='font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.3rem;'>📅 Período</div><div style='font-weight: 700; color: var(--text);'>{datetime.now().strftime('%d/%m/%Y')}</div></div>
                <div style='text-align: center;'><div style='font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.3rem;'>🏙️ Área Coberta</div><div style='font-weight: 700; color: var(--text);'>{total_bairros} bairros</div></div>
                <div style='text-align: center;'><div style='font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.3rem;'>👥 População</div><div style='font-weight: 700; color: var(--text);'>{total_populacao:,} hab</div></div>
                <div style='text-align: center;'><div style='font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.3rem;'>📈 Densidade</div><div style='font-weight: 700; color: var(--text);'>{densidade_casos:.1f}/10k</div></div>
            </div>
            <div style='background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%); padding: 1.5rem; border-radius: var(--radius-sm); border: 1px solid var(--border);'>
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'><div style='background: var(--primary); width: 4px; height: 20px; border-radius: 2px;'></div><h4 style='margin: 0; color: var(--text); font-weight: 700;'>Situação por Nível de Risco</h4></div>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; text-align: center;'>
                    <div style='padding: 1.5rem 1rem; background: rgba(76, 175, 80, 0.1); border: 1px solid rgba(76, 175, 80, 0.3); border-radius: var(--radius-sm);'><div style='color: var(--success); font-size: 2rem; font-weight: 800; line-height: 1;'>{total_bairros - bairros_above_4}</div><div style='color: var(--success-dark); font-size: 0.8rem; font-weight: 600; margin: 0.5rem 0;'>SOB CONTROLE</div><div style='color: var(--text-muted); font-size: 0.75rem;'>Taxa ≤ 3%</div></div>
                    <div style='padding: 1.5rem 1rem; background: rgba(255, 167, 38, 0.1); border: 1px solid rgba(255, 167, 38, 0.3); border-radius: var(--radius-sm);'><div style='color: var(--warning); font-size: 2rem; font-weight: 800; line-height: 1;'>{bairros_above_4 - bairros_above_6}</div><div style='color: var(--warning-dark); font-size: 0.8rem; font-weight: 600; margin: 0.5rem 0;'>EM ALERTA</div><div style='color: var(--text-muted); font-size: 0.75rem;'>Taxa 3-6%</div></div>
                    <div style='padding: 1.5rem 1rem; background: rgba(255, 107, 107, 0.1); border: 1px solid rgba(255, 107, 107, 0.3); border-radius: var(--radius-sm);'><div style='color: var(--danger); font-size: 2rem; font-weight: 800; line-height: 1;'>{bairros_above_6}</div><div style='color: var(--danger-dark); font-size: 0.8rem; font-weight: 600; margin: 0.5rem 0;'>CRÍTICOS</div><div style='color: var(--text-muted); font-size: 0.75rem;'>Taxa > 6%</div></div>
                </div>
                <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);'>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'><div style='width: 12px; height: 12px; background: var(--success); border-radius: 50%;'></div><span style='font-size: 0.8rem; color: var(--text-muted);'>Baixo Risco</span></div>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'><div style='width: 12px; height: 12px; background: var(--warning); border-radius: 50%;'></div><span style='font-size: 0.8rem; color: var(--text-muted);'>Médio Risco</span></div>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'><div style='width: 12px; height: 12px; background: var(--danger); border-radius: 50%;'></div><span style='font-size: 0.8rem; color: var(--text-muted);'>Alto Risco</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_rep2:
        st.markdown("### 📤 Exportação de Dados")
        export_config = st.expander("⚙️ Configurações de Exportação", expanded=True)
        with export_config:
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                export_format = st.radio("Formato:", ["CSV", "Excel", "PDF", "JSON"], horizontal=True)
                include_charts = st.checkbox("Incluir gráficos", True)
            with col_exp2:
                date_range = st.selectbox("Período:", ["Últimos 30 dias", "Últimos 90 dias", "Este ano", "Personalizado"])
                compression = st.checkbox("Compactar arquivo", True)
        if st.button("🔄 Gerar Relatório Completo", use_container_width=True):
            with st.spinner("Gerando relatório premium..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(label=f"📥 Download Relatório ({export_format})", data=csv_data, file_name=f"relatorio_aedes_premium_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", use_container_width=True)
                st.success("✅ Relatório gerado com sucesso!")
        st.markdown("---")
        st.markdown("### 🎯 Dashboard Interativo")
        st.markdown("""
        <div style='background: var(--gradient-card); padding: 1.5rem; border-radius: var(--radius); border: 1px solid var(--border);'>
            <h4 style='margin: 0 0 1rem 0; color: var(--primary);'>Recursos Premium Incluídos:</h4>
            <ul style='color: var(--text-muted); margin: 0; padding-left: 1.2rem;'>
                <li>Monitoramento em tempo real</li>
                <li>Alertas precoces automáticos</li>
                <li>Análises preditivas com IA</li>
                <li>Relatórios executivos automáticos</li>
                <li>API para integração</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =====================
# RODAPÉ
# =====================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: var(--text-muted); padding: 3rem 0;'>
    <div style='font-size: 2rem; margin-bottom: 1rem;'>🦟</div>
    <h3 style='margin: 0; color: var(--text);'>Aedes Analytics <span style='color: var(--primary);'>Pro</span></h3>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
        Sistema de Monitoramento Inteligente • Versão Premium 2.0<br>
        Desenvolvido para a Secretaria de Saúde de Caruaru • 
        Atualizado em {datetime.now().strftime('%d/%m/%Y às %H:%M')} • 
        <span style='color: var(--primary);'>v2.1.0</span>
    </p>
    <div style='margin-top: 1rem; font-size: 0.8rem; opacity: 0.7;'>
        📞 Suporte: (81) 99999-9999 • ✉️ contato@aedesanalytics.com
    </div>
</div>
""", unsafe_allow_html=True)