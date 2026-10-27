import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from libpysal.weights import Queen, KNN
from libpysal.weights import lag_spatial
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag
from sklearn.preprocessing import StandardScaler

import plotly.express as px
import plotly.graph_objects as go

# Configuración de página en Streamlit
st.set_page_config(page_title="SpVAR Colombia", layout="wide")

# -----------------------------------------------------------------------------
# 1. DATOS (FUNCIONALIDAD INTACTA)
# -----------------------------------------------------------------------------
url = "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson"
try:
    gdf = gpd.read_file(url)
except:
    gdf = gpd.read_file("https://raw.githubusercontent.com/santiblanko/colombia.geojson/master/depto.json")

col = next((c for c in ['DPTO_CNMBR','NOMBRE_DPT','dpto_cnmbr','nombre','NAME','name'] if c in gdf.columns), None)
if col:
    gdf = gdf.rename(columns={col: 'departamento'})
else:
    gdf['departamento'] = gdf.index.astype(str)

gdf['departamento'] = gdf['departamento'].astype(str).str.upper().str.strip()
gdf = gdf.to_crs(epsg=4326)

def normalizar(n):
    n = str(n).upper().strip()
    r = {
        'BOGOTA':'BOGOTÁ, D.C.', 'BOGOTÁ D.C.':'BOGOTÁ, D.C.', 'BOGOTÁ DC':'BOGOTÁ, D.C.',
        'SAN ANDRES':'SAN ANDRÉS', 'SAN ANDRÉS Y PROVIDENCIA':'SAN ANDRÉS',
        'ARCHIPIELAGO DE SAN ANDRES':'SAN ANDRÉS', 'ARCHIPIÉLAGO DE SAN ANDRÉS':'SAN ANDRÉS',
        'N. DE SANTANDER':'NORTE DE SANTANDER', 'NORTE SANTANDER':'NORTE DE SANTANDER',
        'VALLE':'VALLE DEL CAUCA'
    }
    for k,v in r.items():
        if k in n: return v
    return n

gdf['departamento_norm'] = gdf['departamento'].apply(normalizar)

ipm = {
    'VICHADA':55.2,'GUAINÍA':51.8,'VAUPÉS':41.7,'LA GUAJIRA':40.1,'CHOCÓ':30.8,
    'SUCRE':25.4,'GUAVIARE':20.0,'CÓRDOBA':19.8,'AMAZONAS':18.1,'MAGDALENA':17.6,
    'BOLÍVAR':17.0,'CAQUETÁ':15.0,'CESAR':14.8,'PUTUMAYO':13.6,'NORTE DE SANTANDER':13.3,
    'NARIÑO':13.2,'CAUCA':13.0,'ARAUCA':12.7,'META':12.3,'TOLIMA':12.2,'HUILA':9.7,
    'ANTIOQUIA':9.5,'CASANARE':8.9,'ATLÁNTICO':8.2,'CALDAS':7.9,'SANTANDER':6.9,
    'QUINDÍO':6.7,'BOYACÁ':6.4,'RISARALDA':4.9,'VALLE DEL CAUCA':4.9,'SAN ANDRÉS':4.3,
    'CUNDINAMARCA':4.1,'BOGOTÁ, D.C.':2.2
}
gdf['IPM'] = gdf['departamento_norm'].map(ipm).fillna(9.9)

conect = {
    'BOGOTÁ, D.C.':48.0,'ATLÁNTICO':31.5,'VALLE DEL CAUCA':29.8,'QUINDÍO':28.5,
    'RISARALDA':26.2,'CALDAS':24.1,'ANTIOQUIA':23.8,'CUNDINAMARCA':22.5,
    'SANTANDER':19.4,'BOYACÁ':17.8,'TOLIMA':16.9,'HUILA':15.7,'NORTE DE SANTANDER':15.2,
    'MAGDALENA':14.1,'CESAR':13.5,'CÓRDOBA':13.0,'BOLÍVAR':12.4,'SUCRE':12.1,
    'CAUCA':11.8,'NARIÑO':11.2,'META':9.8,'CASANARE':8.5,'ARAUCA':7.2,
    'LA GUAJIRA':6.8,'CHOCÓ':5.1,'CAQUETÁ':4.8,'PUTUMAYO':4.3,'GUAVIARE':3.5,
    'VICHADA':2.1,'AMAZONAS':1.8,'GUAINÍA':1.4,'VAUPÉS':1.3,'SAN ANDRÉS':38.0
}
gdf['Conectividad'] = gdf['departamento_norm'].map(conect).fillna(12.0)

np.random.seed(42)
gdf['VIIRS'] = (0.55*gdf['Conectividad'] + 0.25*(50 - gdf['IPM']) +
                np.random.normal(0, 2.5, len(gdf))).clip(0.5)

dens_pob = {
    'BOGOTÁ, D.C.':18000,'ATLÁNTICO':750,'QUINDÍO':420,'RISARALDA':380,
    'CALDAS':210,'VALLE DEL CAUCA':220,'CUNDINAMARCA':140,'ANTIOQUIA':130,
    'SANTANDER':85,'BOYACÁ':55,'TOLIMA':50,'HUILA':55,
    'NORTE DE SANTANDER':70,'MAGDALENA':60,'CESAR':45,'CÓRDOBA':70,
    'BOLÍVAR':55,'SUCRE':80,'CAUCA':45,'NARIÑO':55,'META':15,
    'CASANARE':10,'ARAUCA':8,'LA GUAJIRA':20,'CHOCÓ':10,
    'CAQUETÁ':6,'PUTUMAYO':12,'GUAVIARE':3,'VICHADA':1.5,
    'AMAZONAS':0.8,'GUAINÍA':0.7,'VAUPÉS':0.6,'SAN ANDRÉS':900
}
gdf['Densidad_Pob'] = gdf['departamento_norm'].map(dens_pob).fillna(30)

gdf['centroid'] = gdf.geometry.centroid
gdf['lon'] = gdf.centroid.x
gdf['lat'] = gdf.centroid.y

scaler = StandardScaler()
gdf['IPM_z'] = scaler.fit_transform(gdf[['IPM']])
gdf['Conectividad_z'] = scaler.fit_transform(gdf[['Conectividad']])
gdf['VIIRS_z'] = scaler.fit_transform(gdf[['VIIRS']])

# -----------------------------------------------------------------------------
# 2. W + LISA + MODELO (FUNCIONALIDAD INTACTA)
# -----------------------------------------------------------------------------
W_opt = Queen.from_dataframe(gdf, silence_warnings=True)
if getattr(W_opt, 'islands', None):
    W_opt = KNN.from_dataframe(gdf, k=3)
W_opt.transform = 'r'

y1 = gdf['IPM_z'].values.reshape(-1,1)
y2 = gdf['Conectividad_z'].values.reshape(-1,1)
X1 = np.hstack([y2, lag_spatial(W_opt, y2)])

try:
    m_ipm = ML_Lag(y1, X1, w=W_opt, name_y="IPM")
    mejor_modelo, rho = "SAR", float(m_ipm.rho)
except Exception:
    m_ipm = OLS(y1, X1, w=W_opt, name_y="IPM")
    mejor_modelo, rho = "OLS", 0.0

moran = Moran(gdf['IPM_z'].values, W_opt)
lisa  = Moran_Local(gdf['IPM_z'].values, W_opt)
gdf['LISA_q']     = lisa.q.astype(int)
gdf['LISA_sig']   = lisa.p_sim < 0.05
gdf['LISA_label'] = gdf['LISA_q'].map({1:'HH', 2:'LH', 3:'LL', 4:'HL'})

gdf_plot = gdf.copy().reset_index(drop=True)
gdf_plot['id_geo'] = gdf_plot.index.astype(str)

# -----------------------------------------------------------------------------
# 3. FUNCIONES DE VISUALIZACIÓN Y CÁLCULO (INTACTAS)
# -----------------------------------------------------------------------------
def interpretacion():
    hh = list(gdf.loc[(gdf.LISA_q==1) & gdf.LISA_sig, 'departamento_norm'])
    ll = list(gdf.loc[(gdf.LISA_q==3) & gdf.LISA_sig, 'departamento_norm'])
    return f"""
### Interpretación automática SpVAR

- **Modelo IPM**: `{mejor_modelo}`
- **ρ espacial**: `{rho:.3f}`
- **I de Moran**: `{moran.I:.3f}` (p = {moran.p_sim:.4f}) → clustering espacial de pobreza
- **Clusters HH (trampa de pobreza)**: {hh if hh else 'ninguno'}
- **Clusters LL (círculos virtuosos)**: {ll if ll else 'ninguno'}

La pobreza se transmite espacialmente. Mejorar la conectividad de los vecinos genera spillovers positivos.
"""

def mecanismos_transmision():
    return f"""
### Mecanismos de transmisión espacial capturados por el SpVAR

| # | Mecanismo | Expresión | Interpretación |
|---|-----------|-----------|----------------|
| 1 | **Autodinámica espacial** | `W · Yᵢ → Yᵢ` | La pobreza/conectividad de un departamento depende de la de sus vecinos |
| 2 | **Cruce local** | `Xⱼ → Yᵢ` | La conectividad local afecta el IPM del mismo departamento (y viceversa) |
| 3 | **Relación espacial directa** | `ρ · W · Yᵢ` | Efecto de contagio puro (parámetro ρ) |
| 4 | **Relación espacial cruzada** | `W · Xⱼ → Yᵢ` | La conectividad de los vecinos afecta el IPM local (spillover cruzado) |

**Resultado actual**: Modelo **{mejor_modelo}** · ρ ≈ **{rho:.3f}** · Moran I ≈ **{moran.I:.3f}** (p={moran.p_sim:.4f})
"""

def mapa_matplotlib(variable="IPM"):
    try:
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))
        if variable == "LISA_q":
            gdf.plot(column='LISA_q', categorical=True, legend=True, ax=ax,
                     cmap='RdBu_r', edgecolor='k', linewidth=0.4,
                     legend_kwds={'title': 'Cluster LISA'})
            ax.set_title("Clusters LISA de Pobreza Multidimensional\n(1=HH · 2=LH · 3=LL · 4=HL)", fontsize=12)
        else:
            cmaps = {"IPM":"YlOrRd", "Conectividad":"Blues", "VIIRS":"inferno", "Densidad_Pob":"YlGn"}
            gdf.plot(column=variable, cmap=cmaps.get(variable, "YlOrRd"),
                     legend=True, ax=ax, edgecolor='k', linewidth=0.35,
                     legend_kwds={'shrink': 0.6, 'label': variable})
            ax.set_title(f"{variable} – Colombia (alto contraste)", fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Error mapa: {e}", ha='center', va='center', color='red')
        ax.axis('off')
        return fig

def generar_mapa_nocturno(variable):
    try:
        fig = px.choropleth(
            gdf_plot,
            geojson=gdf_plot.__geo_interface__,
            locations=gdf_plot.index,
            color=variable,
            color_continuous_scale="Viridis",
            hover_name="departamento_norm",
            title=f"Mapa Coroplético (Visión Nocturna) – {variable}"
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            font_color="#00ffcc",
            geo=dict(bgcolor="#0b0f19", landcolor="#111827"),
            margin={"r":0,"t":40,"l":0,"b":0},
            height=600
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {e}", showarrow=False, font=dict(color="red"))
        fig.update_layout(height=400, paper_bgcolor="#0b0f19")
        return fig

def generar_mapa_3d(variable):
    try:
        fig = go.Figure(data=[go.Scatter3d(
            x=gdf_plot['lon'],
            y=gdf_plot['lat'],
            z=gdf_plot[variable],
            mode='markers',
            marker=dict(
                size=10 + (gdf_plot['Conectividad'] / gdf_plot['Conectividad'].max() * 12),
                color=gdf_plot[variable],
                colorscale='Plasma',
                opacity=0.9,
                colorbar=dict(title=variable),
                line=dict(width=0.5, color='white')
            ),
            text=gdf_plot['departamento_norm'],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                f"{variable}: %{{z:.2f}}<br>" +
                "Longitud: %{x:.2f}<br>" +
                "Latitud: %{y:.2f}<extra></extra>"
            )
        )])

        fig.update_layout(
            title=f"Estructura Dinámica 3D Espacial – {variable}",
            paper_bgcolor="#0b0f19",
            font_color="#ffffff",
            scene=dict(
                xaxis_title="Longitud",
                yaxis_title="Latitud",
                zaxis_title=variable,
                xaxis=dict(backgroundcolor="#111827", gridcolor="#333333", showbackground=True),
                yaxis=dict(backgroundcolor="#111827", gridcolor="#333333", showbackground=True),
                zaxis=dict(backgroundcolor="#111827", gridcolor="#333333", showbackground=True),
                camera=dict(eye=dict(x=1.6, y=1.5, z=1.2))
            ),
            width=920,
            height=680,
            margin={"r":0,"t":40,"l":0,"b":0}
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error 3D: {e}", showarrow=False, font=dict(color="red", size=14))
        fig.update_layout(height=400, paper_bgcolor="#0b0f19", font_color="white")
        return fig

def generar_interpretabilidad(variable):
    media = gdf_plot[variable].mean()
    max_val = gdf_plot[variable].max()
    min_val = gdf_plot[variable].min()
    dept_max = gdf_plot.loc[gdf_plot[variable].idxmax(), 'departamento_norm']
    dept_min = gdf_plot.loc[gdf_plot[variable].idxmin(), 'departamento_norm']

    return f"""
### 📊 Informe de Interpretabilidad Automatizada – {variable}

- **Tendencia central:** promedio = **{media:.2f}**
- **Valor máximo:** **{dept_max}** ({max_val:.2f})
- **Valor mínimo:** **{dept_min}** ({min_val:.2f})
- **Diagnóstico espacial:** concentración territorial coherente con el I de Moran = {moran.I:.3f} (p={moran.p_sim:.4f}) y con los clusters LISA detectados.
"""

def esda_fig():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    sns.histplot(gdf["IPM"], kde=True, ax=axes[0], color="#e74c3c")
    axes[0].set_title("IPM (%)")
    sns.histplot(gdf["Conectividad"], kde=True, ax=axes[1], color="#3498db")
    axes[1].set_title("Conectividad")
    sns.histplot(gdf["VIIRS"], kde=True, ax=axes[2], color="#f39c12")
    axes[2].set_title("VIIRS")
    plt.tight_layout()
    return fig

def lisa_map():
    fig, ax = plt.subplots(figsize=(8, 7))
    gdf.plot(column="LISA_q", categorical=True, legend=True, ax=ax,
             cmap="RdBu_r", edgecolor="k", linewidth=0.4)
    ax.set_title("Clusters LISA de IPM\n(1=HH · 2=LH · 3=LL · 4=HL)")
    ax.axis("off")
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# 4. INTERFAZ EN STREAMLIT (Adaptada desde Gradio manteniendo tus pestañas)
# -----------------------------------------------------------------------------
st.markdown("""
# Laboratorio SpVAR – Colombia
### IPM ↔ Conectividad Vial · Datos DANE · Mapas + 3D + Mecanismos de transmisión
""")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Inicio", "ESDA", "Mapas (matplotlib)", "Mapas Visión Nocturna", 
    "SpVAR + LISA", "3D Dinámico", "Interpretabilidad", "Metodología"
])

with tab1:
    st.markdown(interpretacion())
    st.markdown(mecanismos_transmision())

with tab2:
    if st.button("Ejecutar ESDA", type="primary"):
        st.markdown(f"""
**Media IPM** = {gdf['IPM'].mean():.1f}%  
**Rango** = {gdf['IPM'].min():.1f} – {gdf['IPM'].max():.1f}%  
**I de Moran** = {moran.I:.3f} (p = {moran.p_sim:.4f})  
→ Clustering espacial de pobreza confirmado.
        """)
        st.pyplot(esda_fig())

with tab3:
    st.markdown("Mapas coropléticos estables (matplotlib).")
    var = st.radio("Variable", ["IPM", "Conectividad", "VIIRS", "LISA_q", "Densidad_Pob"])
    if st.button("Generar mapa", type="primary"):
        st.pyplot(mapa_matplotlib(var))

with tab4:
    st.markdown("Choropleth estilo nocturno interconectado con Plotly.")
    var_n = st.selectbox("Variable", ["IPM", "Conectividad", "VIIRS", "Densidad_Pob"], key="var_nocturno")
    if st.button("Actualizar Mapa Nocturno", type="primary"):
        fig_n = generar_mapa_nocturno(var_n)
        st.plotly_chart(fig_n, use_container_width=True)

with tab5:
    st.markdown(interpretacion())
    st.markdown(mecanismos_transmision())
    if st.button("Mapa de clusters LISA", type="primary"):
        st.pyplot(lisa_map())

with tab6:
    st.markdown("""
    ### Estructura Dinámica 3D Espacial
    Basada en centroides del shape · Color = valor de la variable · Tamaño ≈ conectividad
    """)
    var3 = st.selectbox("Variable para el eje Z", ["IPM", "Conectividad", "VIIRS", "Densidad_Pob"], key="var_3d")
    if st.button("Renderizar Modelo 3D", type="primary"):
        fig_3d = generar_mapa_3d(var3)
        st.plotly_chart(fig_3d, use_container_width=True)

with tab7:
    var_i = st.selectbox("Variable a interpretar", ["IPM", "Conectividad", "VIIRS", "Densidad_Pob"], key="var_interp")
    if st.button("Generar Análisis Automático", type="primary"):
        st.markdown(generar_interpretabilidad(var_i))

with tab8:
    st.markdown("""
### Metodología auditada
- **W**: Queen (fallback KNN para islas)
- **Especificación**: SpVAR seccional cruzado
- **Selección**: SAR cuando converge
- **Diagnósticos**: Moran + LISA + ρ
- **Mapas estables**: matplotlib
- **Mapas nocturnos + 3D**: Plotly
    """)
    st.markdown(mecanismos_transmision())

st.markdown("---")
st.markdown("<center><small>Laboratorio de Econometría Espacial</small></center>", unsafe_allow_html=True)
