import os
import io
import json
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from protocolo_fadiga_emg import EMGFatigueAnalyzer, generate_simulated_emg_fatigue
from gerar_pdf_fadiga import generate_pdf_report

# Configuração da Página Streamlit
st.set_page_config(
    page_title="Análise de Fadiga sEMG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização Customizada
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F7FAFC;
        border-left: 5px solid #3182CE;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚡ Portal de Análise de Fadiga Eletromiográfica (sEMG)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Plataforma dedicada para processamento de sinais de sEMG de 30 segundos (2000 Hz) a 50% da CIVM</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# BARRA LATERAL: CONFIGURAÇÕES E UPLOAD DE ARQUIVO
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configurações da Análise")

# 1. Upload do Arquivo
uploaded_file = st.sidebar.file_drop_target if hasattr(st.sidebar, 'file_drop_target') else None
uploaded_file = st.sidebar.file_uploader(
    "Carregar Arquivo da Coleta (.txt, .csv, .tsv)",
    type=["txt", "csv", "tsv"],
    help="Selecione o arquivo bruto com as colunas de sEMG adquiridas a 2000 Hz."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Parâmetros do Sinal")

fs = st.sidebar.number_input("Frequência de Amostragem (Hz)", value=2000.0, step=100.0)
notch_freq = st.sidebar.selectbox("Frequência do Filtro Notch (Hz)", options=[60.0, 50.0], index=0)
window_size = st.sidebar.slider("Tamanho da Janela Móvel (s)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
overlap_pct = st.sidebar.slider("Sobreposição / Overlap (%)", min_value=0.0, max_value=75.0, value=50.0, step=5.0)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Dica de Formato:** O leitor detecta e remove automaticamente linhas de metadados no topo do seu arquivo .txt e aceita qualquer separador (tabulação, espaço ou vírgula).")

# -----------------------------------------------------------------------------
# FUNÇÕES DE LEITURA BLINDADA DE DADOS
# -----------------------------------------------------------------------------
@st.cache_data
def load_emg_data(file_bytes, filename):
    """
    Lê o arquivo de forma blindada, detectando linhas de metadados e delimitadores.
    """
    content = file_bytes.decode('latin1', errors='replace').splitlines()
    
    # Encontra a primeira linha com conteúdo numérico
    skip_rows = 0
    for i, line in enumerate(content[:50]):
        parts = line.strip().split()
        if len(parts) > 0:
            # Verifica se a maioria das partes é numérica
            num_count = sum(1 for p in parts if p.replace('.', '', 1).replace('-', '', 1).isdigit())
            if num_count >= max(1, len(parts) * 0.5):
                skip_rows = i
                break
                
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        skiprows=skip_rows,
        sep=r'\s+|,|;',
        engine='python',
        header=None
    )
    
    # Remove colunas totalmente vazias ou não numéricas
    df = df.dropna(how='all', axis=1)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Se a primeira coluna parecer ser o tempo (0, 0.0005, 0.001...), ajusta
    first_col = df.iloc[:, 0].values
    if len(first_col) > 1 and np.all(np.diff(first_col) > 0) and first_col[-1] < 100.0:
        df = df.iloc[:, 1:] # Remove coluna de tempo e recria
        
    # Nomeia os canais
    num_channels = df.shape[1]
    default_names = [
        'peitoral_maior_direito', 'peitoral_maior_esquerdo',
        'triceps_braquial_direito', 'triceps_braquial_esquerdo',
        'deltoide_direito', 'deltoide_esquerdo'
    ]
    
    if num_channels <= len(default_names):
        col_names = default_names[:num_channels]
    else:
        col_names = [f"canal_emg_{i+1}" for i in range(num_channels)]
        
    df.columns = col_names
    df['tempo'] = np.arange(len(df)) / fs
    
    return df

# -----------------------------------------------------------------------------
# PROCESSAMENTO DE ARQUIVO OU MODO DEMONSTRAÇÃO
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        df_emg = load_emg_data(uploaded_file.getvalue(), uploaded_file.name)
        st.success(f"✅ Arquivo `{uploaded_file.name}` carregado com sucesso! ({len(df_emg)} amostras | {len(df_emg)/fs:.1f} segundos)")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}. Verifique o formato do arquivo.")
        st.stop()
else:
    st.warning("⚠️ Nenhum arquivo carregado. Exibindo demonstração com dados simulados de 30s @ 2000 Hz.")
    df_emg = generate_simulated_emg_fatigue(duration=30.0, fs=fs)

# Mostra prévia dos dados
with st.expander("📊 Visualizar Prévia dos Dados Brutos"):
    st.dataframe(df_emg.head(10), use_container_width=True)

# -----------------------------------------------------------------------------
# PROCESSAMENTO DA ANÁLISE DE FADIGA
# -----------------------------------------------------------------------------
analyzer = EMGFatigueAnalyzer(fs=fs, notch_freq=notch_freq)
results, time_series = analyzer.analyze_fatigue(
    df_emg, 
    window_size_sec=window_size, 
    overlap_percent=overlap_pct
)

# -----------------------------------------------------------------------------
# PAINEL DE CARDS E KPIs PRINCIPAIS
# -----------------------------------------------------------------------------
st.markdown("### 📈 Diagnóstico Geral de Fadiga")

col1, col2, col3, col4 = st.columns(4)

# Músculo com maior queda em MDF
max_fatigue_ch = min(results.keys(), key=lambda k: results[k]['MDF_Slope_Hz_s'])
max_fatigue_slope = results[max_fatigue_ch]['MDF_Slope_Hz_s']
max_fatigue_pct = results[max_fatigue_ch]['MDF_Change_Percent']

# Músculo com maior aumento no RMS
max_rec_ch = max(results.keys(), key=lambda k: results[k]['RMS_Slope_uV_s'])
max_rec_slope = results[max_rec_ch]['RMS_Slope_uV_s']
max_rec_pct = results[max_rec_ch]['RMS_Change_Percent']

col1.metric(
    label="Maior Fadiga (MDF Slope)",
    value=f"{max_fatigue_ch.replace('_', ' ').title()}",
    delta=f"{max_fatigue_slope:.3f} Hz/s ({max_fatigue_pct:.1f}%)",
    delta_color="inverse"
)

col2.metric(
    label="Maior Recrutamento (RMS Slope)",
    value=f"{max_rec_ch.replace('_', ' ').title()}",
    delta=f"+{max_rec_slope:.3f} uV/s (+{max_rec_pct:.1f}%)",
    delta_color="normal"
)

col3.metric(
    label="Canais Analisados",
    value=f"{len(results)} canais",
    delta="30 segundos"
)

col4.metric(
    label="Taxa de Amostragem",
    value=f"{int(fs)} Hz",
    delta=f"{int(fs*30)} amostras"
)

st.markdown("---")

# -----------------------------------------------------------------------------
# TABELA DETALHADA DE REGRESSÃO LINEAR
# -----------------------------------------------------------------------------
st.markdown("### 📋 Resultados Detalhados das Regressões Lineares")

summary_data = []
for ch, m in results.items():
    summary_data.append({
        'Músculo / Canal': ch.replace('_', ' ').title(),
        'MDF Inicial (Hz)': round(m['MDF_Initial_Hz'], 1),
        'MDF Final (Hz)': round(m['MDF_Final_Hz'], 1),
        'MDF Slope (Hz/s)': round(m['MDF_Slope_Hz_s'], 3),
        'Queda MDF (%)': f"{m['MDF_Change_Percent']:.1f}%",
        'R² (MDF)': round(m['MDF_R2'], 2),
        'p-valor (MDF)': round(m['MDF_p_value'], 4),
        'RMS Slope (uV/s)': round(m['RMS_Slope_uV_s'], 3),
        'Aumento RMS (%)': f"+{m['RMS_Change_Percent']:.1f}%",
        'R² (RMS)': round(m['RMS_R2'], 2)
    })

df_summary = pd.DataFrame(summary_data)
st.dataframe(df_summary, use_container_width=True)

# -----------------------------------------------------------------------------
# VISUALIZAÇÃO GRÁFICA DE REGRESSÃO
# -----------------------------------------------------------------------------
st.markdown("### 📉 Curvas de Regressão Linear (MDF e RMS vs. Tempo)")

os.makedirs('/workspace/scratch', exist_ok=True)
chart_path = '/workspace/scratch/grafico_streamlit_fadiga.png'
analyzer.plot_fatigue_summary(time_series, results, chart_path)

if os.path.exists(chart_path):
    st.image(chart_path, caption="Evolução da Frequência Mediana (esquerda) e Amplitude RMS (direita) com retas de regressão ajustadas por mínimos quadrados.", use_column_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# EXPORTAÇÃO E DOWNLOADS
# -----------------------------------------------------------------------------
st.markdown("### 📥 Exportar Resultados e Relatórios")

col_down1, col_down2 = st.columns(2)

# 1. Download do CSV de Métricas
csv_data = df_summary.to_csv(index=False).encode('utf-8')
col_down1.download_button(
    label="📄 Baixar Tabela de Resultados (CSV)",
    data=csv_data,
    file_name="resultados_fadiga_emg.csv",
    mime="text/csv",
    use_container_width=True
)

# 2. Download do Relatório PDF
pdf_path = '/workspace/scratch/relatorio_streamlit_fadiga.pdf'
generate_pdf_report(results, chart_path, pdf_path)

if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    col_down2.download_button(
        label="📑 Baixar Relatório Acadêmico Completo (PDF)",
        data=pdf_bytes,
        file_name="relatorio_fadiga_emg.pdf",
        mime="application/pdf",
        use_container_width=True
    )
