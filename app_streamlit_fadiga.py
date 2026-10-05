import os
import io
import json
import tempfile
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from protocolo_fadiga_emg import EMGFatigueAnalyzer, generate_simulated_emg_fatigue, DEFAULT_CHANNELS
from gerar_pdf_fadiga import generate_pdf_report

# -----------------------------------------------------------------------------
# FUNÇÃO PARA DIRETÓRIO TEMPORÁRIO SEGURO (PREVINE PERMISSIONERROR)
# -----------------------------------------------------------------------------
def get_safe_temp_dir():
    """
    Cria e retorna um diretório temporário seguro no diretório de execução atual
    ou na pasta temporária do SO, prevenindo PermissionError em nuvens como Streamlit Cloud.
    """
    candidates = [
        os.path.join(os.getcwd(), "scratch"),
        os.path.join(os.getcwd(), "temp_output"),
        tempfile.gettempdir()
    ]
    for target in candidates:
        try:
            os.makedirs(target, exist_ok=True)
            test_path = os.path.join(target, ".write_test")
            with open(test_path, "w") as f:
                f.write("ok")
            os.remove(test_path)
            return target
        except Exception:
            continue
    return tempfile.gettempdir()

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
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚡ Portal de Análise de Fadiga Eletromiográfica (sEMG)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Plataforma dedicada para processamento de sinais de sEMG de 30 segundos (2000 Hz) a 50% da CIVM</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# BARRA LATERAL: CONFIGURAÇÕES E UPLOAD DE ARQUIVO
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configurações da Análise")

# 1. Upload do Arquivo
uploaded_file = st.sidebar.file_uploader(
    "Carregar Arquivo da Coleta (.txt, .csv, .tsv)",
    type=["txt", "csv", "tsv"],
    help="Selecione o arquivo bruto com as colunas adquiridas a 2000 Hz."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Parâmetros do Sinal")

fs = st.sidebar.number_input("Frequência de Amostragem (Hz)", value=2000.0, step=100.0)
notch_freq = st.sidebar.selectbox("Frequência do Filtro Notch (Hz)", options=[60.0, 50.0], index=0)
window_size = st.sidebar.slider("Tamanho da Janela Móvel (s)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
overlap_pct = st.sidebar.slider("Sobreposição / Overlap (%)", min_value=0.0, max_value=75.0, value=50.0, step=5.0)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Dica de Formato:** O leitor mapeia automaticamente até 8 canais na sequência padronizada:\n"
               "1- Peitoral maior direito (PMD)\n"
               "2- Peitoral maior esquerdo (PME)\n"
               "3- Tríceps braquial direito (TBD)\n"
               "4- Tríceps braquial esquerdo (TBE)\n"
               "5- Deltóide direito (DLD)\n"
               "6- Deltóide esquerdo (DLE)\n"
               "7- Célula de carga direita (CCD)\n"
               "8- Célula de carga esquerda (CCE)")

# -----------------------------------------------------------------------------
# FUNÇÃO DE LEITURA BLINDADA DE DADOS COM MAPEAMENTO DOS 8 CANAIS
# -----------------------------------------------------------------------------
@st.cache_data
def load_emg_data(file_bytes, filename, fs_rate):
    """
    Lê o arquivo de forma blindada, detectando linhas de metadados e delimitadores.
    Mapeia rigorosamente a sequência de 8 canais solicitada.
    """
    content = file_bytes.decode('latin1', errors='replace').splitlines()
    
    # Encontra a primeira linha com conteúdo numérico
    skip_rows = 0
    for i, line in enumerate(content[:50]):
        parts = line.strip().split()
        if len(parts) > 0:
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
    
    # Se a primeira coluna parecer ser o tempo, ajusta
    first_col = df.iloc[:, 0].values
    if len(first_col) > 1 and np.all(np.diff(first_col) > 0) and first_col[-1] < 100.0:
        df = df.iloc[:, 1:]
        
    num_channels = df.shape[1]
    
    # Sequência Padronizada de Nomes
    if num_channels <= len(DEFAULT_CHANNELS):
        col_names = DEFAULT_CHANNELS[:num_channels]
    else:
        col_names = DEFAULT_CHANNELS + [f"Canal {i+1}" for i in range(len(DEFAULT_CHANNELS), num_channels)]
        
    df.columns = col_names
    df['tempo'] = np.arange(len(df)) / fs_rate
    
    return df

# -----------------------------------------------------------------------------
# PROCESSAMENTO DE ARQUIVO OU MODO DEMONSTRAÇÃO
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        df_emg = load_emg_data(uploaded_file.getvalue(), uploaded_file.name, fs)
        st.success(f"✅ Arquivo `{uploaded_file.name}` carregado com sucesso! ({len(df_emg)} amostras | {len(df_emg)/fs:.1f} segundos | {df_emg.shape[1]-1} canais)")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}. Verifique o formato do arquivo.")
        st.stop()
else:
    st.warning("⚠️ Nenhum arquivo carregado. Exibindo demonstração simulada de 30s @ 2000 Hz com os 8 canais.")
    df_emg = generate_simulated_emg_fatigue(duration=30.0, fs=fs)

# Mostra prévia dos dados
with st.expander("📊 Visualizar Prévia dos Dados Brutos"):
    st.dataframe(df_emg.head(10), use_container_width=True)

# -----------------------------------------------------------------------------
# PROCESSAMENTO DA ANÁLISE DE FADIGA
# -----------------------------------------------------------------------------
# Filtra apenas os canais de sEMG para análise espectral (exclui células de carga)
emg_channels_to_analyze = [
    c for c in df_emg.columns 
    if c.lower() != 'tempo' and not c.startswith('Célula')
]

analyzer = EMGFatigueAnalyzer(fs=fs, notch_freq=notch_freq)
results, time_series = analyzer.analyze_fatigue(
    df_emg, 
    window_size_sec=window_size, 
    overlap_percent=overlap_pct,
    channel_names=emg_channels_to_analyze
)

# -----------------------------------------------------------------------------
# PAINEL DE CARDS E KPIs PRINCIPAIS
# -----------------------------------------------------------------------------
st.markdown("### 📈 Diagnóstico Geral de Fadiga (sEMG)")

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
    value=f"{max_fatigue_ch}",
    delta=f"{max_fatigue_slope:.3f} Hz/s ({max_fatigue_pct:.1f}%)",
    delta_color="inverse"
)

col2.metric(
    label="Maior Recrutamento (RMS Slope)",
    value=f"{max_rec_ch}",
    delta=f"+{max_rec_slope:.3f} uV/s (+{max_rec_pct:.1f}%)",
    delta_color="normal"
)

col3.metric(
    label="Músculos Analisados",
    value=f"{len(results)} canais",
    delta="sEMG"
)

col4.metric(
    label="Taxa de Amostragem",
    value=f"{int(fs)} Hz",
    delta=f"{int(len(df_emg))} amostras"
)

st.markdown("---")

# -----------------------------------------------------------------------------
# TABELA DETALHADA DE REGRESSÃO LINEAR
# -----------------------------------------------------------------------------
st.markdown("### 📋 Resultados Detalhados das Regressões Lineares")

summary_data = []
for ch, m in results.items():
    summary_data.append({
        'Músculo / Canal': ch,
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
# VISUALIZAÇÃO GRÁFICA DE REGRESSÃO COM DIRETÓRIO SEGURO
# -----------------------------------------------------------------------------
st.markdown("### 📉 Curvas de Regressão Linear (MDF e RMS vs. Tempo)")

output_dir = get_safe_temp_dir()
chart_path = os.path.join(output_dir, 'grafico_streamlit_fadiga.png')
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

# 2. Download do Relatório PDF usando Diretório Seguro
pdf_path = os.path.join(output_dir, 'relatorio_streamlit_fadiga.pdf')
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
