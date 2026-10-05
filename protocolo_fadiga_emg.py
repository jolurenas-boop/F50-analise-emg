import os
import sys
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

# Configurações visuais
sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')
plt.rcParams['figure.dpi'] = 150

# Lista Padrão Sequencial dos 8 Canais
DEFAULT_CHANNELS = [
    'Peitoral maior direito (PMD)',
    'Peitoral maior esquerdo (PME)',
    'Tríceps braquial direito (TBD)',
    'Tríceps braquial esquerdo (TBE)',
    'Deltóide direito (DLD)',
    'Deltóide esquerdo (DLE)',
    'Célula de carga direita (CCD)',
    'Célula de carga esquerda (CCE)'
]

def get_safe_temp_dir():
    """
    Retorna um diretório temporário seguro no diretório de execução atual
    ou na pasta temporária do SO, prevenindo PermissionError.
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

class EMGFatigueAnalyzer:
    """
    Analisador Dedicado de Fadiga Eletromiográfica (sEMG).
    Processa registros de 30s a 2000 Hz.
    """
    def __init__(self, fs=2000.0, notch_freq=60.0):
        self.fs = fs
        self.notch_freq = notch_freq

    def preprocess_channel(self, raw_signal, low_cut=20.0, high_cut=450.0):
        """
        Condicionamento do sinal de sEMG de acordo com SENIAM e ISEK:
        1. Filtro Notch de 60 Hz (Q=35) de fase zero (filtfilt)
        2. Filtro Passa-Banda Butterworth 4ª Ordem (20 - 450 Hz) de fase zero
        3. Retificação de onda completa (valor absoluto)
        4. Envelope Linear com filtro Passa-Baixas Butterworth 4ª Ordem (6 Hz)
        """
        nyq = 0.5 * self.fs
        
        # 1. Notch Filter
        w0 = self.notch_freq / nyq
        b_notch, a_notch = signal.iirnotch(w0, Q=35.0)
        filtered_notch = signal.filtfilt(b_notch, a_notch, raw_signal)
        
        # 2. Band-pass Filter
        low = low_cut / nyq
        high = high_cut / nyq
        b_band, a_band = signal.butter(4, [low, high], btype='band')
        filtered_emg = signal.filtfilt(b_band, a_band, filtered_notch)
        
        # 3. Retificação
        rectified_emg = np.abs(filtered_emg)
        
        # 4. Envelope Linear (6 Hz)
        b_low, a_low = signal.butter(4, 6.0 / nyq, btype='low')
        envelope = signal.filtfilt(b_low, a_low, rectified_emg)
        envelope = np.clip(envelope, 0, None)
        
        return filtered_emg, rectified_emg, envelope

    def compute_spectral_metrics(self, window_signal):
        """
        Calcula a Frequência Mediana (MDF) e Média (MNF) via Welch PSD.
        """
        n = len(window_signal)
        freqs, psd = signal.welch(window_signal, fs=self.fs, nperseg=min(n, 1024), window='hann')
        
        total_power = np.sum(psd)
        if total_power == 0:
            return 0.0, 0.0
            
        mnf = np.sum(freqs * psd) / total_power
        
        cum_power = np.cumsum(psd)
        mdf_idx = np.where(cum_power >= (total_power / 2.0))[0][0]
        mdf = freqs[mdf_idx]
        
        return float(mdf), float(mnf)

    def analyze_fatigue(self, df_emg, window_size_sec=1.0, overlap_percent=50.0, channel_names=None):
        """
        Processa cada canal de sEMG em janelas móveis:
        - Regressão Linear da Frequência Mediana (MDF Slope, R², p-value)
        - Regressão Linear da Amplitude RMS (RMS Slope, R², p-value)
        """
        if channel_names is None:
            # Seleciona canais ignorando a coluna de tempo e células de carga se identificadas
            channel_names = [
                col for col in df_emg.columns 
                if col.lower() != 'tempo' and not col.startswith('Célula')
            ]
            
        n_samples = len(df_emg)
        window_samples = int(window_size_sec * self.fs)
        step_samples = int(window_samples * (1.0 - overlap_percent / 100.0))
        
        results = {}
        time_series_data = {}
        
        for ch in channel_names:
            raw_sig = df_emg[ch].values
            filt_sig, rect_sig, env_sig = self.preprocess_channel(raw_sig)
            
            time_centers = []
            mdf_list = []
            mnf_list = []
            rms_list = []
            mav_list = []
            
            start = 0
            while start + window_samples <= n_samples:
                end = start + window_samples
                win_filt = filt_sig[start:end]
                win_rect = rect_sig[start:end]
                
                t_center = (start + window_samples / 2.0) / self.fs
                time_centers.append(t_center)
                
                mdf, mnf = self.compute_spectral_metrics(win_filt)
                mdf_list.append(mdf)
                mnf_list.append(mnf)
                
                rms = np.sqrt(np.mean(win_filt ** 2))
                mav = np.mean(win_rect)
                rms_list.append(rms)
                mav_list.append(mav)
                
                start += step_samples
                
            time_centers = np.array(time_centers)
            mdf_array = np.array(mdf_list)
            mnf_array = np.array(mnf_list)
            rms_array = np.array(rms_list)
            mav_array = np.array(mav_list)
            
            # Regressão Linear - MDF
            res_mdf = linregress(time_centers, mdf_array)
            mdf_initial = res_mdf.intercept + res_mdf.slope * time_centers[0]
            mdf_final = res_mdf.intercept + res_mdf.slope * time_centers[-1]
            mdf_pct_change = ((mdf_final - mdf_initial) / mdf_initial) * 100.0 if mdf_initial != 0 else 0.0
            
            # Regressão Linear - RMS
            res_rms = linregress(time_centers, rms_array)
            rms_initial = res_rms.intercept + res_rms.slope * time_centers[0]
            rms_final = res_rms.intercept + res_rms.slope * time_centers[-1]
            rms_pct_change = ((rms_final - rms_initial) / rms_initial) * 100.0 if rms_initial != 0 else 0.0
            
            results[ch] = {
                'MDF_Slope_Hz_s': float(res_mdf.slope),
                'MDF_Intercept': float(res_mdf.intercept),
                'MDF_R2': float(res_mdf.rvalue ** 2),
                'MDF_p_value': float(res_mdf.pvalue),
                'MDF_Initial_Hz': float(mdf_initial),
                'MDF_Final_Hz': float(mdf_final),
                'MDF_Change_Percent': float(mdf_pct_change),
                
                'RMS_Slope_uV_s': float(res_rms.slope),
                'RMS_Intercept': float(res_rms.intercept),
                'RMS_R2': float(res_rms.rvalue ** 2),
                'RMS_p_value': float(res_rms.pvalue),
                'RMS_Initial_uV': float(rms_initial),
                'RMS_Final_uV': float(rms_final),
                'RMS_Change_Percent': float(rms_pct_change),
                
                'Mean_MDF': float(np.mean(mdf_array)),
                'Mean_RMS': float(np.mean(rms_array))
            }
            
            time_series_data[ch] = {
                'time_centers': time_centers,
                'mdf': mdf_array,
                'mnf': mnf_array,
                'rms': rms_array,
                'mav': mav_array,
                'raw': raw_sig,
                'envelope': env_sig
            }
            
        return results, time_series_data

    def plot_fatigue_summary(self, time_series_data, results, output_filepath):
        """
        Gera o painel visual com gráficos de regressão para MDF e RMS.
        """
        channels = list(results.keys())
        n_ch = len(channels)
        
        fig, axes = plt.subplots(n_ch, 2, figsize=(14, 3 * n_ch), sharex='col')
        if n_ch == 1:
            axes = np.expand_dims(axes, axis=0)
            
        fig.suptitle('Análise de Fadiga Eletromiográfica (sEMG - 30s a 50% CIVM)', fontsize=15, fontweight='bold', y=0.99)
        
        for i, ch in enumerate(channels):
            t_win = time_series_data[ch]['time_centers']
            mdf = time_series_data[ch]['mdf']
            rms = time_series_data[ch]['rms']
            res_ch = results[ch]
            
            # Subplot 1: MDF
            ax_mdf = axes[i, 0]
            ax_mdf.plot(t_win, mdf, 'o-', color='tab:blue', alpha=0.6, markersize=4, label='MDF Observado')
            reg_mdf_line = res_ch['MDF_Intercept'] + res_ch['MDF_Slope_Hz_s'] * t_win
            ax_mdf.plot(t_win, reg_mdf_line, 'r--', linewidth=2, 
                        label=f"Regressão: {res_ch['MDF_Slope_Hz_s']:.3f} Hz/s (R²={res_ch['MDF_R2']:.2f}, p={res_ch['MDF_p_value']:.3f})")
            ax_mdf.set_ylabel(f"{ch}\nMDF (Hz)", fontsize=8, fontweight='bold')
            ax_mdf.legend(loc='upper right', fontsize=8)
            ax_mdf.grid(True, alpha=0.3)
            
            # Subplot 2: RMS
            ax_rms = axes[i, 1]
            ax_rms.plot(t_win, rms, 's-', color='tab:orange', alpha=0.6, markersize=4, label='RMS Observado')
            reg_rms_line = res_ch['RMS_Intercept'] + res_ch['RMS_Slope_uV_s'] * t_win
            ax_rms.plot(t_win, reg_rms_line, 'g--', linewidth=2, 
                        label=f"Regressão: +{res_ch['RMS_Slope_uV_s']:.3f} uV/s (R²={res_ch['RMS_R2']:.2f}, p={res_ch['RMS_p_value']:.3f})")
            ax_rms.set_ylabel(f"{ch}\nRMS (uV)", fontsize=8, fontweight='bold')
            ax_rms.legend(loc='upper left', fontsize=8)
            ax_rms.grid(True, alpha=0.3)
            
        axes[-1, 0].set_xlabel('Tempo de Sustentação (s)', fontsize=11, fontweight='bold')
        axes[-1, 1].set_xlabel('Tempo de Sustentação (s)', fontsize=11, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Garante diretório seguro
        target_dir = os.path.dirname(output_filepath)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
            
        fig.savefig(output_filepath, dpi=150, bbox_inches='tight')
        plt.close()

def generate_simulated_emg_fatigue(duration=30.0, fs=2000.0):
    """
    Gera dados simulados contendo os 8 canais padrão.
    """
    t = np.arange(0, duration, 1.0 / fs)
    n_samples = len(t)
    
    sim_configs = [
        ('Peitoral maior direito (PMD)', 160.0, -0.25, 0.20),
        ('Peitoral maior esquerdo (PME)', 620.0, -0.80, 1.75),
        ('Tríceps braquial direito (TBD)', 350.0, -1.10, 1.10),
        ('Tríceps braquial esquerdo (TBE)', 190.0, -0.10, 0.25),
        ('Deltóide direito (DLD)', 320.0, -0.50, 0.65),
        ('Deltóide esquerdo (DLE)', 740.0, -0.45, 1.50),
        ('Célula de carga direita (CCD)', 28.0, 0.00, 0.01),
        ('Célula de carga esquerda (CCE)', 22.0, 0.00, -0.02)
    ]
    
    df_data = {'tempo': t}
    
    for name, base_val, mdf_rate, rms_rate in sim_configs:
        if name.startswith("Célula"):
            # Célula de carga (Sinal mecânico constante com pequeno ruído)
            df_data[name] = np.ones(n_samples) * base_val + np.random.normal(0, 0.1, n_samples)
        else:
            # sEMG
            white = np.random.normal(0, 1.0, n_samples)
            b, a = signal.butter(4, [30.0/(fs/2), 220.0/(fs/2)], btype='band')
            colored = signal.filtfilt(b, a, white)
            amp_mod = 1.0 + (rms_rate * t / base_val)
            sig = colored * base_val * amp_mod + 8.0 * np.sin(2 * np.pi * 60.0 * t)
            df_data[name] = sig
            
    return pd.DataFrame(df_data)

if __name__ == '__main__':
    print("Executando módulo de teste de fadiga sEMG...")
    analyzer = EMGFatigueAnalyzer(fs=2000.0, notch_freq=60.0)
    df_sim = generate_simulated_emg_fatigue(duration=30.0, fs=2000.0)
    res, time_series = analyzer.analyze_fatigue(df_sim)
    
    safe_dir = get_safe_temp_dir()
    plot_path = os.path.join(safe_dir, 'grafico_fadiga_emg.png')
    analyzer.plot_fatigue_summary(time_series, res, plot_path)
    print(f"Sucesso! Gráfico salvo em: {plot_path}")
