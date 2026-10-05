import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether, Flowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

COLORS = {
    'heading':    HexColor('#1A365D'), # Navy Escuro
    'body':       HexColor('#2D3748'), # Cinza Escuro
    'accent':     HexColor('#3182CE'), # Azul Vivo
    'danger':     HexColor('#E53E3E'), # Vermelho
    'muted':      HexColor('#718096'), # Cinza Médio
    'bg_alt':     HexColor('#F7FAFC'), # Off-white
    'bg_header':  HexColor('#2B6CB0'), # Azul Cabeçalho
    'white':      HexColor('#FFFFFF'),
}

class SectionDivider(Flowable):
    def __init__(self, width, colors):
        Flowable.__init__(self)
        self._width = width
        self.colors = colors
        self._height = 12

    def wrap(self, availWidth, availHeight):
        return self._width, self._height

    def draw(self):
        y = self._height / 2
        self.canv.setStrokeColor(self.colors['accent'])
        self.canv.setLineWidth(1)
        self.canv.line(0, y, self._width, y)

def generate_pdf_report(results_dict, chart_image_path, output_pdf_path):
    """
    Gera um relatório acadêmico/clínico em PDF padronizado para análise de fadiga sEMG.
    """
    doc = BaseDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    usable_w = A4[0] - 4*cm
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('DocTitle', fontName='Helvetica-Bold', fontSize=20, textColor=COLORS['heading'], leading=24, spaceAfter=6))
    styles.add(ParagraphStyle('DocSubtitle', fontName='Helvetica', fontSize=10, textColor=COLORS['muted'], leading=14, spaceAfter=15))
    styles.add(ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=14, textColor=COLORS['heading'], leading=18, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=11, textColor=COLORS['heading'], leading=15, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle('Body', fontName='Helvetica', fontSize=9.5, textColor=COLORS['body'], leading=13.5, spaceAfter=6, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle('Caption', fontName='Helvetica-Oblique', fontSize=8, textColor=COLORS['muted'], leading=11, alignment=TA_CENTER))
    styles.add(ParagraphStyle('TableHead', fontName='Helvetica-Bold', fontSize=8.5, textColor=COLORS['white'], leading=11, alignment=TA_CENTER))
    styles.add(ParagraphStyle('TableBody', fontName='Helvetica', fontSize=8.5, textColor=COLORS['body'], leading=11, alignment=TA_CENTER))
    styles.add(ParagraphStyle('TableBodyLeft', fontName='Helvetica-Bold', fontSize=8.5, textColor=COLORS['heading'], leading=11, alignment=TA_LEFT))

    frame = Frame(doc.leftMargin, doc.bottomMargin, usable_w, A4[1] - 4*cm, id='main')
    
    def on_page(canvas, doc_obj):
        canvas.saveState()
        # Header Line
        canvas.setStrokeColor(COLORS['accent'])
        canvas.setLineWidth(0.8)
        canvas.line(2*cm, A4[1] - 1.5*cm, A4[0] - 2*cm, A4[1] - 1.5*cm)
        canvas.setFont('Helvetica-Bold', 8)
        canvas.setFillColor(COLORS['muted'])
        canvas.drawString(2*cm, A4[1] - 1.3*cm, "RELATÓRIO DE FADIGA ELETROMIOGRÁFICA (sEMG)")
        canvas.drawRightString(A4[0] - 2*cm, A4[1] - 1.3*cm, "PROTOCOLO ISOMÉTRICO 30s @ 2000Hz")
        
        # Footer Line
        canvas.setStrokeColor(COLORS['bg_alt'])
        canvas.setLineWidth(0.5)
        canvas.line(2*cm, 1.5*cm, A4[0] - 2*cm, 1.5*cm)
        canvas.setFont('Helvetica', 8)
        canvas.drawCentredString(A4[0]/2, 1.1*cm, f"Página {doc_obj.page}")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id='main_page', frames=frame, onPage=on_page)])
    
    story = []
    
    # Título do Relatório
    story.append(Paragraph("Relatório de Análise de Fadiga Eletromiográfica", styles['DocTitle']))
    story.append(Paragraph("Avaliação Isométrica Sustentada (50% CIVM) | Frequência de Amostragem: 2000 Hz | Duração: 30s", styles['DocSubtitle']))
    story.append(SectionDivider(usable_w, COLORS))
    story.append(Spacer(1, 8))
    
    # Seção 1: Resumo Metodológico
    story.append(Paragraph("1. Metodologia de Processamento e Análise Espectral", styles['H1']))
    resumo_texto = (
        "O sinal de eletromiografia de superfície (sEMG) foi pré-processado por meio de filtração digital de fase zero "
        "(filtração bidirecional progressiva e regressiva para evitar atraso de fase). Aplicou-se um filtro Notch de 60 Hz (Q=35.0) "
        "para atenuação de interferências da rede elétrica, seguido por um filtro passa-banda Butterworth de 4ª ordem (20 - 450 Hz). "
        "A análise de fadiga foi computada via janelas móveis de 1,0 segundo com 50% de sobreposição (overlap). Para cada janela, "
        "extraiu-se a Frequência Mediana (MDF) e a Média (MNF) por meio da Densidade Espectral de Potência (método de Welch com janela de Hanning), "
        "além da amplitude RMS (Root Mean Square). A taxa de fadiga local foi quantificada pela inclinação da reta de regressão linear (MDF Slope, Hz/s) "
        "e o recrutamento compensatório pelo ganho de amplitude (RMS Slope, uV/s)."
    )
    story.append(Paragraph(resumo_texto, styles['Body']))
    story.append(Spacer(1, 8))
    
    # Seção 2: Tabela de Resultados
    story.append(Paragraph("2. Tabela Comparativa de Fadiga e Recrutamento Neural", styles['H1']))
    
    headers = [
        Paragraph("Músculo / Canal", styles['TableHead']),
        Paragraph("MDF Inicial<br/>(Hz)", styles['TableHead']),
        Paragraph("MDF Final<br/>(Hz)", styles['TableHead']),
        Paragraph("MDF Slope<br/>(Hz/s)", styles['TableHead']),
        Paragraph("Queda MDF<br/>(%)", styles['TableHead']),
        Paragraph("RMS Slope<br/>(uV/s)", styles['TableHead']),
        Paragraph("R² (MDF)", styles['TableHead']),
        Paragraph("p-valor", styles['TableHead'])
    ]
    
    rows = []
    for ch, metrics in results_dict.items():
        ch_clean = ch.replace('_', ' ').title()
        rows.append([
            Paragraph(ch_clean, styles['TableBodyLeft']),
            Paragraph(f"{metrics['MDF_Initial_Hz']:.1f}", styles['TableBody']),
            Paragraph(f"{metrics['MDF_Final_Hz']:.1f}", styles['TableBody']),
            Paragraph(f"{metrics['MDF_Slope_Hz_s']:.3f}", styles['TableBody']),
            Paragraph(f"{metrics['MDF_Change_Percent']:.1f}%", styles['TableBody']),
            Paragraph(f"+{metrics['RMS_Slope_uV_s']:.3f}", styles['TableBody']),
            Paragraph(f"{metrics['MDF_R2']:.2f}", styles['TableBody']),
            Paragraph(f"{metrics['MDF_p_value']:.3f}", styles['TableBody'])
        ])
        
    col_widths = [usable_w * 0.24, usable_w * 0.10, usable_w * 0.10, usable_w * 0.12, usable_w * 0.11, usable_w * 0.12, usable_w * 0.10, usable_w * 0.11]
    
    t = Table([headers] + rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLORS['bg_header']),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLORS['white'], COLORS['bg_alt']]),
        ('GRID', (0, 0), (-1, -1), 0.5, COLORS['muted']),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 12))
    
    # Seção 3: Visualização dos Gráficos de Regressão Linear
    if os.path.exists(chart_image_path):
        story.append(Paragraph("3. Curvas de Regressão Linear (MDF e RMS Slopes)", styles['H1']))
        img_w = usable_w
        img_h = usable_w * 0.55
        story.append(Image(chart_image_path, width=img_w, height=img_h))
        story.append(Spacer(1, 4))
        story.append(Paragraph("Figura 1: Evolução da Frequência Mediana (MDF) e Amplitude RMS ao longo do tempo de sustentação.", styles['Caption']))
        story.append(Spacer(1, 10))
        
    # Seção 4: Interpretação Fisiológica
    story.append(Paragraph("4. Discussão Fisiológica e Diagnóstico de Fadiga", styles['H1']))
    
    # Encontrar o músculo com maior taxa de fadiga
    max_fatigue_ch = min(results_dict.keys(), key=lambda k: results_dict[k]['MDF_Slope_Hz_s'])
    max_fatigue_val = results_dict[max_fatigue_ch]['MDF_Slope_Hz_s']
    max_fatigue_pct = results_dict[max_fatigue_ch]['MDF_Change_Percent']
    
    disc_texto = (
        f"A avaliação da fadiga muscular sustentada a 50% da CIVM evidencia a resposta eletrofisiológica adaptativa do sistema neuromuscular. "
        f"O músculo com maior taxa de fadiga metabólica periférica foi o <b>{max_fatigue_ch.replace('_', ' ').title()}</b>, apresentando uma inclinação espectral "
        f"de <b>{max_fatigue_val:.3f} Hz/s</b> (variação acumulada de <b>{max_fatigue_pct:.1f}%</b> na MDF). "
        f"O declínio na Frequência Mediana (MDF Slope negativo) reflete o acúmulo de metabólitos ácidos (íons H+ e lactato) e a consequente redução na velocidade de condução "
        f"do potencial de ação na membrana muscular. Concomitantemente, observa-se uma inclinação positiva na amplitude sEMG RMS (RMS Slope positivo), "
        f"caracterizando o recrutamento neural compensatório e o aumento na frequência de disparo das unidades motoras para manter o nível submáximo de força exigido pela tarefa."
    )
    story.append(Paragraph(disc_texto, styles['Body']))
    
    doc.build(story)
    print(f"Relatório PDF gerado com sucesso em: {output_pdf_path}")

if __name__ == '__main__':
    # Teste de geração de PDF
    from protocolo_fadiga_emg import EMGFatigueAnalyzer, generate_simulated_emg_fatigue
    
    analyzer = EMGFatigueAnalyzer(fs=2000.0)
    df_sim = generate_simulated_emg_fatigue()
    res, time_series = analyzer.analyze_fatigue(df_sim)
    
    os.makedirs('/workspace/scratch', exist_ok=True)
    img_path = '/workspace/scratch/grafico_fadiga_emg.png'
    analyzer.plot_fatigue_summary(time_series, res, img_path)
    
    pdf_path = '/workspace/scratch/relatorio_fadiga_emg.pdf'
    generate_pdf_report(res, img_path, pdf_path)
