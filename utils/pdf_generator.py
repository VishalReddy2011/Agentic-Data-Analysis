from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
import os
import pandas as pd
import math

def get_hardcoded_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Heading1', fontSize=20, spaceAfter=14))
    styles.add(ParagraphStyle(name='Heading2', fontSize=16, spaceAfter=12))
    styles.add(ParagraphStyle(name='Heading3', fontSize=12, spaceAfter=6, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='BodyText', fontSize=10, spaceAfter=6, leading=12))
    return styles

def _format_value(v):
    if v is None:
        return ''
    if isinstance(v, float):
        if math.isfinite(v):
            return f"{v:.6g}"
        return str(v)
    return str(v)

def dataframe_to_table(df: pd.DataFrame, styles):
    if df is None or df.empty:
        return Paragraph("No data available.", styles['BodyText'])
    cols = list(df.columns)
    header = [''] + [str(c) for c in cols]
    rows = []
    for idx in df.index:
        row = [str(idx)] + [_format_value(df.at[idx, col]) for col in cols]
        rows.append(row)
    data = [header] + rows
    tbl = Table(data, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f2f2f2')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    return tbl

def build_report(session_id: str, descriptive_stats, list_of_insights, output_dir='reports'):
    os.makedirs(output_dir, exist_ok=True)
    doc_path = os.path.join(output_dir, f"{session_id}.pdf")
    doc = SimpleDocTemplate(doc_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = get_hardcoded_styles()
    story = []
    story.append(Paragraph("Data Analysis Report", styles['Heading1']))
    story.append(Paragraph(f"Report ID: {session_id}", styles['BodyText']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Descriptive Statistics", styles['Heading2']))
    story.append(Paragraph(f"Total Rows: {getattr(descriptive_stats, 'file_length', 'N/A')}", styles['BodyText']))
    story.append(Paragraph(f"Total Columns: {getattr(descriptive_stats, 'total_columns', 'N/A')}", styles['BodyText']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Numerical Column Statistics", styles['Heading3']))
    num_tbl = dataframe_to_table(getattr(descriptive_stats, 'numeric_stats_df', None), styles)
    story.append(num_tbl)
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Categorical Column Statistics", styles['Heading3']))
    cat_tbl = dataframe_to_table(getattr(descriptive_stats, 'categorical_stats_df', None), styles)
    story.append(cat_tbl)
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Categorical Value Counts (top 20)", styles['Heading3']))
    vc = getattr(descriptive_stats, 'value_counts', {}) or {}
    for col, mapping in vc.items():
        story.append(Paragraph(str(col), styles['BodyText']))
        rows = [["Value", "Count"]]
        for k, v in mapping.items():
            rows.append([str(k), str(v)])
        table = Table(rows, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f7f7f7')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("AI-Generated Insights", styles['Heading2']))
    for i, insight in enumerate(list_of_insights or []):
        technique = getattr(insight, 'technique_used', '') or ''
        text = getattr(insight, 'insight_text', '') or ''
        graph = getattr(insight, 'graph_filename', None)
        story.append(Paragraph(f"Insight {i+1}: {technique}", styles['Heading3']))
        story.append(Paragraph(text, styles['BodyText']))
        if graph:
            try:
                img_path = graph
                if os.path.exists(img_path):
                    story.append(Spacer(1, 0.08*inch))
                    story.append(Image(img_path, width=5*inch, height=3.5*inch))
            except Exception:
                pass
        story.append(Spacer(1, 0.2*inch))
    doc.build(story)
    return doc_path
