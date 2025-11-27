from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os


def get_hardcoded_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="MyHeading1", fontSize=20, spaceAfter=14))
    styles.add(ParagraphStyle(name="MyHeading2", fontSize=16, spaceAfter=10))
    styles.add(ParagraphStyle(name="MyBodyText", fontSize=12, leading=14))
    return styles


def build_report(session_id, descriptive_stats, insights, output_dir="./reports"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{session_id}_report.pdf")
    styles = get_hardcoded_styles()

    story = []
    story.append(Paragraph("Data Analysis Report", styles["MyHeading1"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Summary Statistics", styles["MyHeading2"]))
    story.append(Spacer(1, 10))

    ds = descriptive_stats or {}

    simple_stats = {
        "file_length": ds.get("file_length"),
        "total_columns": ds.get("total_columns"),
        "numeric_columns": ", ".join(ds.get("numeric_columns", [])),
        "categorical_columns": ", ".join(ds.get("categorical_columns", [])),
    }

    stats_table = []
    for key, value in simple_stats.items():
        stats_table.append([str(key), str(value)])

    table = Table(stats_table, colWidths=[200, 300])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Insights", styles["MyHeading2"]))
    story.append(Spacer(1, 10))

    for idx, ins in enumerate(insights):
        story.append(Paragraph(f"Insight {idx+1}", styles["MyHeading2"]))
        story.append(Paragraph(ins.insight_text, styles["MyBodyText"]))
        story.append(Paragraph(f"Technique Used: {ins.technique_used}", styles["MyBodyText"]))
        story.append(Spacer(1, 10))

        image_path = getattr(ins, "graph_filename", None)
        if image_path and os.path.exists(image_path):
            story.append(Image(image_path, width=400, height=300))
            story.append(Spacer(1, 20))

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    doc.build(story)

    return file_path
