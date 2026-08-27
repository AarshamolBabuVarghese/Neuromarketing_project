"""
generate_client_report.py
---------------------------
Turns reports/analysis_summary.json + reports/ab_test_results.json
into a polished, non-technical Word document for the client:
reports/client_report.docx
"""
import json
import os
import yaml
from docx import Document
from docx.shared import Pt
from datetime import date

from analysis import run as run_analysis
from ab_testing import run as run_ab_testing

with open("config/config.yaml", "r") as f:
    _CFG = yaml.safe_load(f)

REPORTS_DIR = _CFG["paths"]["reports_dir"]


def build_report(summary: dict, ab_results: dict, client_name="Client"):
    doc = Document()

    doc.add_heading("Neuromarketing Insights Report", level=0)
    doc.add_paragraph(f"Prepared for: {client_name}")
    doc.add_paragraph(f"Date: {date.today().isoformat()}")

    doc.add_heading("1. Overview", level=1)
    stats = summary["descriptive_stats"]
    doc.add_paragraph(
        f"This report summarizes emotional and visual-attention data from "
        f"{stats['n_trials']} trials across {stats['n_participants']} participants."
    )
    doc.add_paragraph(f"Average engagement score: {stats['avg_engagement_score']}")
    doc.add_paragraph(f"Average attention time: {stats['avg_attention_time']}s")

    doc.add_heading("2. Emotional Response", level=1)
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text = "Emotion", "Share of Trials"
    for emotion, share in stats["emotion_distribution"].items():
        row = table.add_row().cells
        row[0].text, row[1].text = emotion.capitalize(), f"{share*100:.1f}%"

    doc.add_heading("3. Engagement by Creative Variant", level=1)
    if stats["engagement_by_stimulus"]:
        table = doc.add_table(rows=1, cols=2)
        table.style = "Light Grid Accent 1"
        hdr = table.rows[0].cells
        hdr[0].text, hdr[1].text = "Variant", "Avg Engagement Score"
        for variant, score in stats["engagement_by_stimulus"].items():
            row = table.add_row().cells
            row[0].text, row[1].text = variant, str(score)
    else:
        doc.add_paragraph("No completed product-testing sessions yet.")

    doc.add_heading("4. A/B Test Findings", level=1)
    for pair in ab_results.get("engagement_score", []):
        if "note" in pair:
            doc.add_paragraph(
                f"{pair['variant_a']} vs {pair['variant_b']}: {pair['note']}"
            )
            continue
        verdict = (
            f"{pair['likely_better_variant']} performed better "
            f"(p={pair['p_value']})" if pair["significant"]
            else f"No statistically significant difference (p={pair['p_value']})"
        )
        doc.add_paragraph(
            f"{pair['variant_a']} (avg {pair['mean_a']}) vs "
            f"{pair['variant_b']} (avg {pair['mean_b']}): {verdict}"
        )

    doc.add_heading("5. Consumer Segments", level=1)
    seg = summary["segmentation"]
    if "segment_profiles" in seg:
        table = doc.add_table(rows=1, cols=4)
        table.style = "Light Grid Accent 1"
        hdr = table.rows[0].cells
        hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = (
            "Segment", "Avg Emotion Score", "Avg Attention Time", "Avg Engagement Score"
        )
        for seg_id, profile in seg["segment_profiles"].items():
            row = table.add_row().cells
            row[0].text = str(seg_id)
            row[1].text = str(profile["avg_emotion_score"])
            row[2].text = str(profile["avg_attention_time"])
            row[3].text = str(profile["avg_engagement_score"])
    else:
        doc.add_paragraph(seg.get("warning", "Segmentation unavailable."))

    doc.add_heading("6. Recommendations", level=1)
    doc.add_paragraph(
        "Recommendations should be filled in based on the significant A/B "
        "findings above and which creative variant/product feature drove "
        "the strongest positive-emotion + attention combination."
    )

    return doc


def run(client_name="Client"):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary = run_analysis()
    ab_results = run_ab_testing()

    doc = build_report(summary, ab_results, client_name=client_name)
    out_path = os.path.join(REPORTS_DIR, "client_report.docx")
    doc.save(out_path)
    print(f"Client report saved to {out_path}")
    return out_path


if __name__ == "__main__":
    run()