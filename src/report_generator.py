"""
report_generator.py
---------------------
Builds/regenerates a detailed, professional Word report
(reports/client_report.docx) from the full snapshot history produced by
dataset_pipeline.py (which analyzes data/unified_dataset_clean.csv directly).

Call generate_report() once after every capture_snapshot_from_dataset()
call (or simply after the dashboard has run an analysis), and the .docx
is fully rebuilt: latest numbers, embedded charts, and a running
revision/change-log so every past analysis run stays visible.

Drop into: Neuromarketing_project/src/report_generator.py

Usage (typically at the end of main.py, right after your dataset updates):

    from dataset_pipeline import capture_snapshot_from_dataset
    from report_generator import generate_report

    capture_snapshot_from_dataset()
    generate_report()
"""

import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from dataset_pipeline import load_snapshot_history

REPORT_PATH = os.path.join("reports", "client_report.docx")
HISTORY_PATH = os.path.join("reports", "dashboard_history.jsonl")
CHART_DIR = os.path.join("reports", "_report_charts")

NAVY = RGBColor(0x1F, 0x2A, 0x44)
ACCENT_HEX = "1F2A44"
GOLD = RGBColor(0xC0, 0x8A, 0x2E)
GREY = RGBColor(0x5A, 0x5A, 0x5A)

CHART_PALETTE = ["#1F2A44", "#C08A2E", "#4C7A8C", "#8C4C4C", "#5A8C4C", "#7A5A8C"]


# =======================================================================
# LOW-LEVEL DOCX HELPERS
# =======================================================================
def _shade_cell(cell, hex_color):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(shd)


def _header_row(table, headers):
    row = table.rows[0]
    for i, text in enumerate(headers):
        cell = row.cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _shade_cell(cell, ACCENT_HEX)


def _add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    _header_row(table, headers)
    for r in rows:
        cells = table.add_row().cells
        for i, val in enumerate(r):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    return table


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _write_markdown_runs(p, text, size=None):
    """Writes **bold** marked text as actual bold runs into an EXISTING paragraph."""
    pos = 0
    for m in _BOLD_RE.finditer(text):
        if m.start() > pos:
            run = p.add_run(text[pos:m.start()])
            if size:
                run.font.size = size
        run = p.add_run(m.group(1))
        run.bold = True
        if size:
            run.font.size = size
        pos = m.end()
    if pos < len(text):
        run = p.add_run(text[pos:])
        if size:
            run.font.size = size
    return p


def _add_markdown_paragraph(doc, text, style=None, size=None):
    """Creates a NEW paragraph on `doc` and renders **bold** markers as bold runs."""
    p = doc.add_paragraph(style=style)
    return _write_markdown_runs(p, text, size=size)


def _bullet(doc, text):
    _add_markdown_paragraph(doc, text, style="List Bullet")


def _add_page_number_footer(doc):
    section = doc.sections[0]
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


def _add_toc(doc):
    doc.add_heading("Table of Contents", level=1)
    p = doc.add_paragraph()
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-2" \\h \\z \\u'
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_text = OxmlElement("w:t")
    fld_text.text = "Right-click and choose 'Update Field' to populate the table of contents."
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_text)
    run._r.append(fld_end)
    doc.add_page_break()


def _add_highlight_box(doc, text):
    """A single-cell shaded 'callout box' table, used for the executive summary."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.rows[0].cells[0]
    _shade_cell(cell, "F0EDE4")
    cell.paragraphs[0].text = ""
    _write_markdown_runs(cell.paragraphs[0], text)
    for p in cell.paragraphs:
        for run in p.runs:
            run.font.size = Pt(11)
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "12")
        el.set(qn("w:color"), ACCENT_HEX)
        borders.append(el)
    tblPr.append(borders)


# =======================================================================
# CHART GENERATION (matplotlib -> PNG -> embedded in docx)
# =======================================================================
def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _chart_emotion_pie(emotion_distribution, out_path):
    labels = [k.title() for k in emotion_distribution.keys()]
    values = list(emotion_distribution.values())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=CHART_PALETTE, startangle=90,
           textprops={"fontsize": 9})
    ax.set_title("Emotion Distribution", fontsize=12, color=f"#{ACCENT_HEX}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _chart_stimulus_bars(engagement_by_stimulus, attention_by_stimulus, out_path):
    stimuli = list(engagement_by_stimulus.keys())
    eng_vals = [engagement_by_stimulus[s] for s in stimuli]
    att_vals = [attention_by_stimulus.get(s, 0) for s in stimuli]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    axes[0].bar(stimuli, eng_vals, color=CHART_PALETTE[:len(stimuli)])
    axes[0].set_title("Avg Engagement by Stimulus", fontsize=11, color=f"#{ACCENT_HEX}")
    _style_ax(axes[0])

    axes[1].bar(stimuli, att_vals, color=CHART_PALETTE[:len(stimuli)])
    axes[1].set_title("Avg Attention Time (s) by Stimulus", fontsize=11, color=f"#{ACCENT_HEX}")
    _style_ax(axes[1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _chart_segment_pie(segment_sizes, out_path):
    labels = [f"Segment {k}" for k in segment_sizes.keys()]
    values = list(segment_sizes.values())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(values, labels=labels, autopct="%1.0f%%", colors=CHART_PALETTE, startangle=90,
           textprops={"fontsize": 9})
    ax.set_title("Participant Segment Sizes", fontsize=12, color=f"#{ACCENT_HEX}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _chart_trend(history, out_path):
    timestamps = [datetime.fromisoformat(s["timestamp"]) for s in history]
    trials = [s.get("n_trials") for s in history]
    engagement = [s.get("avg_engagement_score") for s in history]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    axes[0].plot(timestamps, trials, marker="o", color=CHART_PALETTE[0])
    axes[0].set_title("Dataset Growth (Total Trials)", fontsize=11, color=f"#{ACCENT_HEX}")
    _style_ax(axes[0])
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].plot(timestamps, engagement, marker="o", color=CHART_PALETTE[1])
    axes[1].set_title("Avg Engagement Score Trend", fontsize=11, color=f"#{ACCENT_HEX}")
    _style_ax(axes[1])
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =======================================================================
# RECOMMENDATION ENGINE (rule-based, derived only from real numbers)
# =======================================================================
def _generate_recommendations(latest: dict) -> list:
    recs = []

    eng_by_stim = latest.get("engagement_by_stimulus", {})
    if eng_by_stim:
        best = max(eng_by_stim, key=eng_by_stim.get)
        worst = min(eng_by_stim, key=eng_by_stim.get)
        recs.append(
            f"**Prioritize {best}** — it currently shows the highest engagement score "
            f"({eng_by_stim[best]:.2f}), outperforming {worst} ({eng_by_stim[worst]:.2f})."
        )

    emo_dist = latest.get("emotion_distribution", {})
    if emo_dist:
        negative_emotions = {"sad", "angry", "disgust", "fear"}
        neg_share = sum(v for k, v in emo_dist.items() if k in negative_emotions)
        if neg_share >= 0.4:
            recs.append(
                f"**Review the stimulus design** — negative emotions (sad/angry/disgust/fear) "
                f"account for {neg_share*100:.1f}% of reactions, which may indicate confusion "
                f"or discomfort worth investigating."
            )

    ab = latest.get("ab_test_results", {})
    insufficient_flags, significant_flags = [], []
    for metric_name, comparisons in ab.items():
        for comp in comparisons:
            if comp.get("significant"):
                significant_flags.append(f"{metric_name.replace('_',' ')}: {comp.get('likely_better_variant')} (p={comp.get('p_value')})")
            elif comp.get("note", "").startswith("insufficient"):
                insufficient_flags.append(f"{comp['variant_a']} vs {comp['variant_b']}")

    if significant_flags:
        recs.append("**Act on statistically confirmed results:** " + "; ".join(significant_flags) + ".")
    if insufficient_flags:
        pairs = ", ".join(sorted(set(insufficient_flags)))
        recs.append(
            f"**Collect more data before concluding a winner** for these comparisons "
            f"(currently below the minimum sample threshold): {pairs}."
        )

    seg_profiles = latest.get("segment_profiles", {})
    if seg_profiles:
        top_seg = max(seg_profiles, key=lambda k: seg_profiles[k].get("avg_engagement_score", 0))
        recs.append(
            f"**Tailor messaging toward Segment {top_seg}** — it shows the highest average "
            f"engagement ({seg_profiles[top_seg].get('avg_engagement_score', 0):.2f}) of all "
            f"identified participant clusters."
        )

    if not recs:
        recs.append("Not enough data yet to generate confident recommendations.")
    return recs


def _confidence_notes(latest: dict) -> list:
    """Statistical caveats / limitations, stated plainly for a non-technical reader."""
    notes = []
    ab = latest.get("ab_test_results", {})
    small_n_pairs = set()
    for comparisons in ab.values():
        for comp in comparisons:
            n_a, n_b = comp.get("n_a", 0), comp.get("n_b", 0)
            if n_a < 30 or n_b < 30:
                small_n_pairs.add(f"{comp['variant_a']} vs {comp['variant_b']} (n={n_a} / n={n_b})")
    if small_n_pairs:
        notes.append(
            "Several stimulus comparisons currently have small sample sizes "
            f"({'; '.join(sorted(small_n_pairs))}). Findings from these comparisons should be "
            "treated as directional, not conclusive, until more participants are tested."
        )
    notes.append(
        "Statistical significance is assessed at the standard p < 0.05 threshold using "
        "independent two-sample t-tests (Welch's, unequal variance assumed)."
    )
    return notes


# =======================================================================
# MAIN REPORT BUILDER
# =======================================================================
def generate_report(history_path: str = HISTORY_PATH, output_path: str = REPORT_PATH,
                     product_name: str = "Neuromarketing Study — Product Packaging Test",
                     prepared_for: str = "Client Stakeholders"):
    history = load_snapshot_history(history_path)
    if not history:
        raise ValueError("No snapshots found. Run capture_snapshot_from_dataset() at least once first.")

    latest = history[-1]
    os.makedirs(CHART_DIR, exist_ok=True)

    # ---- generate charts up front ----
    emo_chart = os.path.join(CHART_DIR, "emotion.png")
    stim_chart = os.path.join(CHART_DIR, "stimulus.png")
    seg_chart = os.path.join(CHART_DIR, "segments.png")
    trend_chart = os.path.join(CHART_DIR, "trend.png")

    if latest.get("emotion_distribution"):
        _chart_emotion_pie(latest["emotion_distribution"], emo_chart)
    if latest.get("engagement_by_stimulus"):
        _chart_stimulus_bars(latest["engagement_by_stimulus"], latest.get("attention_by_stimulus", {}), stim_chart)
    if latest.get("segment_sizes"):
        _chart_segment_pie(latest["segment_sizes"], seg_chart)
    has_trend = len(history) >= 2
    if has_trend:
        _chart_trend(history, trend_chart)

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    # ---------- COVER PAGE ----------
    for _ in range(4):
        doc.add_paragraph()
    title = doc.add_heading(product_name, level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.color.rgb = NAVY

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("Neuromarketing Analysis Report")
    run.font.size = Pt(16)
    run.font.color.rgb = GOLD
    run.italic = True

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_run = meta.add_run(
        f"Prepared for: {prepared_for}\n"
        f"Report generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}\n"
        f"Revision {len(history)} of {len(history)}  ·  Based on {latest.get('n_trials', '—')} trials "
        f"across {latest.get('n_participants', '—')} participants"
    )
    meta_run.font.size = Pt(11)
    meta_run.font.color.rgb = GREY

    doc.add_page_break()

    # ---------- TABLE OF CONTENTS ----------
    _add_toc(doc)

    # ---------- DOCUMENT CONTROL ----------
    doc.add_heading("Document Control & Revision History", level=1)
    doc.add_paragraph(
        "This report is automatically regenerated after every analysis run against the live "
        "dataset. It is cumulative: nothing from a prior run is lost — the table below lists "
        "every analysis run recorded to date, and the Appendix contains the full narrative "
        "history."
    )
    rev_rows = []
    for i, snap in enumerate(history, start=1):
        ts = datetime.fromisoformat(snap["timestamp"]).strftime("%Y-%m-%d %H:%M")
        rev_rows.append([
            i, ts,
            snap.get("n_trials", "—"),
            snap.get("n_participants", "—"),
            f"{snap.get('avg_engagement_score', 0):.2f}" if snap.get("avg_engagement_score") is not None else "—",
        ])
    _add_table(doc, ["Rev #", "Date / Time", "Total Trials", "Participants", "Avg Engagement"], rev_rows,
               widths=[0.6, 1.7, 1.2, 1.2, 1.4])
    doc.add_page_break()

    # ---------- 1. EXECUTIVE SUMMARY ----------
    doc.add_heading("1. Executive Summary", level=1)
    _add_highlight_box(doc, latest.get("explanation", "No explanation available."))
    doc.add_paragraph()
    doc.add_paragraph(
        "This section is regenerated automatically after every analysis and always reflects "
        "the most recently collected data. See Section 8 (Appendix) for the complete "
        "run-by-run history of how these findings have evolved."
    )

    # ---------- 2. OBJECTIVES & SCOPE ----------
    doc.add_heading("2. Objectives & Scope", level=1)
    doc.add_paragraph(
        "This study evaluates customer response to a single product's design variants using "
        "non-intrusive eye-tracking (gaze and fixation analysis) and facial emotion "
        "recognition. The objective is to identify which visual design (stimulus variant) "
        "drives the strongest attention and the most positive emotional engagement, in order "
        "to inform the final packaging and design decision with quantitative, behavioral "
        "evidence rather than opinion alone."
    )
    doc.add_paragraph(
        "The scope covers all participant sessions recorded to date in the underlying "
        "dataset, and is updated automatically as new sessions are added — this report always "
        "reflects the complete, current body of evidence."
    )

    # ---------- 3. METHODOLOGY ----------
    doc.add_heading("3. Methodology", level=1)
    doc.add_paragraph("Data was collected and analyzed using the following pipeline:")
    _bullet(doc, "**Eye tracking:** gaze position (x/y) and fixation duration captured via OpenCV + MediaPipe face/eye landmark detection.")
    _bullet(doc, "**Facial emotion recognition:** frame-level emotion classification via DeepFace, producing one of five categories per trial — happy, sad, angry, surprise, or neutral.")
    _bullet(doc, "**Engagement score:** a composite metric combining attention time and emotion valence, capturing both how long and how positively a participant engaged with the stimulus.")
    _bullet(doc, "**A/B(/C) testing:** independent two-sample t-tests (Welch's) comparing engagement, emotion, and attention-time means across stimulus variants (A, B, C).")
    _bullet(doc, "**Segmentation:** K-means clustering (k=3) of all participants based on emotion score, attention time, and engagement score, to identify distinct behavioral profiles.")
    doc.add_paragraph(
        "All figures in this report are computed directly from the underlying dataset at "
        "generation time — no manual data entry is involved."
    )

    # ---------- 4. KEY METRICS SNAPSHOT ----------
    doc.add_heading("4. Key Metrics Snapshot (Current)", level=1)
    kpi_rows = [
        ["Total Trials", f"{latest.get('n_trials', '—'):,}" if isinstance(latest.get("n_trials"), int) else "—"],
        ["Total Participants", f"{latest.get('n_participants', '—'):,}" if isinstance(latest.get("n_participants"), int) else "—"],
        ["Avg Engagement Score", f"{latest.get('avg_engagement_score', 0):.2f}" if latest.get("avg_engagement_score") is not None else "—"],
        ["Avg Attention Time (s)", f"{latest.get('avg_attention_time', 0):.2f}" if latest.get("avg_attention_time") is not None else "—"],
        ["Analysis Runs Logged", len(history)],
    ]
    _add_table(doc, ["Metric", "Value"], kpi_rows, widths=[3.5, 2.5])

    if has_trend:
        doc.add_paragraph()
        doc.add_heading("Trend Across Analysis Runs", level=2)
        doc.add_picture(trend_chart, width=Inches(6.3))
    doc.add_page_break()

    # ---------- 5. FINDINGS ----------
    doc.add_heading("5. Findings", level=1)

    doc.add_heading("5.1 Emotion Distribution", level=2)
    emo_dist = latest.get("emotion_distribution", {})
    if emo_dist:
        top_emotion = max(emo_dist, key=emo_dist.get)
        _add_markdown_paragraph(
            doc,
            f"Across all recorded trials, **{top_emotion.title()}** was the most frequently "
            f"detected emotion, accounting for {emo_dist[top_emotion]*100:.1f}% of reactions."
        )
        doc.add_picture(emo_chart, width=Inches(4.2))
        emo_rows = [[k.title(), f"{v*100:.1f}%"] for k, v in sorted(emo_dist.items(), key=lambda x: -x[1])]
        _add_table(doc, ["Emotion", "Share of Trials"], emo_rows, widths=[3.5, 2.5])

    doc.add_heading("5.2 Engagement & Attention by Stimulus", level=2)
    eng = latest.get("engagement_by_stimulus", {})
    att = latest.get("attention_by_stimulus", {})
    if eng:
        best = max(eng, key=eng.get)
        worst = min(eng, key=eng.get)
        _add_markdown_paragraph(
            doc,
            f"**{best}** currently leads in average engagement ({eng[best]:.2f}), while "
            f"**{worst}** trails ({eng[worst]:.2f}). This comparison is based on the subset "
            f"of trials where a stimulus variant was explicitly assigned."
        )
        doc.add_picture(stim_chart, width=Inches(6.3))
        stim_rows = [[s, f"{eng.get(s, 0):.2f}", f"{att.get(s, 0):.2f}"] for s in eng.keys()]
        _add_table(doc, ["Stimulus", "Avg Engagement", "Avg Attention Time (s)"], stim_rows,
                   widths=[2.3, 2.3, 2.4])
    else:
        doc.add_paragraph("No stimulus-variant data recorded yet.")

    doc.add_page_break()
    doc.add_heading("5.3 Participant Segmentation", level=2)
    seg_profiles = latest.get("segment_profiles", {})
    seg_sizes = latest.get("segment_sizes", {})
    if seg_profiles:
        total_participants = sum(int(v) for v in seg_sizes.values())
        best_seg = max(seg_profiles, key=lambda k: seg_profiles[k].get("avg_engagement_score", 0))
        worst_seg = min(seg_profiles, key=lambda k: seg_profiles[k].get("avg_engagement_score", 0))
        best_pct = (int(seg_sizes.get(best_seg, 0)) / total_participants * 100) if total_participants else 0
        _add_markdown_paragraph(
            doc,
            f"Three behavioral segments were identified via clustering. Segment **{best_seg}** "
            f"shows the highest engagement ({seg_profiles[best_seg]['avg_engagement_score']:.2f}) "
            f"and represents {best_pct:.0f}% of all trials — the strongest-responding audience "
            f"profile. Segment **{worst_seg}** shows the lowest engagement "
            f"({seg_profiles[worst_seg]['avg_engagement_score']:.2f}) and may warrant a "
            f"different creative or messaging approach."
        )
        doc.add_picture(seg_chart, width=Inches(4.2))
        seg_rows = [
            [seg_id, seg_sizes.get(seg_id, "—"),
             f"{p.get('avg_emotion_score', 0):.2f}", f"{p.get('avg_attention_time', 0):.2f}",
             f"{p.get('avg_engagement_score', 0):.2f}"]
            for seg_id, p in seg_profiles.items()
        ]
        _add_table(doc, ["Segment", "Size", "Avg Emotion", "Avg Attention", "Avg Engagement"], seg_rows,
                   widths=[1.2, 1.0, 1.3, 1.3, 1.4])

    doc.add_page_break()

    # ---------- 6. A/B TEST RESULTS ----------
    doc.add_heading("6. A/B Test Results", level=1)
    doc.add_paragraph(
        "Each stimulus pair was compared using an independent two-sample t-test. A result is "
        "marked significant only when p < 0.05; comparisons with fewer than 3 samples per "
        "variant are marked as not tested rather than \"no difference\", since significance "
        "cannot be meaningfully assessed on that little data."
    )
    ab = latest.get("ab_test_results", {})
    for metric_name, comparisons in ab.items():
        doc.add_heading(metric_name.replace("_", " ").title(), level=2)
        rows = []
        for comp in comparisons:
            has_p = "p_value" in comp
            sig = ("Yes" if comp.get("significant") else "No") if has_p else "Not tested"
            rows.append([
                comp.get("variant_a"), comp.get("variant_b"),
                comp.get("mean_a", "—"), comp.get("mean_b", "—"),
                comp.get("p_value", "—"),
                sig,
                comp.get("likely_better_variant") or comp.get("note", ""),
            ])
        _add_table(doc, ["Variant A", "Variant B", "Mean A", "Mean B", "p-value", "Sig.?", "Result / Note"], rows,
                   widths=[0.9, 0.9, 0.7, 0.7, 0.7, 0.8, 2.3])
        doc.add_paragraph()

    doc.add_heading("Confidence & Limitations", level=2)
    for note in _confidence_notes(latest):
        _bullet(doc, note)

    doc.add_page_break()

    # ---------- 7. RECOMMENDATIONS ----------
    doc.add_heading("7. Recommendations", level=1)
    doc.add_paragraph("Based on the current data, the following actions are recommended:")
    for rec in _generate_recommendations(latest):
        _bullet(doc, rec)

    doc.add_page_break()

    # ---------- 8. APPENDIX ----------
    doc.add_heading("8. Appendix: Full Analysis Change Log", level=1)
    doc.add_paragraph(
        "Every analysis run ever recorded is listed below, oldest first, so the evolution of "
        "findings over time remains fully auditable."
    )
    for i, snap in enumerate(history, start=1):
        ts = datetime.fromisoformat(snap["timestamp"]).strftime("%Y-%m-%d %H:%M")
        doc.add_heading(f"Run {i} — {ts}  ({snap.get('n_trials', '—')} trials)", level=3)
        _add_markdown_paragraph(doc, snap.get("explanation", "—"))

    _add_page_number_footer(doc)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    return output_path


if __name__ == "__main__":
    path = generate_report()
    print(f"Report generated: {path}")