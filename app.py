"""
app.py — Streamlit demo UI for the Redrob Candidate Ranker.

Supports:
  - JSON file upload (up to 100 sample candidates)
  - One-click "Run Ranker" button
  - Color-coded ranked table (green=1–10, yellow=11–30, white=rest)
  - Score breakdown bars per candidate
  - Download CSV button
  - Execution time + stats
"""

from __future__ import annotations

import json
import time
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #a8b2d8;
        margin-top: 0.5rem;
    }

    .stat-card {
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .stat-card h3 {
        color: #e94560;
        font-size: 1.8rem;
        margin: 0;
    }
    .stat-card p {
        color: #a8b2d8;
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
    }

    .rank-badge-gold { color: #FFD700; font-weight: 700; }
    .rank-badge-silver { color: #C0C0C0; font-weight: 700; }
    .rank-badge-bronze { color: #CD7F32; font-weight: 700; }

    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c62a47);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.4);
    }

    .score-label {
        font-size: 0.75rem;
        color: #a8b2d8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="main-header">
        <h1>🎯 Redrob Candidate Ranker</h1>
        <p>Intelligent Candidate Discovery & Ranking · Senior AI Engineer Role</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Ranker Settings")
    top_k = st.slider("Top K candidates", min_value=10, max_value=100, value=100, step=10)
    show_breakdown = st.checkbox("Show score breakdown", value=True)
    st.divider()
    st.markdown("### 📋 JD Summary")
    st.markdown("""
    **Role**: Senior AI Engineer  
    **Location**: Pune/Noida preferred  
    **Experience**: 5–9 years  
    **Must have**:
    - Production embeddings/vector search
    - FAISS/Pinecone/Qdrant/Elasticsearch
    - Ranking evaluation (NDCG, MRR, MAP)
    - Strong Python
    
    **Disqualifiers**:
    - TCS/Infosys/Wipro/Accenture careers
    - No production deployment
    - Pure CV/robotics background
    """)
    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "CPU-only · No API calls · <5min runtime\n\n"
        "Scoring: Skills (30%) + Career (25%) + Product Co (20%) + "
        "Experience (15%) + Location (10%) × Behavioral Multiplier"
    )

# ─── File Upload ──────────────────────────────────────────────────────────────

st.markdown("### 📂 Upload Candidate Data")

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload candidates JSON/JSONL file (up to 100 candidates for demo)",
        type=["json", "jsonl"],
        help="Upload a JSON array or JSONL file with candidate objects.",
    )

with col_info:
    st.info(
        "**Supported formats:**\n"
        "- JSON array: `[{...}, {...}]`\n"
        "- JSONL: one candidate per line\n\n"
        "For the full 100K dataset, use the CLI:\n"
        "`python rank.py --candidates candidates.jsonl --out submission.csv`"
    )

# ─── Load and Parse ───────────────────────────────────────────────────────────


def parse_upload(file_bytes: bytes) -> list[dict]:
    """Parse uploaded file into list of candidate dicts.

    Supports JSON array or JSONL format.

    Args:
        file_bytes: Raw bytes from uploaded file.

    Returns:
        List of candidate dicts.
    """
    text = file_bytes.decode("utf-8", errors="replace")

    # Try JSON array first
    stripped = text.strip()
    if stripped.startswith("["):
        return json.loads(stripped)

    # Try JSONL
    candidates = []
    for line in stripped.splitlines():
        line = line.strip()
        if line:
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return candidates


# ─── Run Ranker ───────────────────────────────────────────────────────────────

if uploaded_file is not None:
    try:
        raw_bytes = uploaded_file.read()
        candidates = parse_upload(raw_bytes)
        st.success(f"✅ Loaded **{len(candidates):,}** candidates from `{uploaded_file.name}`")
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        candidates = []

    if candidates:
        st.divider()
        col_btn, col_note = st.columns([1, 3])
        with col_btn:
            run_btn = st.button("🚀 Run Ranker", use_container_width=True)
        with col_note:
            st.caption(
                f"Will rank {min(len(candidates), top_k)} candidates from {len(candidates)} uploaded. "
                "Processing runs in-memory (no GPU/API needed)."
            )

        if run_btn:
            with st.spinner("Running candidate scoring pipeline..."):
                t0 = time.perf_counter()

                # Import here to avoid slow startup
                from src.ranker import rank_candidates_from_list

                results = rank_candidates_from_list(candidates, top_k=top_k)
                elapsed = time.perf_counter() - t0

            # ── Stats row ─────────────────────────────────────────────────
            st.divider()
            st.markdown("### 📊 Results Summary")

            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

            with stat_col1:
                st.markdown(
                    f'<div class="stat-card"><h3>{len(results)}</h3>'
                    f"<p>Candidates Ranked</p></div>",
                    unsafe_allow_html=True,
                )
            with stat_col2:
                st.markdown(
                    f'<div class="stat-card"><h3>{elapsed:.1f}s</h3>'
                    f"<p>Processing Time</p></div>",
                    unsafe_allow_html=True,
                )
            with stat_col3:
                if results:
                    top_score = results[0]["score"]
                    st.markdown(
                        f'<div class="stat-card"><h3>{top_score:.3f}</h3>'
                        f"<p>Top Score</p></div>",
                        unsafe_allow_html=True,
                    )
            with stat_col4:
                if results:
                    avg_score = sum(r["score"] for r in results) / len(results)
                    st.markdown(
                        f'<div class="stat-card"><h3>{avg_score:.3f}</h3>'
                        f"<p>Avg Score (Top {len(results)})</p></div>",
                        unsafe_allow_html=True,
                    )

            # ── Build display dataframe ───────────────────────────────────
            st.divider()
            st.markdown("### 🏆 Ranked Candidates")

            table_rows = []
            for r in results:
                cand = r.get("candidate", {}) or {}
                prof = cand.get("profile", {}) or {}
                signals = cand.get("redrob_signals", {}) or {}

                table_rows.append({
                    "Rank": r["rank"],
                    "Candidate ID": r["candidate_id"],
                    "Title": str(prof.get("current_title") or "—")[:40],
                    "Company": str(prof.get("current_company") or "—")[:30],
                    "YoE": f"{float(prof.get('years_of_experience') or 0):.1f}",
                    "Location": str(prof.get("location") or "—")[:25],
                    "Score": f"{r['score']:.4f}",
                    "Skill": f"{r['scores'].get('skill_depth', 0):.2f}",
                    "Career": f"{r['scores'].get('career_trajectory', 0):.2f}",
                    "Product": f"{r['scores'].get('product_company', 0):.2f}",
                    "Beh.": f"{r['behavioral_multiplier']:.2f}",
                    "Reasoning": r["reasoning"],
                })

            df = pd.DataFrame(table_rows)

            # Color coding function
            def highlight_rows(row: pd.Series) -> list[str]:
                rank = int(row["Rank"])
                if rank <= 10:
                    bg = "background-color: #0d3b1e; color: #6ee7b7"
                elif rank <= 30:
                    bg = "background-color: #3b2d00; color: #fcd34d"
                else:
                    bg = ""
                return [bg] * len(row)

            styled_df = df.style.apply(highlight_rows, axis=1)

            st.dataframe(
                styled_df,
                use_container_width=True,
                height=600,
                hide_index=True,
            )

            # ── Score breakdown visualization ─────────────────────────────
            if show_breakdown and results:
                st.divider()
                st.markdown("### 📈 Score Breakdown — Top 20")

                top20 = results[:20]
                categories = ["skill_depth", "career_trajectory", "product_company", "experience_band", "location"]
                labels = ["Skill Depth", "Career", "Product Co", "Experience", "Location"]
                colors = ["#e94560", "#0f3460", "#533483", "#05c46b", "#f9ca24"]

                fig = go.Figure()

                for cat, label, color in zip(categories, labels, colors):
                    vals = [r["scores"].get(cat, 0) for r in top20]
                    ids = [f"#{r['rank']} {r['candidate_id']}" for r in top20]
                    fig.add_trace(go.Bar(
                        name=label,
                        x=ids,
                        y=vals,
                        marker_color=color,
                    ))

                fig.update_layout(
                    barmode="stack",
                    template="plotly_dark",
                    xaxis_title="Candidate",
                    yaxis_title="Score Component",
                    legend_title="Component",
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=80),
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Behavioral multiplier chart
                st.markdown("#### ⚡ Behavioral Multiplier — Top 20")
                beh_vals = [r["behavioral_multiplier"] for r in top20]
                beh_ids = [f"#{r['rank']}" for r in top20]

                fig2 = go.Figure(go.Bar(
                    x=beh_ids,
                    y=beh_vals,
                    marker=dict(
                        color=beh_vals,
                        colorscale="RdYlGn",
                        cmin=0.3,
                        cmax=1.0,
                        showscale=True,
                    ),
                ))
                fig2.update_layout(
                    template="plotly_dark",
                    xaxis_title="Rank",
                    yaxis_title="Behavioral Multiplier",
                    height=300,
                    margin=dict(l=0, r=0, t=20, b=40),
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ── Download CSV ──────────────────────────────────────────────
            st.divider()
            st.markdown("### 💾 Download Submission")

            csv_buffer = StringIO()
            import csv
            writer = csv.DictWriter(
                csv_buffer,
                fieldnames=["candidate_id", "rank", "score", "reasoning"],
            )
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "candidate_id": r["candidate_id"],
                    "rank": r["rank"],
                    "score": f"{r['score']:.6f}",
                    "reasoning": r["reasoning"],
                })
            csv_str = csv_buffer.getvalue()

            col_dl, col_preview = st.columns([1, 2])
            with col_dl:
                st.download_button(
                    label="⬇️ Download submission.csv",
                    data=csv_str,
                    file_name="submission.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption(f"File: {len(results)} rows · {len(csv_str):,} bytes")

            with col_preview:
                st.markdown("**Preview (first 5 rows):**")
                preview_df = pd.DataFrame(table_rows[:5])[
                    ["Rank", "Candidate ID", "Score", "Reasoning"]
                ]
                st.dataframe(preview_df, use_container_width=True, hide_index=True)

else:
    # ── No file uploaded — show instructions ─────────────────────────────────
    st.divider()
    st.markdown(
        """
        ### 🚀 Getting Started

        1. **Upload** a candidates JSON or JSONL file using the uploader above
        2. Click **Run Ranker** to score and rank candidates
        3. Review the **ranked table** with color-coded tiers
        4. **Download** the submission CSV

        ---

        ### 🔧 CLI Usage (for full 100K dataset)

        ```bash
        # Install dependencies
        pip install -r requirements.txt

        # Run full ranking (100K candidates, ~60–90 seconds)
        python rank.py --candidates candidates.jsonl --out submission.csv

        # Quick test with 5000 candidates
        python rank.py --candidates candidates.jsonl --out test.csv --max 5000
        ```

        ---

        ### 🧠 Scoring Architecture

        | Component | Weight | What it measures |
        |-----------|--------|-----------------|
        | Skill Depth | **30%** | Tier1–4 weighted skills + career description signals |
        | Career Trajectory | **25%** | Title quality + consulting penalty + job-hopper detection |
        | Product Company | **20%** | % career at product/tech companies |
        | Experience Band | **15%** | 5–9 year target range |
        | Location | **10%** | Pune/Noida preferred → relocation bonus |
        | × Behavioral | **multiplier** | Recency × Availability × Responsiveness × Notice |
        """,
        unsafe_allow_html=False,
    )

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "Redrob Candidate Ranker · CPU-only · No external APIs · "
    "NDCG@10 optimized · Built for the Redrob Hackathon 2026"
    "</p>",
    unsafe_allow_html=True,
)
