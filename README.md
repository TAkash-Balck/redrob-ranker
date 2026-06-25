# Redrob Candidate Ranker

**Hackathon**: Intelligent Candidate Discovery & Ranking Challenge  
**Role**: Senior AI Engineer — Founding Team at Redrob AI  
**Scoring**: NDCG@10 (50%) + NDCG@50 (30%) + MAP (15%) + P@10 (5%)

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full ranking (100K candidates → ~60–90 seconds on 4-core CPU)
python rank.py --candidates candidates.jsonl --out submission.csv

# 3. Validate output format
python -c "
import csv
rows = list(csv.DictReader(open('submission.csv')))
print(f'Rows: {len(rows)} (expected 100)')
scores = [float(r[\"score\"]) for r in rows]
print(f'Score range: {min(scores):.4f} – {max(scores):.4f}')
print(f'Monotonic: {all(scores[i]>=scores[i+1] for i in range(len(scores)-1))}')
"

# 4. Launch demo UI (Streamlit)
streamlit run app.py

# 5. Run tests
pytest tests/ -v
```

---

## 🏗️ Architecture

```
final_score = raw_fit_score × behavioral_multiplier

raw_fit_score = (
    0.30 × skill_depth_score       ← Weighted tier system (FAISS > Docker)
  + 0.25 × career_trajectory_score ← Product companies, consulting penalty
  + 0.20 × product_company_score   ← Swiggy/Uber/Google vs TCS/Infosys
  + 0.15 × experience_band_score   ← 5–9y target band
  + 0.10 × location_score          ← Pune/Noida preferred
)

behavioral_multiplier = recency × availability × responsiveness × notice × github_bonus
```

### Component Details

| Component | Weight | Key Logic |
|-----------|--------|-----------|
| **Skill Depth** | 30% | Tier 1–4 weighted skills. FAISS/embeddings/NDCG = Tier1 (2×). Career description keyword bonuses for "shipped ranking", "vector search", etc. |
| **Career Trajectory** | 25% | Title quality × consulting penalty × job-hopper detection. Consulting firms get 0.4–1.0x multiplier. |
| **Product Company** | 20% | Total months at 100+ known product companies. Full score at 48+ months. TCS/Infosys = 0 contribution. |
| **Experience Band** | 15% | 5–9y = 1.0, <3y = 0.15, >12y = 0.45 |
| **Location** | 10% | Pune/Noida = 1.0, NCR = 0.90, Metro = 0.85 |
| **Behavioral** | ×multiplier | Recency × open_to_work × response_rate × notice_period × github |

---

## 🧠 Design Decisions

### Why keyword matching fails
Naive keyword matching on the `skills` list leads to scoring an HR Manager
with "Python" in their skills above a Recommendation Systems Engineer who
actually shipped FAISS in production. The sample submission (the BAD example)
does exactly this, ranking HR Managers in top 10.

### What we do instead
1. **Tier-weighted skills**: Not all skills are equal. FAISS (Tier1, weight=2.0) 
   is 4× more important than Docker (Tier4, weight=0.5).

2. **Career description text analysis**: We scan job descriptions for production
   signal phrases: "shipped ranking", "vector search", "NDCG evaluation", 
   "real users". A candidate who wrote these things in their descriptions
   actually did the work.

3. **Company history over title**: "Software Engineer who shipped vector search 
   at Swiggy" > "ML Engineer who spent 5 years writing reports at TCS".

4. **Behavioral multiplier**: A perfect-fit candidate inactive for 6 months gets
   a 0.25× recency penalty — making them effectively unranked unless there's
   no better alternative.

5. **Honeypot detection**: 5 checks filter synthetic profiles (timeline 
   impossibility, 9+ expert skills, expert skills with 0 months, ancient 
   tenure dates, title-vs-description domain mismatch).

---

## 📁 Project Structure

```
redrob-ranker/
├── rank.py                  # Main CLI entry point
├── app.py                   # Streamlit demo UI
├── requirements.txt         # All dependencies with versions
├── submission_metadata.yaml # Submission metadata
│
├── src/
│   ├── loader.py            # Fast streaming JSONL loader (orjson + tqdm)
│   ├── honeypot.py          # 5-check honeypot detection
│   ├── filters.py           # Hard domain filters (~85% of dataset)
│   ├── skill_scorer.py      # Tier1–4 weighted skill scoring
│   ├── career_scorer.py     # Career trajectory + consulting penalty
│   ├── product_scorer.py    # Product company detection and scoring
│   ├── behavioral.py        # Behavioral multiplier calculation
│   ├── location_scorer.py   # Location/relocation scoring
│   ├── reasoner.py          # Per-candidate reasoning generator
│   └── ranker.py            # Pipeline orchestration + heapq
│
├── config/
│   └── scoring.yaml         # All weights, thresholds (fully tunable)
│
├── tests/
│   ├── test_honeypot.py     # Honeypot detection unit tests
│   ├── test_scorer.py       # Component scoring unit tests
│   └── test_output_format.py # Output format compliance tests
│
└── notebooks/
    └── analysis.ipynb       # Data exploration notebook
```

---

## ⚙️ Compute Profile

| Resource | Usage |
|----------|-------|
| CPU | All cores via `multiprocessing.Pool` |
| GPU | Not used (CPU only) |
| RAM (peak) | ~2 GB (streaming JSONL, heap buffer=300) |
| Runtime (100K) | ~60–90s on 4-core machine |
| Runtime (100K) | ~30–45s on 8-core machine |
| External APIs | None |
| Network access | None during ranking |

### Performance strategy
- **Streaming JSONL**: File never fully loaded; processed line-by-line
- **Early filter**: ~85% of non-tech candidates rejected before scoring
- **Min-heap**: Only top-300 candidates kept in memory at any time
- **Multiprocessing**: 5K-candidate batches scored in parallel via `Pool.map()`
- **orjson**: 3× faster JSON parsing vs stdlib (automatic fallback)

---

## 🔧 Tuning

All weights and thresholds are in `config/scoring.yaml`. To adjust:

```yaml
# Increase product company weight (e.g., for roles where startup exp matters more)
score_weights:
  product_company: 0.25  # was 0.20
  location: 0.05         # was 0.10

# Add a new tier1 skill
skill_tiers:
  tier1:
    skills:
      - your_new_skill_here
```

No Python changes needed.

---

## 📊 Output Format

```csv
candidate_id,rank,score,reasoning
CAND_0000031,1,1.000000,"6y applied ML/AI engineer from Swiggy & Uber; demonstrated FAISS and sentence-transformers in production; Hyderabad, 30d notice, 90% response rate."
CAND_0000042,2,0.934521,"Senior ML Engineer with 7y exp and strong NLP background; consulting-heavy career (45% at services firms) limits ranking but product company background (Razorpay) is a positive."
...
```

**Format guarantees:**
- Exactly 100 rows
- Ranks 1–100 used exactly once each
- Scores in [0.0, 1.0]
- Scores monotonically non-increasing
- Reasoning: 1–2 sentences, specific facts only

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Single test file
pytest tests/test_scorer.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Expected results:
- `test_honeypot.py`: ~12 tests, all pass
- `test_scorer.py`: ~20 tests, all pass
- `test_output_format.py`: ~15 tests, all pass
