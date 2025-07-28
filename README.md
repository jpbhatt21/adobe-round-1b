# adobe-round-1b
# PDF Q&A Pipeline (Round 1B)

A generic, offline, CPU-only system that ingests PDFs, extracts structured sections, and ranks them semantically against any “persona + job-to-be-done” query. Built atop your robust 1A parser, this solution uses lightweight embeddings, dynamic keyword expansion, and precision-driven filters to surface the most relevant passages.

---

## 🚀 Features

- **1A Parsing**: Merges fragmented text, detects headers by font size/weight, slices into coherent sections.  
- **Dynamic Query Expansion**: KeyBERT + semantic search against a corpus vocabulary for domain-specific jargon.  
- **IDF-Weighted Embedding**: Blends full‐query, rare-phrase and “last‐word” embeddings for laser-focused relevance.  
- **Chunk-Level Ranking**: Cosine‐similarity on paragraph chunks, aggregated per section.  
- **Precision Filters**: Similarity threshold, n-gram overlap, and post-rank summary checks.  
- **JSON Output**: Metadata, top N sections, importance ranks, and concise summaries.

---

## 📦 Requirements

- Python 3.10+  
- PyMuPDF (`fitz`)  
- `sentence-transformers`  
- `keybert`  
- `scikit-learn`  
- `summa` (TextRank)  
- `numpy`

Install all dependencies with:

bash
pip install -r requirements.txt
`

---

## 🔧 Configuration

Edit the top of `app.py`:

python
- MODEL_NAME          = 'sentence-transformers/all-MiniLM-L6-v2'
- TOP_N               = 10
- SIM_THRESHOLD_RATIO = 0.6    # keep sections ≥60% of top score
- NGRAM_N             = 3      # bigrams for phrase filtering
- ALPHA               = 0.3    # last-word boost weight
- BETA                = 0.6    # IDF-weighted keyphrase blend weight


Adjust these to trade off precision vs. recall.

---

## 🏃 Usage

1. **Place** your PDFs under `PDFs/`.

2. **Set** `PERSONA` and `JOB` in `main.py` (or notebook).

3. **Run**:

   bash
   python main.py
   

4. **Result** is written to `app/output/challenge1b_output.json`.

---

## ⚙ Advanced Tuning

* **Threshold** (`SIM_THRESHOLD_RATIO`): higher → more precise, less recall.
* **n-gram size** (`NGRAM_N`): lower → catch shorter phrase fragments.
* **Alpha/Beta**: increase **β** to emphasize rare keyphrases; increase **α** to stress the final term of your JBTD.

Iterate to hit 80–90% precision while preserving recall.

---

## 🗂 Project Structure


project/
├── pdf_parser.py    # 1A parsing logic
├── app.py          # 1B pipeline
├── requirements.txt
└── README.md


---
