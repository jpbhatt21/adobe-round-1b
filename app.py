#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import datetime
import re
import numpy as np
import fitz  # PyMuPDF

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from summa import summarizer
from keybert import KeyBERT

from pdf_parser import extract_from_pdf_with_merge  # your 1A parsing logic


# ────────────── Configuration ─────────────────────────────────────────────
MODEL_NAME           = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_N                = 10
SIM_THRESHOLD_RATIO  = 0.6    # keep only sections ≥60% of top score
NGRAM_N              = 3      # size of token n-grams for phrase filtering
ALPHA                = 0.3    # last-word boost weight in query blend
BETA                 = 0.6    # IDF-weighted keyphrase blend weight


# ────────────── Vocabulary & Keyword Expansion ──────────────────────────
def build_vocabulary_from_pdfs(pdf_paths):
    """Build a set of all unique words (3+ letters) across the PDF corpus."""
    vocab = set()
    for path in pdf_paths:
        doc = fitz.open(path)
        for page in doc:
            text = page.get_text("text").lower()
            for w in re.findall(r"\b[a-z]{3,}\b", text):
                vocab.add(w)
    return list(vocab)


def expand_keywords(initial_kps, vocab, vocab_embs, model, top_k=5):
    """Use semantic search to expand initial keyphrases from the corpus vocabulary."""
    if not initial_kps or not vocab:
        return []
    init_emb = model.encode(initial_kps, convert_to_tensor=True)
    hits = util.semantic_search(init_emb, vocab_embs, top_k=top_k)
    return list({ vocab[h['corpus_id']] for sub in hits for h in sub })


def compute_idf_weights(keyphrases, pdf_paths):
    """
    Compute a simple IDF (inverse document frequency) for each keyphrase
    across the PDF collection.
    """
    N = len(pdf_paths)
    idf = {}
    for kp in keyphrases:
        df = 0
        for path in pdf_paths:
            full = "".join(pg.get_text("text").lower() for pg in fitz.open(path))
            if kp.lower() in full:
                df += 1
        idf[kp] = np.log((N + 1) / (df + 1)) + 1
    return idf


# ────────────── Query‐Vector Construction & Blending ─────────────────────
def create_query_vector(persona, jbtd, model, kw_model, vocab, vocab_embs, pdf_paths):
    """
    Build a rich query embedding by:
      1. Extracting KeyBERT phrases.
      2. Expanding them via semantic search over the corpus vocabulary.
      3. IDF-weighting those keyphrases.
      4. Blending with the base persona+JBTD embed and a last-word boost.
    """
    # Base text and initial keyphrases
    base = f"{persona}. {jbtd}"
    init_kps = [kp for kp, _ in kw_model.extract_keywords(
        base,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=5
    )]

    # Expand keyphrases from the corpus
    exp_kps = expand_keywords(init_kps, vocab, vocab_embs, model)
    all_kps = init_kps + exp_kps

    # 1) Embed the full concatenated query
    full_text = " ".join([base] + all_kps)
    full_emb = model.encode([full_text], convert_to_tensor=False)

    # 2) Compute IDF-weighted keyphrase average
    idf_wts = compute_idf_weights(all_kps, pdf_paths)
    kp_embs = model.encode(all_kps, convert_to_tensor=False)
    weights = np.array([idf_wts[k] for k in all_kps])
    kp_avg = (weights[:, None] * kp_embs).sum(axis=0) / weights.sum()

    # Blend full query and kp_avg
    mixed = (1 - BETA) * full_emb + BETA * kp_avg

    # 3) Last-word boost
    last_word = jbtd.strip().split()[-1] if jbtd.strip() else ""
    last_emb = model.encode([last_word], convert_to_tensor=False) if last_word else mixed

    # Final query vector
    qv = (1 - ALPHA) * mixed + ALPHA * last_emb
    return qv, all_kps


# ────────────── Section Embedding & Ranking ──────────────────────────────
def chunk_and_embed_content(sections, model):
    """Split each section into paragraphs, embed them, return list of chunks."""
    chunks = []
    for idx, sec in enumerate(sections):
        paras = [p.strip() for p in sec['content'].split("\n\n") if p.strip()]
        if not paras:
            continue
        embs = model.encode(paras, convert_to_tensor=False)
        for emb, txt in zip(embs, paras):
            chunks.append({'section_index': idx, 'emb': emb, 'text': txt})
    return chunks


def rank_sections(qv, chunks, sections):
    """Score and sort sections by their max-chunk similarity to the query vector."""
    if not chunks:
        return []

    emb_matrix = np.stack([c['emb'] for c in chunks])
    sims = cosine_similarity(qv, emb_matrix)[0]

    # Aggregate max chunk score per section
    scores = {i: 0.0 for i in range(len(sections))}
    for score, chunk in zip(sims, chunks):
        idx = chunk['section_index']
        scores[idx] = max(scores[idx], float(score))

    # Attach scores and sort
    ranked = []
    for i, sec in enumerate(sections):
        item = sec.copy()
        item['score'] = scores.get(i, 0.0)
        ranked.append(item)

    return sorted(ranked, key=lambda x: x['score'], reverse=True)


# ────────────── Filtering & Summarization ─────────────────────────────────
def ngram_overlap(text, phrase, n=NGRAM_N):
    """Return True if any n-gram of `phrase` appears in `text`."""
    toks = text.lower().split()
    pks = phrase.lower().split()
    if len(pks) < n:
        return phrase.lower() in text.lower()
    p_ngrams = {tuple(pks[i:i+n]) for i in range(len(pks)-n+1)}
    t_ngrams = {tuple(toks[i:i+n]) for i in range(len(toks)-n+1)}
    return bool(p_ngrams & t_ngrams)


def generate_refined_text(text):
    """Extractive summary via TextRank, fallback to snippet."""
    try:
        summary = summarizer.summarize(text, word_count=150)
        return summary or text[:500] + "..."
    except Exception:
        return text[:500] + "..."


# ────────────── End-to-End Pipeline ────────────────────────────────────────
def process_collection(persona, jbtd, pdf_paths, model, kw_model):
    # 1) Extract sections using 1A logic
    sections = []
    for path in pdf_paths:
        _, secs = extract_from_pdf_with_merge(path)
        for sec in secs:
            sec['document'] = os.path.basename(path)
        sections.extend(secs)

    # 2) Build & embed corpus vocabulary
    vocab = build_vocabulary_from_pdfs(pdf_paths)
    vocab_embs = model.encode(vocab, convert_to_tensor=True)

    # 3) Create query vector
    qv, all_kps = create_query_vector(
        persona, jbtd, model, kw_model, vocab, vocab_embs, pdf_paths
    )

    # 4) Embed content and rank
    chunks = chunk_and_embed_content(sections, model)
    ranked = rank_sections(qv, chunks, sections)

    # 5) Prune by similarity threshold
    if ranked:
        top_score = ranked[0]['score']
        ranked = [s for s in ranked if s['score'] >= SIM_THRESHOLD_RATIO * top_score]

    # 6) Phrase-based filter
    kps_lower = [kp.lower() for kp in all_kps]
    filtered = [
        s for s in ranked
        if any(ngram_overlap(s['title'] + " " + s['content'], kp) for kp in kps_lower)
    ]

    # 7) Post-rank summary check
    final = []
    for sec in filtered:
        summ = generate_refined_text(sec['content'])
        if any(k.lower() in summ.lower() for k in all_kps):
            sec['summary'] = summ
            final.append(sec)
    # Fill up to TOP_N if needed
    for sec in filtered:
        if len(final) >= TOP_N:
            break
        if sec not in final:
            sec['summary'] = generate_refined_text(sec['content'])
            final.append(sec)

    # 8) Assemble output JSON
    output = {
        'metadata': {
            'input_documents': [os.path.basename(p) for p in pdf_paths],
            'persona': persona,
            'job_to_be_done': jbtd,
            'timestamp': datetime.datetime.now().isoformat()
        },
        'extracted_sections': [],
        'subsection_analysis': []
    }

    for rank, sec in enumerate(final[:TOP_N], start=1):
        output['extracted_sections'].append({
            'document': sec['document'],
            'page_number': sec['page'],
            'section_title': sec['title'],
            'importance_rank': rank
        })
        output['subsection_analysis'].append({
            'document': sec['document'],
            'page_number': sec['page'],
            'refined_text': sec.get('summary', '')
        })

    return output


# ────────────── Script Entry Point ────────────────────────────────────────
if __name__ == '__main__':
    INPUT_DIR = 'app/PDFs'
    OUTPUT_FP = 'app/output/challenge1b_output.json'
    PERSONA   = 'Travel Planner'
    JOB       = 'Plan a trip of 4 days for a group of 10 college friends.'

    # Gather all PDF file paths
    pdf_files = [
        os.path.join(INPUT_DIR, fname)
        for fname in os.listdir(INPUT_DIR)
        if fname.lower().endswith('.pdf')
    ]

    # Load models
    model = SentenceTransformer(MODEL_NAME)
    kw_model = KeyBERT(model=model)

    # Process and write results
    result = process_collection(PERSONA, JOB, pdf_files, model, kw_model)
    with open(OUTPUT_FP, 'w', encoding='utf-8') as fout:
        json.dump(result, fout, indent=2, ensure_ascii=False)

    print(f"✅ Output written to {OUTPUT_FP}")
