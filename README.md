# adobe-round-1b
The Round 1B pipeline builds directly on 1A parser to deliver a fully generic, persona-driven “PDF Q\&A” system. Here’s how each stage works:

1. *Accurate PDF Parsing (1A Module)*
   We first reuse your existing extract_from_pdf_with_merge() function—packaged in pdf_parser.py—to extract clean, merged text and logical section boundaries from any PDF. This combines low-level span extraction with font-based header detection and a maximal‐overlap merging algorithm, ensuring fragmented lines become coherent blocks and headings are recognized by size and weight .

2. *Corpus Vocabulary & Dynamic Keyword Expansion*
   Once every PDF is sliced into titled sections, we scan the entire collection to build a corpus vocabulary of all multi-letter words. KeyBERT then pulls the top 5 phrases from the user’s “persona + job-to-be-done” (JBTD) description, and we semantically expand these via a lightweight SentenceTransformer search against that vocabulary. This allows the system to learn domain-specific jargon directly from the documents, without any hard-coding .

3. *IDF-Weighted Query Embedding with Last-Word Boost*
   To form our query vector, we embed three components:

   * The full concatenated string: (persona + JBTD + all keyphrases)
   * An IDF-weighted average of each keyphrase embedding, which up-weights rare, discriminative terms
   * The embedding of the final word of the JBTD, emphasizing the user’s ultimate focus

   We blend these using configurable weights (β for IDF-keyphrase mix, α for last-word boost), producing a highly targeted vector that reflects both broad context and critical specifics .

4. *Section Ranking by Chunk Similarity*
   Each section is split into paragraph “chunks” and embedded. We compute cosine similarity between the query vector and every chunk, then score each section by its highest-scoring chunk. Sorting by these scores yields a relevance ranking across the entire corpus.

5. *Precision-Driven Filtering & Summarization*
   To weed out generic or marginally related content, we apply:

   * A *similarity threshold*, discarding sections scoring below a fraction of the top score
   * An *n-gram overlap filter*, requiring at least one bigram or trigram match between the dynamic keyphrases and the section’s title+content
   * A *post-rank summary scan*, where we generate an extractive summary (via TextRank) and only keep sections whose summary still contains a keyphrase

   These layered filters boost precision without severely harming recall.

6. *Structured JSON Output*
   Finally, we emit a JSON containing metadata, the top N extracted sections with rank and source, and concise summaries for downstream consumption.

This architecture is fully offline, CPU-only, and agnostic to any specific persona or document set—simply plug in new PDFs, personas, and JBTDs, and it adapts dynamically.

1. Core Libraries & Models

PyMuPDF (fitz) for PDF parsing: extracts every text span along with font size, weight, and position—enabling us to merge fragmented lines and detect headings purely by layout.

SentenceTransformers using all-MiniLM-L6-v2: a 22 MB, CPU-friendly embedding model. We use it both to encode the user’s “persona + job-to-be-done” (JBTD) query and to embed every paragraph chunk in the documents.

KeyBERT to extract the top 5 keyphrases from the persona/JBTD text, giving us an initial set of domain-relevant terms.

scikit-learn’s cosine_similarity for fast, vector-based chunk-to-query scoring.

Summa’s TextRank (summarizer) for lightweight, extractive summaries of each selected section.

Standard Python (numpy, re, json, os, datetime) for numeric operations, regex-driven tokenization, I/O, and metadata.

2. Robust 1A Parsing
We encapsulate your 1A code (extract_from_pdf_with_merge) in a pdf_parser.py module. It:

Pulls all text spans (with font metrics),

Merges overlapping fragments via a maximal-overlap heuristic,

Identifies and deduplicates headers by font size/boldness,

Slices the merged flow into titled sections.
This ensures each section’s content is clean, coherent, and correctly bounded.

3. Dynamic Query Expansion

Build a corpus vocabulary by scanning all PDFs for 3+ letter words.

Use KeyBERT to extract initial keyphrases from the persona+JBTD, then semantic_search (SentenceTransformers) against the corpus vocabulary to dynamically expand them.
This adapts to any domain without hard-coding.

4. IDF-Weighted & Last-Word-Boosted Query Embedding

Embed the full query string (persona + JBTD + all keyphrases).

Compute an IDF-weighted average of each keyphrase embedding—rare, discriminative terms get more influence.

Blend these two with weight β (e.g. 0.6).

Finally, embed the last word of the JBTD (α ≈ 0.3) and mix it in, emphasizing the user’s ultimate focus.

5. Chunking, Ranking & Filtering

Split each section into paragraph chunks and embed them.

Score every chunk by cosine similarity to the query vector; assign each section its maximum chunk score and sort.

Apply a similarity threshold (keep only ≥ 60% of top-score sections).

Enforce an n-gram overlap filter (bigrams/trigrams) between dynamic keyphrases and section title+content.

Do a post-rank summary scan: generate a TextRank summary and drop any section whose summary no longer contains a keyphrase.

6. JSON Output
We emit a structured JSON with metadata, the top N sections (document, page, title, rank), and concise summaries.

Altogether, this CPU-only, offline pipeline—from your proven 1A parser through dynamic semantic ranking—delivers high precision and recall for any PDF set and any persona/JBTD query.