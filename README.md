

One notebook per week; outputs are kept so results can be read without re-running.
 
---
 
## Week 1: Embeddings, chunking, incremental pipeline, model comparison
 
**Notebook:** [`week1/week1_embeddings.ipynb`](week1/week1_embeddings.ipynb)
 
| Section | What it does |

|---|---|

| 1. Embedding experiments (A–F) | What similarity captures, silent truncation, query prefixes, throughput, tokenizer differences, a minimal retriever |

| 2. Token-based chunker | `chunk_by_tokens` with overlap, input validation, and tests |

| 3. Pipeline + model comparison | Incremental embedding pipeline (manifest, content hashes, deletes, validation, Parquet) and a retrieval evaluation harness (golden set, recall@k, MRR) |
 
### Key findings (measured on a Colab CPU)
 
| Finding | Result |

|---|---|

| Embeddings capture **topic, not logic** | "fee **waived**" vs "fee **charged**": cosine **0.94** (MiniLM), **0.91** (bge-base) |

| **Silent truncation** past model limits | Two 663-token texts with different endings → cosine **1.0000** in both models; the only signal was a log warning |

| Tokenizers differ | Same banking text: **28** tokens (GPT-4 / GPT-4o tokenizers) vs **29** (BERT-based embedding models) |

| Throughput | MiniLM **114** chunks/s vs bge-base **15** chunks/s (**7.6× slower**) |

| Cost of a 10M-chunk backfill | MiniLM **24 h / 15.4 GB** vs bge-base **190 h / 30.7 GB** (float32) |

| Token-based chunking | 2,001 tokens → **25** chunks (size 100, overlap 20), matching ⌈(N−O)/(S−O)⌉ |

| **Idempotent re-runs** | Re-running with no changes embedded **0** chunks |

| **Incremental updates** | A fee change re-embedded only **4** chunks, kept **3** unchanged, and deleted **3** stale chunks from a removed document; queries returned only the new fee |
 
### Model comparison (12-chunk banking corpus, 16-question golden set)
 
| Model | recall@1 | recall@3 | MRR | Query latency | Storage / 10M |

|---|---|---|---|---|---|

| **all-MiniLM-L6-v2** | 0.81 | **1.00** | **0.90** | **18 ms** | **15.4 GB** |

| bge-base-en-v1.5 | 0.81 | 0.94 | 0.88 | 103 ms | 30.7 GB |

| bge-base + query prefix | 0.81 | **1.00** | **0.90** | 169 ms | 30.7 GB |
 
**recall@1 by question type** (all models): exact codes **1.0**, numbers **1.0**, paraphrases **0.8**, conditions **0.5**.
 
**What the errors showed:**

- **All three models failed the same condition questions.** "Is there a charge for *receiving* a wire?" retrieved the *outgoing* wire fee. A bigger model doesn't fix this; better retrieval does (a reranker that reads the question and passage together, or hybrid keyword + vector search).

- bge confused **overdraft** (item paid) with **NSF** (item returned) on "bounced check": a domain-vocabulary gap that public leaderboards don't measure. The bge query prefix moved the correct chunk into the top 3.
 
**Decision:** MiniLM. Same recall@1, matching or better recall@3 and MRR, at **5.8× lower query latency** and **half the storage**. (16 questions is a small sample, where one question is 6 points, so I'd confirm on a larger golden set before production.)
 
### Design decisions

- **Deterministic chunk IDs** (`doc_id:chunk_index`) make upserts idempotent, like `MERGE` on a key.

- **Content hashes in a manifest** mean only new or changed chunks are re-embedded; chunks that disappear are **deleted**, so stale facts are never served.

- **Validation before load:** vector shape, NaN, normalisation, and each chunk's token count against the embedding model's own limit.

- **Model upgrades are blue/green:** chunking and hashing are shared, with a separate manifest and collection per model, and traffic switches only after recall is compared. Vectors from different models are never mixed.

- **float32 Parquet** storage: half the size of float64, and loadable into a warehouse vector column.
 
### Run it

Open [`week1/week1_embeddings.ipynb`](week1/week1_embeddings.ipynb) in Google Colab (CPU runtime is fine) and choose **Runtime → Run all**. The first run downloads two small embedding models (about 530 MB) into the Colab session.
 
---
 
## Roadmap
 
| Week | Focus | Status |

|---|---|---|

| 1 | Embeddings, chunking, incremental embedding pipeline, model comparison | Done |

| 2 | Vector database (Qdrant), metadata filtering for multi-tenant isolation, hybrid BM25 + vector search | Next |

| 3 | Document parsing (PDFs, tables), structure-aware chunking, grounded RAG with citations | |

| 4 | Reranking and query rewriting, measured against the Week 1 baseline | |

| 5 | Embedding freshness with orchestration (Dagster), model migration | |

| 6 | Vector search and AI functions in Snowflake | |

| 7 | PII masking and access-controlled retrieval | |

| 8 | RAG evaluation (faithfulness, context recall) and regression testing | |
 
