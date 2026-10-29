# AI Data Foundations
 
**Building and measuring the data layer behind RAG**: token-based chunking, incremental

embedding pipelines, tenant-isolated vector search, and hybrid retrieval, with every claim

backed by a measured result.
 
**Highlights**

- **Missing filters fail silently:** without a date filter a *superseded* fee ranked #1; without a tenant filter *another bank's* document ranked #1. No errors, just confident wrong answers.

- **Incremental, idempotent embedding:** re-runs embed 0 chunks; a document change re-embeds only what changed and deletes stale chunks.

- **Honest benchmarks:** equal-weight hybrid search did not beat BM25 alone on recall@1; it exactly tied on look-alike regulation codes, which motivates weighted fusion and reranking next.
 
**Background:** data engineer (Snowflake, CDC pipelines, idempotent loads) in banking,

applying the same engineering discipline to AI data pipelines.
 
## Repository structure
 
    ai-data-foundations/

    ├── README.md

    ├── LICENSE

    ├── requirements.txt

    ├── .gitignore

    ├── week1_embeddings.ipynb

    └── week2_vector_search.ipynb
 
One notebook per week; outputs are kept so results can be read without re-running.
 
---
 
## Week 2: Vector search, metadata filters, tenant isolation, hybrid retrieval
 
**Notebook:** [`week2_vector_search.ipynb`](week2_vector_search.ipynb)
 
Qdrant (in-memory) with payload filters, a tenant leak test, BM25 written from scratch, and

Reciprocal Rank Fusion, evaluated on a 19-question banking golden set (two tenants, look-alike

ACH codes and CFR sections, one superseded fee schedule).
 
### Key findings
 
| Finding | Result |

|---|---|

| **Missing freshness filter = wrong answer** | Without a date filter, the **superseded 2025 fee ranked #1** for "How much is the overdraft fee?"; with the filter, the current fee ranked #1 |

| **Missing tenant filter = data leak** | Without a tenant filter, **another bank's chunk ranked #1** for bank A's query; with the server-side filter the leak test **passes** |

| Code-preserving tokenization | `12 CFR 1005.17`, `R04` kept intact as single tokens for keyword search |
 
### Retrieval results (19 questions, current documents, tenant bank_a)
 
| Method | recall@1 | recall@3 | MRR |

|---|---|---|---|

| BM25 | **0.79** | 0.89 | 0.84 |

| Dense (MiniLM) | 0.74 | **1.00** | **0.86** |

| Hybrid (RRF, equal weights) | 0.74 | 0.95 | 0.85 |
 
| recall@1 by type | BM25 | Dense | Hybrid |

|---|---|---|---|

| exact codes | **1.00** | 0.86 | 0.86 |

| conditions | **0.75** | 0.50 | 0.50 |

| numbers | 0.67 | **1.00** | **1.00** |

| paraphrases | 0.60 | 0.60 | 0.60 |
 
**What the errors showed:**

- **Look-alike codes:** dense search confused `12 CFR 1005.17` with `1005.10`; BM25 got it right. Equal-weight RRF produced an **exact tie** (each retriever ranked the two chunks #1 and #2 in opposite order) and kept the wrong one.

- BM25 fixed "How do I avoid the monthly account fee?" through the shared keyword "monthly", but **no method** matched "receiving a wire" to "incoming wire transfers": a synonym with no shared word.

- Equal-weight hybrid **did not beat the best single retriever** on recall@1. Its value here is the candidate pool: dense recall@3 is 1.00, so the right chunk is always in the top 3.
 
**Next:** weighted fusion or query routing for code-like queries, and a cross-encoder reranker over the fused top-10 (Week 4), targeting condition and paraphrase questions.
 
### Design decisions

- **Tenant and freshness filters are applied server-side in one search function**; the unfiltered path exists only as a demo, and a leak test checks isolation.

- **Pre-filtered keyword search:** one BM25 index per tenant over current documents only.

- **Fusion by rank (RRF), not score:** dense and BM25 scores are on incompatible scales. O(L·n + u log u) time.

- **Payload indexes** on filtered fields (`tenant_id`, `effective_year`); a no-op in local mode, required on a real server.
 
---
 
## Week 1: Embeddings, chunking, incremental pipeline, model comparison
 
**Notebook:** [`week1_embeddings.ipynb`](week1_embeddings.ipynb)
 
### Key findings (measured on a Colab CPU)
 
| Finding | Result |

|---|---|

| Embeddings capture **topic, not logic** | "fee **waived**" vs "fee **charged**": cosine **0.94** (MiniLM), **0.91** (bge-base) |

| **Silent truncation** past model limits | Two 663-token texts with different endings → cosine **1.0000**; the only signal was a log warning |

| Tokenizers differ | Same banking text: **28** tokens (GPT-4 / GPT-4o tokenizers) vs **29** (BERT-based embedding models) |

| Throughput | MiniLM **114** chunks/s vs bge-base **15** chunks/s (**7.6× slower**) |

| Cost of a 10M-chunk backfill | MiniLM **24 h / 15.4 GB** vs bge-base **190 h / 30.7 GB** (float32) |

| Token-based chunking | 2,001 tokens → **25** chunks (size 100, overlap 20), matching ⌈(N−O)/(S−O)⌉ |

| **Idempotent re-runs** | Re-running with no changes embedded **0** chunks |

| **Incremental updates** | A fee change re-embedded only **4** chunks, kept **3** unchanged, and deleted **3** stale chunks; queries returned only the new fee |
 
### Model comparison (16-question golden set)
 
| Model | recall@1 | recall@3 | MRR | Query latency | Storage / 10M |

|---|---|---|---|---|---|

| **all-MiniLM-L6-v2** | 0.81 | **1.00** | **0.90** | **18 ms** | **15.4 GB** |

| bge-base-en-v1.5 | 0.81 | 0.94 | 0.88 | 103 ms | 30.7 GB |

| bge-base + query prefix | 0.81 | **1.00** | **0.90** | 169 ms | 30.7 GB |
 
**recall@1 by question type** (all models): exact codes **1.0**, numbers **1.0**, paraphrases **0.8**, conditions **0.5**.
 
All three models failed the same condition questions, so the fix is retrieval technique, not a bigger model. bge confused **overdraft** (item paid) with **NSF** (item returned) on "bounced check", a domain-vocabulary gap that public leaderboards don't measure. **MiniLM chosen:** same quality at 5.8× lower query latency and half the storage.
 
### Design decisions

- **Deterministic chunk IDs** (`doc_id:chunk_index`) make upserts idempotent, like `MERGE` on a key.

- **Content hashes in a manifest:** only new or changed chunks are re-embedded; removed chunks are deleted, so stale facts are never served.

- **Validation before load:** vector shape, NaN, normalisation, and token count against the model's limit.

- **Blue/green model upgrades:** shared chunking and hashing, a separate collection per model; never mix vectors from different models.

- **float32 Parquet** storage: half the size of float64, and loadable into a warehouse vector column.
 
---
 
## Roadmap
 
| Week | Focus | Status |

|---|---|---|

| 1 | Embeddings, chunking, incremental embedding pipeline, model comparison | Done |

| 2 | Vector database, metadata filtering, tenant isolation, hybrid BM25 + vector search | Done |

| 3 | Document parsing (PDFs, tables), structure-aware chunking, grounded RAG with citations | Next |

| 4 | Reranking and query rewriting, measured against the Week 1–2 baselines | |

| 5 | Embedding freshness with orchestration (Dagster), model migration | |

| 6 | Vector search and AI functions in Snowflake | |

| 7 | PII masking and access-controlled retrieval | |

| 8 | RAG evaluation (faithfulness, context recall) and regression testing | |
 
## Run it

Open any notebook in Google Colab (CPU runtime is fine) and choose **Runtime → Run all**.
 
## License

MIT. See [LICENSE](LICENSE).
 
