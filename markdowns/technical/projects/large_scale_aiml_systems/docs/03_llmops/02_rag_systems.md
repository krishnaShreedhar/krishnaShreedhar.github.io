---
title: "RAG Systems"
subtitle: "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by retrieving relevant documents from an external knowledge base and including them in the context window before generation. RAG..."
category: technical
project: large_scale_aiml_systems
project_title: "Large Scale AI/ML Systems"
date: 2025-04-01
reading_time: 4
tags:
  - large-scale-aiml-systems
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_aiml_systems/docs/03_llmops/02_rag_systems.html"
---
Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by retrieving relevant documents from an external knowledge base and including them in the context window before generation. RAG addresses the key limitations of standalone LLMs: knowledge cutoff dates, hallucination of facts, and inability to access private or proprietary knowledge — without requiring expensive model fine-tuning.

## RAG Architecture

```mermaid
graph TD
    subgraph Indexing[Offline Indexing Pipeline]
        Docs[Source Documents\nPDFs, wikis, databases\ncode, emails, reports]
        Load[Document Loader\nparse PDF, HTML, Markdown\nextract text and metadata]
        Chunk[Chunking\nsplit into passages\n256-1024 tokens each\noverlap 10-20%]
        Embed[Embedding Model\ntext-embedding-3-small\nor sentence-transformers\ncreates dense vectors]
        VectorStore[Vector Store\nFAISS, Pinecone, Weaviate\nChroma, pgvector\nindexes vectors for ANN search]

        Docs --> Load --> Chunk --> Embed --> VectorStore
    end

    subgraph Query[Online Query Pipeline]
        UserQ[User Query]
        QueryEmbed[Embed Query\nsame embedding model\nas indexing]
        Retrieve[ANN Search\nTop-K most similar\nchunks - K=3 to 10]
        Rerank[Optional Reranking\ncross-encoder reranker\nmore accurate than ANN\nbut slower]
        Augment[Augmented Prompt\nSystem prompt\nRetrieved context\nUser question]
        LLM[LLM Generation\nGPT-4 Claude Llama\nanswers using context]
        Answer[Generated Answer\nwith source citations]

        UserQ --> QueryEmbed --> Retrieve --> Rerank --> Augment --> LLM --> Answer
    end

    VectorStore --> Retrieve

    style VectorStore fill:#dbeafe,stroke:#2563eb,stroke-width:2px
    style LLM fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

## Chunking Strategies

```mermaid
graph TD
    subgraph Chunking[Document Chunking Approaches]
        subgraph Fixed[Fixed-Size Chunking]
            F1[Split by token count\n512 tokens with 50 token overlap\nSimple to implement\nIgnores document structure]
        end

        subgraph Semantic[Semantic Chunking]
            S1[Split at natural boundaries\nparagraphs, sections, sentences\nPreserves meaning units\nVariable chunk size]
        end

        subgraph Hierarchical[Hierarchical Chunking]
            H1[Parent document: full section\nChild chunks: paragraphs\nRetrieve children\nReturn parent context\nPreserves context around retrieved chunk]
        end

        subgraph Agentic[Agentic Chunking]
            A1[LLM determines chunk boundaries\nbased on semantic meaning\nHighest quality\nExpensive to compute at index time]
        end

        Fixed --> Semantic --> Hierarchical --> Agentic
    end
```

## Retrieval Quality Improvement

```mermaid
graph TD
    subgraph Improvements[RAG Quality Techniques]
        subgraph QueryTransform[Query Transformation]
            HyDE[HyDE - Hypothetical Document Embedding\nGenerate a hypothetical answer\nembed that instead of query\nImproves retrieval for question-answering]
            QueryExpand[Query Expansion\nGenerate multiple phrasings\nof the user question\nretrieve for each\nmerge results]
        end

        subgraph Hybrid[Hybrid Retrieval]
            Dense[Dense Retrieval\nembedding similarity\nsemantics]
            Sparse[Sparse Retrieval\nBM25 keyword search\nexact term matching]
            HybridSearch[Hybrid Search\nReciprocal Rank Fusion\ncombines dense and sparse\nbetter than either alone]
            Dense & Sparse --> HybridSearch
        end

        subgraph Reranking[Cross-Encoder Reranking]
            Rerank[Cross-Encoder Model\nconsiders query AND document together\nmore accurate than bi-encoder ANN\nreranks top-50 candidates to top-5\nbge-reranker cohere-rerank]
        end
    end
```

## Key Concepts

- **Vector Embeddings**: Dense numerical representations of text that capture semantic meaning — similar texts have nearby vectors in embedding space. Computed by embedding models (OpenAI text-embedding-3-small, Cohere Embed, sentence-transformers). The quality of the embedding model directly determines retrieval quality. Embedding the query with the same model used for indexing is essential.

- **Approximate Nearest Neighbor (ANN) Search**: Finding the K vectors most similar to the query vector from potentially millions of indexed chunks. Exact search scales as O(n*d) which is too slow for large knowledge bases. ANN algorithms (HNSW, IVF, LSH) trade perfect recall for dramatic speed improvements, typically achieving 95-99% recall at 10-100x speedup.

- **Chunking**: Splitting long documents into passages small enough to fit in the context window while large enough to be semantically meaningful. Chunk size is a critical hyperparameter — too small misses context, too large includes irrelevant text. Overlapping adjacent chunks ensures information at chunk boundaries is not lost.

- **Context Window Stuffing**: After retrieval, the top-K chunks are concatenated into the LLM context window along with the system prompt and user query. Context windows of 128K tokens (GPT-4 Turbo, Claude) can accommodate many chunks, but LLM performance degrades for information in the middle of very long contexts (the "lost in the middle" problem). Reranking and selecting fewer, higher-quality chunks often outperforms retrieving more chunks.

- **Hybrid Search**: Combining dense embedding search (captures semantics) with sparse keyword search (BM25 — captures exact term matches). Many queries benefit from both: "What is the revenue of AAPL?" needs keyword match for AAPL and semantic understanding of revenue. Reciprocal Rank Fusion (RRF) is the standard method for merging ranked lists from multiple retrieval methods.

- **RAG Evaluation**: Measuring RAG system quality requires evaluating multiple components: retrieval recall (were the relevant documents retrieved?), context relevance (are the retrieved documents actually relevant to the query?), answer faithfulness (does the answer only use information from the retrieved context?), and answer correctness (is the answer factually correct?). RAGAS is a framework that automates these evaluations using LLM-as-judge.

- **Metadata Filtering**: Restricting retrieval to a subset of the knowledge base using metadata filters (date range, document type, department, access permissions). Metadata filtering combines with vector search to implement access control and scoped search. Example: only retrieve documents from the past 6 months, or only retrieve documents the user has permission to see.

## Trade-offs

| Retrieval Method | Semantic Quality | Exact Match | Speed | Complexity |
|----------------|-----------------|------------|-------|-----------|
| Dense (ANN only) | High | Low | Very Fast | Low |
| Sparse (BM25 only) | Low | High | Fast | Low |
| Hybrid (Dense + Sparse) | High | High | Fast | Medium |
| Hybrid + Reranker | Highest | High | Medium | High |

## When to Use

- **RAG over fine-tuning**: When the knowledge base changes frequently (news, product docs, pricing), when you need source citations, or when domain knowledge is vast and structured — RAG is cheaper and more maintainable than fine-tuning
- **Fine-tuning over RAG**: When the task requires a specific output format or style that cannot be achieved by prompting, or when the knowledge is stable and the model needs to internalize reasoning patterns rather than facts
- **Hybrid search**: Default for production RAG systems — pure dense retrieval misses exact keyword matches that users expect
- **Hierarchical chunking**: When source documents have clear structure (documentation, reports) and surrounding context is important for understanding retrieved passages