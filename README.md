# 🏔️ Smart Outdoor Gear Discovery Engine

A production-grade, hybrid-driven **Retrieval-Augmented Generation (RAG)** recommendation system and semantic search engine built for outdoor equipment catalog spaces. The application leverages deep text transformers to encode 2,222 unique product listings across 6 commercial pillars into a high-density vector space, utilizing an integrated mathematical scoring framework to balance contextual alignment with performance and value metrics.

---

## 🏗️ System Architecture & RAG Pipeline Flow

This system replaces archaic, rigid keyword search mechanics with a structured **Dense Retrieval and Multi-Attribute Ranking Layer** mirroring a structured RAG pattern:

```text
  [ User Text Query ] 
          │
          ▼
 1. RETRIEVAL (R)  ──► Encode via all-MiniLM-L6-v2 ──► Matrix Scan (Cosine Similarity)
          │
          ▼
 2. AUGMENTATION (A) ──► Blend Similarity with normalized Ratings & Price Discounts
          │
          ▼
 3. GENERATION (G) ──► Streamlit Engine dynamically constructs UI Card Interfaces