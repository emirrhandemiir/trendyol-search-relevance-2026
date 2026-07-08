# trendyol-search-relevance-2026
End-to-End Machine Learning Pipeline for E-Commerce Search Relevance. Features: Cross-Encoder Fine-Tuning, Embedding-based Hard Negative Mining, and LightGBM with advanced NLP features.
# 🚀 Trendyol E-Commerce Search Relevance Pipeline (2026)

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-ffce54?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-ffcc00?style=for-the-badge&logo=huggingface)

## 📌 Project Overview
This repository contains an end-to-end Machine Learning pipeline designed for the **Trendyol 2026 E-Commerce Search Relevance Competition**. Developed by the **Tek Projekt** team, this pipeline predicts the relevance of search queries to product listings using advanced NLP techniques and gradient boosting.

The architecture eliminates data leakage, handles Turkish morphological challenges, and utilizes semantic domain adaptation to understand e-commerce-specific terminologies (e.g., model codes, brands).

## 🏗️ Architecture & Pipeline

The system is built on a robust 3-Phase architecture:

### Phase 1: Advanced Feature Engineering
* **Attribute Cleansing:** Filtered raw product attributes to extract only high-signal data (color, size, material, gender).
* **Regex Model Code Extraction:** Custom regex patterns to capture complex alphanumeric e-commerce model codes (e.g., `TB0A6BVJW021`, `SM-G990`) from both queries and product texts.
* **Fuzzy Matching:** Implemented `RapidFuzz` (`fuzz_ratio`, `fuzz_partial`) to handle Turkish suffix/prefix structures seamlessly.

### Phase 2: Embedding-Based Hard Negative Mining
To train a highly discriminative model, random negative sampling was abandoned. 
* Utilized **Bi-Encoder** (`paraphrase-multilingual-MiniLM-L12-v2`) for semantic search across the product catalog.
* Employed **Cosine Similarity** (`util.cos_sim`) to mine the top 50 most confusing negative products for each query.
* Achieved a strict and balanced **1:2 (Positive:Negative) ratio** (~342k hard negatives) to prevent extreme class imbalance and category dominance.

### Phase 3: Cross-Encoder Fine-Tuning (Domain Adaptation)
* General-purpose Cross-Encoder (`mmarco-mMiniLMv2-L12-H384-v1`) was fine-tuned specifically on Trendyol e-commerce data.
* **Optimizations:** Implemented `use_amp=True` (Automatic Mixed Precision) for memory efficiency and `save_best_model=True` to prevent overfitting during the epoch.
* The fine-tuned semantic score was fed back into the final tree-based model.

### 🏆 Final Model
* **Algorithm:** LightGBM Classifier (`class_weight='balanced'`)
* **Evaluation:** GroupShuffleSplit was used to ensure zero data leakage across search queries.

## 📊 Results
* **Baseline F1 Score:** `0.4230` (Simple Rule-Based)
* **Final Validation F1 Score:** `0.8453` (Full Pipeline + Ablation Tested)

## 👨‍💻 Author
**Emirhan Demir** 
Computer Engineering Student @ Nevşehir Hacı Bektaş Veli Üniversitesi (NEVÜ)

*Feel free to reach out for discussions regarding ML architectures, search relevance, or potential internship opportunities.*
