# 🏔️ Smart Outdoor Gear Discovery Engine

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://outdoor-recommender.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

An end-to-end, hybrid-driven recommendation system and semantic search engine built for specialized outdoor equipment catalog spaces. The application encodes **2,222 unique product listings** scraped from *bergfreunde.de* into a high-density vector space, utilizing an integrated mathematical scoring framework to balance contextual alignment with performance and economic value metrics.

🔗 **Explore the interactive application here:** [Live Streamlit Production Deployment](https://outdoor-recommender.streamlit.app/)


---



## 🏗️ System Architecture & Search Pipeline Flow



This system replaces archaic, rigid keyword search mechanics with a structured **Dense Retrieval and Multi-Attribute Ranking Layer** mirroring a structured RAG pattern:



```text

  [ User Text Query ] 

          │

          ▼

 1. RETRIEVAL  ──► Encode via all-MiniLM-L6-v2 ──► Matrix Scan (Cosine Similarity)

          │

          ▼

 2. RANKING & AUGMENTATION ──► Blend Similarity with normalized Ratings & Price Discounts

          │

          ▼

 3. INTERFACE GENERATION ──► Streamlit Engine dynamically constructs UI Card Interfaces



```



* **Retrieval Phase:** RRaw unstructured user text inputs are projected as coordinates into an offline-compiled 384-dimensional vector space, executing an angular vector scan across our data assets using **Cosine Similarity**.

* **Ranking & Augmentation Phase:** The system intercepts retrieved raw semantic rows and overlays metadata columns (`rating_score`, `discount_percentage`) using a dynamic vector scoring formula.

* **Interface Generation Phase:** The balanced payload is delivered to an operational dashboard, dynamically constructing formatted grid interfaces, review metrics, and active store referral routing targets on the fly.



---



## 🧮 Theoretical & Mathematical Framework



### 1. Vector Space Representation

Textual metadata profiles (compiled fields integrating name, brand, category, and descriptive context text weight inputs) are passed through an `all-MiniLM-L6-v2` SentenceTransformer. The model translates natural language features into an immutable coordinate space:

Matrix Dimensions: 2,222 products x 384 feature coordinates

### 2. Angular Cosine Similarity

To measure semantic affinity independent of text segment magnitude or description length, the engine calculates the dot product of normalized vector coordinates:

Cosine Similarity(A, B) = (A . B) / (||A|| * ||B||)

### 3. Multi-Attribute Hybrid Re-ranking Formula

To ensure recommendations balance contextual relevance with consumer value signals (preventing cold, purely text-matching loops), recommendations are sorted using a controlled parameter equation:

Final Score = (Alpha * Cosine_Similarity) + (Beta * (Rating_Score / 5.0)) + (Gamma * (Discount_Percentage / Max_Discount))

> ### 📊 Dynamic Parameter Weight Allocations
> 
> * **Alpha = 0.70** — Contextual Text Alignment Weight
> * **Beta = 0.20** — User Validation Quality (Ratings) Weight
> * **Gamma = 0.10** — Promotional Economic Value (Markdown Discounts) Weight


---



## 📂 Project Repository Map

| File / Directory Path | Tier Level | Architectural Responsibility |
| :--- | :--- | :--- |
| **`data/products.csv`** | Storage | Raw unstructured data crawled from production channels |
| **`data/products_engineered.csv`** | Storage | Compiled string arrays & structural attribute tables |
| **`data/embeddings.npy`** | Matrix | Dense array storage of pre-compiled (2222, 384) coordinate weights |
| **`data/similarity_matrix.npy`** | Matrix | Symmetrical array mapping structural 2222 x 2222 cosine directions |
| **`src/scraper.py`** | Engine Component | Automated listing extraction and web scraping module |
| **`src/feature_engineering.py`** | Engine Component | Automated concatenation text compilers and database normalizers |
| **`src/embedding_generator.py`** | Engine Component | Local vectorization loader hosting the SentenceTransformer engine |
| **`src/recommender_engine.py`** | Core Logic | Executes real-time vector queries and multi-criteria scoring equations |
| **`tests/test_recommender.py`** | Diagnostics | Regression test suite evaluating runtime vector consistency |
| **`app.py`** | Frontend Interface | Serves the interactive Streamlit responsive browser dashboard |


---



## 🛡️ Software Engineering Best Practices Applied



* **Separation of Concerns:** Deep mathematical engine pipelines are completely decoupled from frontend web components. Core scoring rules live entirely in `src/`, enabling independent UI layer modifications.

* **State Performance Caching:** Initializing deep learning architectures on every browser interaction introduces memory and latency bottlenecks. This codebase uses Streamlit `@st.cache_resource` mechanics to hold vector models securely in RAM cache for instantaneous sub-second render loops.

* **Defensive Engineering & Text Regularization:** Raw catalog scraped text features include heavy tracking codes and layout strings. The UI incorporates an isolated `re` regular expression parsing engine to wipe structural anomalies safely on the fly without breaking core dataframe index links.

* **Automated Regression Coverage:** Includes built-in unit tests verifying matrix dimensions, missing storage structures, and reranking behavioral correctness prior to delivery.



---



## 🚀 Installation and Local Execution



### 1. Initialize Your Virtual Environment



```bash
# Clone the repository workspace

git clone <your-repository-url>

cd outdoor-recommender



# Construct and activate isolated virtual environment wrapper

python3 -m venv env

source env/bin/activate
```



### 2. Install Project Dependencies



```bash
pip install -r requirements.txt
```



### 3. Run Automated Testing Validation Suite



```bash
python3 -m unittest tests/test_recommender.py
```



### 4. Launch the Production User Interface



```bash
streamlit run app.py
```