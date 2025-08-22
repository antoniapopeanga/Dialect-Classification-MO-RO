# Dialect Classification: Romanian vs Moldovan  

This project, developed within **NitroNLP**, explores the task of distinguishing between **Romanian Standard** and the **Moldovan dialect** using the **MOROCO corpus**. It combines **linguistic feature analysis** with **machine learning models** (Random Forest, TF-IDF) and **deep learning approaches** (fine-tuned RoBERTa for Romanian).  

## 📌 Project Overview  

- **Goal**: Automatically classify text samples as either Romanian or Moldovan dialect.  
- **Datasets**:  
  - [MOROCO](https://github.com/butnaruandrei/MOROCO) – news corpus for Romanian & Moldovan dialects.  
  - [SlavicNER](https://github.com/BSoboleva/SlavicNER) – used for comparative analysis with Russian texts.  

## ⚙️ Data Processing  

1. **Dataframes creation** – train, validation, and test splits.  
2. **Minimal preprocessing**:  
   - Lowercasing  
   - Removing punctuation, numbers, and extra spaces  
   - Preserving Named Entity tokens  
3. **Linguistic analysis**:  
   - **Diacritics frequency**: Moldovan shows higher density of **î, ș, ț**, with **î** often used inside words.  
   - **POS tagging** (using *Stanza* for Romanian):  
     - Romanian → more verbs, higher POS variety  
     - Moldovan → more nouns  
   - **Russian influence**: similarity scores (cosine similarity with SlavicNER data):  
     - Romanian ↔ Moldovan: **0.99**  
     - Romanian ↔ Russian: **-0.0005**  
     - Moldovan ↔ Russian: **+0.01** (slightly stronger connection than Romanian).  

## 🧪 Classification Approaches  

### 1. **Classical ML with Linguistic Features**  
- **Features**: TF-IDF (up to 15,000 n-grams, range 4–6), diacritics frequency, î/â ratio, POS tags.  
- **Model**: Random Forest.  
- **Performance**:  
  - Validation accuracy: **88.3%**  
  - Test accuracy: **88.4%**  
  - Misclassifications:  
    - 219 / 2472 Moldovan samples  
    - 466 / 3452 Romanian samples  
- **Top discriminative features**:  
  - Frequency of **ș** and **ț**  
  - Words like *moldovenesc* and derivatives  
  - Ratios **î/â** and **ă/a**  
  - POS tags (notably *noun*).  

### 2. **Deep Learning with Transformers**  
- **Model**: Fine-tuned *RoBERT* (Romanian pretrained transformer).  
- **Setup**:  
  - Tokenizer with max length 128  
  - CrossEntropyLoss  
  - Training: 5 epochs, batch size 16  
- **Performance**:  
  - Test accuracy: **94.7%**  
- **Insights**:  
  - Captures subtle linguistic traits without manual feature engineering  
  - Outperforms Random Forest by ~6%  

## 📊 Results Comparison  

| Model                  | Accuracy (Test) | Notes |
|------------------------|----------------|-------|
| Random Forest (TF-IDF + features) | 88.4% | More interpretable, highlights discriminative features |
| Fine-tuned RoBERT      | 94.7% | Less interpretable, but significantly better performance |

## ✅ Conclusions  

- **Random Forest + linguistic features**: offers interpretability, highlights discriminative linguistic markers, but misses subtle patterns.  
- **RoBERT transformer**: superior performance, learns deep language representations automatically, but harder to interpret.  
- **Overall**: Transformers are more effective for **dialect classification**, while classical ML aids **explainability**.  

## 🚀 Future Work  

- Extend analysis with **cross-lingual embeddings** (Romanian–Moldovan–Russian).  
- Apply **explainable AI (XAI)** methods (e.g., SHAP, LIME) to interpret RoBERT predictions.
  
---

🔬 *This project was developed as part of NitroNLP, focusing on the intersection of linguistic analysis and machine learning for dialect classification.*  
