<div align="center">

# 📍 Iota Address Parser

**An intelligent NER-powered engine that transforms messy, unstructured address strings into clean, structured JSON.**<br/>
*Built with SpaCy and Programmatic Weak Supervision.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![SpaCy](https://img.shields.io/badge/SpaCy-NER-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
[![ML Pipeline](https://img.shields.io/badge/Pipeline-End--to--End-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://github.com/SHAILESH-RS-UPADHYAY/iota-address-parser)

</div>

---

## 🚀 How It Works

**End-to-end pipeline:**
> `Raw CSV Addresses` ➔ `Synthetic Data Generator` ➔ `SpaCy NER Training` ➔ `Evaluation` ➔ `Interactive Inference CLI`

```json
// Input: "456 oak avenue, portland, oregon 97201"
{
  "Street": "456 oak avenue",
  "City": "portland",
  "State": "oregon",
  "Zip": "97201"
}
```

The model extracts **4 core entity types** from free-text addresses:

| Entity | Description | Example |
|--------|-------------|---------|
| `STREET` | Street number + name | "123 Main St" |
| `CITY` | City name | "San Francisco" |
| `STATE` | US state name | "California" |
| `ZIP` | Postal code | "94102" |

---

## 🏗️ Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4F46E5', 'edgeLabelBackground':'#ffffff'}}}%%
graph TD
    classDef data fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff,rx:10px,ry:10px;
    classDef process fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff,rx:10px,ry:10px;
    classDef model fill:#8b5cf6,stroke:#7c3aed,stroke-width:2px,color:#fff,rx:10px,ry:10px;
    classDef output fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff,rx:10px,ry:10px;

    A["us_addresses.csv (100+ real US addresses)"]:::data --> B["generate_data.py"]:::process
    B -->|"Programmatic Weak Supervision<br/>7 format variations"| C["training_data.json (2500 samples)"]:::data
    C --> D["train_model.py"]:::model
    D -->|"SpaCy blank:en + NER<br/>SGD, 25% dropout, 50 epochs"| E["address_ner_model/"]:::model
    E --> F["evaluate_model.py"]:::process
    F -->|"80/20 split, seed=42"| G["Precision / Recall / F1"]:::output
    E --> H["inference.py"]:::output
    H --> I["Interactive CLI Parser"]:::output
```

---

## 🧠 Why These Technical Choices?

*   **SpaCy over Transformers/BERT:** For structured entity extraction from short text, SpaCy's statistical NER is faster, lighter, and produces comparable accuracy. BERT would be overkill for 4-entity extraction from single-line inputs.
*   **Programmatic Weak Supervision over Manual Labeling:** Instead of hand-labeling 2,500 addresses, I wrote labeling functions that reverse-engineer structured CSV rows into messy strings while tracking exact character offsets. This generates perfectly labeled data at scale with zero human annotation cost.
*   **7 Format Variations for Robustness:** Real-world addresses come in wildly inconsistent formats. The generator creates various permutations (e.g., standard, missing spaces, lowercase) forcing the model to learn patterns, not just punctuation.
*   **Compounding Batch Sizes (8 ➔ 64):** Starts with small batches for precise early learning, then scales up for stable convergence—a well-known SpaCy training optimization.
*   **25% Dropout Regularization:** Prevents overfitting on a synthetically generated dataset where patterns could otherwise be too clean.

---

## 📁 Project Structure

```text
iota-address-parser/
├── generate_data.py        # Synthetic data generator (Weak Supervision)
├── train_model.py          # SpaCy NER training loop
├── evaluate_model.py       # Precision/Recall/F1 scoring
├── inference.py            # Interactive CLI for live address parsing
├── us_addresses.csv        # Seed data
├── training_data.json      # Generated labeled training samples
├── address_ner_model/      # Trained SpaCy model artifact
│   ├── meta.json
│   ├── config.cfg
│   └── ner/                # Serialized NER weights
└── requirements.txt
```

---

## 💻 Run It Locally

**1. Clone the repository**
```bash
git clone https://github.com/SHAILESH-RS-UPADHYAY/iota-address-parser.git
cd iota-address-parser
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate training data** *(creates 2,500 samples from 100+ real addresses)*
```bash
python generate_data.py
```

**4. Train the model** *(50 epochs, ~2 min on CPU)*
```bash
python train_model.py
```

**5. Evaluate performance** *(prints Precision, Recall, F1 per entity)*
```bash
python evaluate_model.py
```

**6. Try it live!**
```bash
python inference.py
```

---

## 🚧 Known Limitations & Future Improvements

*   **US-Only Addresses:** Currently trained exclusively on US address formats. Supporting international formats (e.g., India, UK) would require additional labeling functions and entity types (e.g., `DISTRICT`, `PIN_CODE`).
*   **No Apartment/Unit Handling:** Addresses like *"123 Main St, Apt 4B"* aren't split into separate street and unit entities yet.
*   **Evaluation on Synthetic Data:** A production evaluation would require a hand-labeled holdout set of real, messy addresses.
*   **No API Wrapper:** Currently CLI-only. In production, this would be wrapped in a FastAPI endpoint for batch processing.

<br/>
<div align="center">
  <i>Built to demonstrate end-to-end ML engineering from data generation to deployment.</i>
</div>
