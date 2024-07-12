# 🏆 Swahili Embeddings Text Leaderboard

Welcome to the Swahili Embeddings Text Leaderboard! This is a collaborative community project aimed at creating a centralized leaderboard for Swahili text embeddings. The models listed here are evaluated using various Swahili text embeddings. Contributions and corrections are always welcome! We define a model as "open" if it can be locally deployed and used commercially.

## 🌐 Interactive Dashboard

Explore our interactive dashboards:

- [Streamlit Dashboard](https://swahili-llm-leaderboard.streamlit.app/)
- [Hugging Face Space](https://huggingface.co/spaces/Mollel/swahili-llm-leaderboard)

## 📊 Leaderboard
| Model Name                                         | Publisher   | Open?   | Basemodel   | Matryoshka   | Dimension   |   Average |   AfriSentiClassification |   AfriSentiLangClassification |   MasakhaNEWSClassification |   MassiveIntentClassification |   MassiveScenarioClassification |   SwahiliNewsClassification |   NTREXBitextMining |   MasakhaNEWSClusteringP2P |   MasakhaNEWSClusteringS2S |    XNLI |   MIRACLReranking |   MIRACLRetrieval |
|:---------------------------------------------------|:------------|:--------|:------------|:-------------|:------------|----------:|--------------------------:|------------------------------:|----------------------------:|------------------------------:|--------------------------------:|----------------------------:|--------------------:|---------------------------:|---------------------------:|--------:|------------------:|------------------:|
| MultiLinguSwahili-bge-small-en-v1.5-nli-matryoshka | sartifyllc  |         |             |              |             |   36.3029 |                    35.107 |                       67.3486 |                     54.1597 |                       38.0027 |                         46.8393 |                     51.2305 |             5.01061 |                    21.7986 |                    17.8461 | 62.3059 |            21.521 |            14.465 |


## 🧪 Evaluation
To evaluate a model on the Swahili Embeddings Text Benchmark, you can use the following Python script:
```python
pip install mteb
pip install sentence-transformers
import mteb
from sentence_transformers import SentenceTransformer

model_name = "sartifyllc/MultiLinguSwahili-bge-small-en-v1.5-nli-matryoshka"
truncate_dim = 256
language = "swa"

model = SentenceTransformer(model_name, truncate_dim = truncate_dim)
tasks = mteb.get_tasks(languages=[language])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
```


## 🤝 How to Contribute

We welcome and appreciate all contributions! You can help by:

### Table Work

- Filling in missing entries.
- New models are added as new rows to the leaderboard (maintaining descending order).
- Add new benchmarks as new columns in the leaderboard and include them in the benchmarks table (maintaining descending order).

### Code Work

- Improving the existing code.
- Requesting and implementing new features.

## 🤝 Sponsorship

This benchmark is Swahili-based, and we need support translating and curating more tasks into Swahili. Sponsorships are welcome to help advance this endeavour. Your sponsorship will facilitate essential translation efforts, bridge language barriers, and make the benchmark accessible to a broader audience. We are grateful for the dedication shown by our collaborators and aim to extend this impact further with the support of sponsors committed to advancing language technologies.

---

Thank you for being part of this effort to advance Swahili language technologies!
