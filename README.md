# 🏆 Swahili Embeddings Text Leaderboard

Welcome to the Swahili Embeddings Text Leaderboard! This is a collaborative community project aimed at creating a centralized leaderboard for Swahili text embeddings. The models listed here are evaluated using various Swahili text embeddings. Contributions and corrections are always welcome! We define a model as "open" if it can be locally deployed and used commercially.

## 🌐 Interactive Dashboard

Explore our interactive dashboards:

- [Streamlit Dashboard](https://swahili-llm-leaderboard.streamlit.app/)
- [Hugging Face Space](https://huggingface.co/spaces/Mollel/swahili-llm-leaderboard)

## 📊 Leaderboard
| Model Name                                                                                                                              | Publisher      | Open?   | Basemodel                 | Matryoshka   | Dimension   |   Average |   AfriSentiClassification |   AfriSentiLangClassification |   MasakhaNEWSClassification |   MassiveIntentClassification |   MassiveScenarioClassification |   SwahiliNewsClassification |   NTREXBitextMining |   MasakhaNEWSClusteringP2P |   MasakhaNEWSClusteringS2S |    XNLI |   MIRACLReranking |   MIRACLRetrieval |
|:----------------------------------------------------------------------------------------------------------------------------------------|:---------------|:--------|:--------------------------|:-------------|:------------|----------:|--------------------------:|------------------------------:|----------------------------:|------------------------------:|--------------------------------:|----------------------------:|--------------------:|---------------------------:|---------------------------:|--------:|------------------:|------------------:|
| [MultiLinguSwahili-serengeti-E250-nli-matryoshka](https://huggingface.co/sartifyllc/MultiLinguSwahili-serengeti-E250-nli-matryoshka)    | sartifyllc     | Yes     | serengeti-E250            | Yes          | 768         |   51.3997 |                   45.4011 |                       77.6318 |                     71.4496 |                       56.4492 |                         61.2038 |                     63.9453 |           63.4926   |                    38.6383 |                   32.6575  | 77.0157 |            19.287 |             9.624 |
| [bert-base-uncased-swahili](https://huggingface.co/flax-community/bert-base-uncased-swahili)                                            | flax-community | Yes     | bert-base-uncased-swahili |              | ?           |   37.8727 |                   41.123  |                       80.8838 |                     66.0714 |                       48.302  |                         51.9334 |                     64.2236 |            0.400601 |                    18.6071 |                    2.04101 | 58.9762 |            13.807 |             8.103 |
| [MultiLinguSwahili-bge-small-en-v1.5-nli-matryoshka](https://huggingface.co/sartifyllc/MultiLinguSwahili-serengeti-E250-nli-matryoshka) | sartifyllc     | Yes     | bge-small-en-v1.5         | Yes          | 256         |   36.3029 |                   35.107  |                       67.3486 |                     54.1597 |                       38.0027 |                         46.8393 |                     51.2305 |            5.01061  |                    21.7986 |                   17.8461  | 62.3059 |            21.521 |            14.465 |
| [bert-base-sw-cased](https://huggingface.co/Geotrend/bert-base-sw-cased)                                                                | Geotrend       | Yes     | bert-base-sw-cased        |              | ?           |   33.6552 |                   35.3342 |                       84.3066 |                     62.3109 |                       36.3685 |                         38.7996 |                     57.9199 |            0.396624 |                    12.9566 |                    6.77267 | 55.6602 |            10.077 |             2.959 |
| [UBC-NLPserengeti-E250](https://huggingface.co/UBC-NLP/serengeti-E250)                                                                  | UCB            | Yes     | UBC-NLPserengeti-E250     |              | ?           |   33.581  |                   44.0374 |                       84.3848 |                     42.1008 |                       37.1957 |                         38.2414 |                     58.1592 |           12.7676   |                    15.7357 |                   14.7948  | 53.3967 |             2.041 |             0.117 |


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
