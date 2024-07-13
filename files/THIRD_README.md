## 🧪 Evaluation
To evaluate a model on the Swahili Embeddings Text Benchmark, you can use the following Python script:
```python
pip install mteb
pip install sentence-transformers
import mteb
from sentence_transformers import SentenceTransformer

model_name = "MultiLinguSwahili-serengeti-E250-nli-matryoshka"	
publisher = "sartifyllc"
models = ["sartifyllc/MultiLinguSwahili-bert-base-sw-cased-nli-matryoshka", f"{publisher}/{model_name}"]

for model_name in models:
    truncate_dim = 768
    language = "swa"
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # if cuda available
    # model = SentenceTransformer(model_name, truncate_dim = truncate_dim, device = device, trust_remote_code=True) # if you want to use matryoshka n dimension
    model = SentenceTransformer(model_name, device = device, trust_remote_code=True)
    
    tasks = [
        mteb.get_task("AfriSentiClassification", languages = ["swa"]),
        mteb.get_task("AfriSentiLangClassification", languages = ["swa"]),
        # "LanguageClassification": "accuracy",
        mteb.get_task("MasakhaNEWSClassification", languages = ["swa"]),
        mteb.get_task("MassiveIntentClassification", languages = ["swa"]),
        mteb.get_task("MassiveScenarioClassification", languages = ["swa"]),
        mteb.get_task("SwahiliNewsClassification", languages = ["swa"]),
        # mteb.get_tasks(task_types=["PairClassification", "Reranking", "BitextMining", "Clustering", "Retrieval"], languages = ["swa"]),
    ]
    
    
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=f"{model_name}")
    
    # results = evaluation.run(model, output_folder=f"{model_name}")
    tasks = mteb.get_tasks(task_types=["PairClassification", "Reranking", "BitextMining", "Clustering", "Retrieval"], languages = ["swa"])
    
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=f"{model_name}")
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