import json
import pandas as pd
import os

# Define the JSONL file path
jsonl_file_path = 'config.jsonl'

# Define the tasks and corresponding metrics
tasks_metrics = {
    "AfriSentiClassification": "accuracy",
    "AfriSentiLangClassification": "accuracy",
    # "LanguageClassification": "accuracy",
    "MasakhaNEWSClassification": "accuracy",
    "MassiveIntentClassification": "accuracy",
    "MassiveScenarioClassification": "accuracy",
    "SwahiliNewsClassification": "accuracy",
    "NTREXBitextMining": "f1",
    "MasakhaNEWSClusteringP2P": "v_measure",
    "MasakhaNEWSClusteringS2S": "v_measure",
    "XNLI": "ap",
    "MIRACLReranking": "MAP@10(MIRACL)",
    "MIRACLRetrieval": "ndcg_at_10"
}

# Function to read JSON file and extract the main score
def read_json(file_path, metric_name, use_train=False):
    with open(file_path, 'r') as file:
        data = json.load(file)
        scores = data.get('scores', {})
        if use_train:
            if 'train' in scores and scores['train']:
                return scores['train'][0].get(metric_name, None)
        else:
            for key in ['test', 'dev', 'validation']:
                if key in scores and scores[key]:
                    if metric_name in scores[key][0]:
                        return scores[key][0][metric_name]
                    # Specific handling for nested structures like in XNLI
                    for sub_metric, value in scores[key][0].items():
                        if isinstance(value, dict) and metric_name in value:
                            return value[metric_name]
    return None

# Function to extract the base model from the directory name
def extract_base_model(directory_name):
    try:
        base_model = directory_name.replace("MultiLinguSwahili-", "").replace("-nli-matryoshka", "")
    except:
        base_model = directory_name
            
    return base_model

# List to hold all data
all_data = []

# Read the JSONL file
with open(jsonl_file_path, 'r') as file:
    for line in file:
        try:
            entry = json.loads(line.strip().replace("'", "\""))
            directory = entry['name']
            base_model = extract_base_model(directory)
            
            # Initialize the data dictionary for each model
            data = {
                "Model Name": [f"[{directory}]({entry['link']})"],
                "Publisher": [entry['publisher']],
                "Open?": ["Yes"],
                "Basemodel": [base_model],
                "Matryoshka": [entry['matryoshka']],
                "Dimension": [entry['dimension']],
            }
            
            # Iterate over the files in the directory and calculate scores
            scores_list = []
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    task_name = filename.replace('.json', '')
                    if task_name in tasks_metrics:
                        metric_name = tasks_metrics[task_name]
                        file_path = os.path.join(directory, filename)
                        use_train = task_name == "SwahiliNewsClassification"
                        score = read_json(file_path, metric_name, use_train=use_train)
                        if score is not None:
                            data[task_name] = [100 * score]
                            scores_list.append(100 * score)
            
            # Calculate the average score
            if scores_list:
                average_score = sum(scores_list) / len(scores_list)
                data["Average"] = [average_score]
            else:
                data["Average"] = [None]
    
            # Add the data to the list
            all_data.append(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e} - Line: {line}")

# Combine all data into a single DataFrame
combined_df = pd.concat([pd.DataFrame(d) for d in all_data], ignore_index=True)

# Ensure the "Average" column is placed before "AfriSentiClassification"
columns_order = ["Model Name", "Publisher", "Open?", "Basemodel", "Matryoshka", "Dimension", "Average"] + list(tasks_metrics.keys())
combined_df = combined_df[columns_order]

# Sort the DataFrame by the "Average" column in descending order
combined_df = combined_df.sort_values(by="Average", ascending=False)

# Display the combined DataFrame
# print(combined_df)

# Save the combined DataFrame to a Markdown file
with open("SECONDARY_README.md", "w") as f:
    f.write(combined_df.to_markdown(index=False))
    
    
    
# Function to combine two markdown files
def combine_markdown_files(file1, file2, output_file, encoding='utf-8'):
    # Read the contents of the first file
    with open(file1, 'r') as f1:
        content1 = f1.read()
    
    # Read the contents of the second file
    with open(file2, 'r', encoding=encoding) as f2:
        content2 = f2.read()
    
    # Combine the contents
    combined_content = content1 + "\n" + content2
    
    # Write the combined content to the output file
    with open(output_file, 'w', encoding=encoding) as out:
        out.write(combined_content)

# Paths to the markdown files
file1 = 'SECONDARY_README.md'
file2 = 'THIRD_README.md'
output_file = 'SECONDARY_README.md'

# Combine the markdown files
combine_markdown_files(file1, file2, output_file)


