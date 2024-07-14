# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:49:19 2024

@author: michael.mollel@sartify.com, msamwelmollel@gmail.com
"""

from flask import Flask, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# HTML template with embedded JavaScript
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>t-SNE Embedding Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="plot"></div>
    <div id="info">
        <h3>Selected Point:</h3>
        <p id="selected-text"></p>
        <h3>Nearest Neighbors:</h3>
        <ol id="neighbors"></ol>
    </div>

    <script>
        fetch('/get_data')
            .then(response => response.json())
            .then(data => {
                const trace = {
                    x: data.x,
                    y: data.y,
                    z: data.z,
                    mode: 'markers',
                    type: 'scatter3d',
                    text: data.text,
                    hoverinfo: 'text',
                    marker: {
                        size: 5,
                        color: data.z,
                        colorscale: 'Viridis',
                        opacity: 0.8
                    }
                };

                const layout = {
                    title: 't-SNE Visualization of Embeddings',
                    scene: {
                        xaxis: { title: 't-SNE 1' },
                        yaxis: { title: 't-SNE 2' },
                        zaxis: { title: 't-SNE 3' }
                    },
                    height: 800,
                    width: 1000
                };

                Plotly.newPlot('plot', [trace], layout);

                document.getElementById('plot').on('plotly_click', function(data) {
                    const point = data.points[0];
                    const index = point.pointIndex;
                    const text = point.text;

                    document.getElementById('selected-text').textContent = text;

                    fetch(`/get_neighbors/${index}`)
                        .then(response => response.json())
                        .then(neighborData => {
                            const neighborsList = document.getElementById('neighbors');
                            neighborsList.innerHTML = '';
                            neighborData.neighbors.forEach(neighbor => {
                                const li = document.createElement('li');
                                li.textContent = neighbor;
                                neighborsList.appendChild(li);
                            });
                        });
                });
            });
    </script>
</body>
</html>
'''

# Load and preprocess data
data = pd.read_csv("data.csv")
# data = data.head(1000)
column_name = 'content'
data = data.dropna(subset=[column_name])
data = data.drop_duplicates(subset=[column_name])
data["query_question"] = data[column_name]

# Load the embedding model
model = SentenceTransformer('sartifyllc/MultiLinguSwahili-bge-small-en-v1.5-nli-matryoshka')

# Generate embeddings
embeddings = model.encode(data['query_question'].tolist())

# Perform t-SNE
tsne = TSNE(n_components=3, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/get_data')
def get_data():
    return jsonify({
        'x': tsne_results[:, 0].tolist(),
        'y': tsne_results[:, 1].tolist(),
        'z': tsne_results[:, 2].tolist(),
        'text': data['query_question'].tolist()
    })

@app.route('/get_neighbors/<int:index>')
def get_neighbors(index):
    k = 5  # Number of neighbors to return
    distances = np.sum((tsne_results - tsne_results[index])**2, axis=1)
    nearest_indices = distances.argsort()[1:k+1]  # Exclude the point itself
    return jsonify({
        'neighbors': data.iloc[nearest_indices]['query_question'].tolist()
    })

if __name__ == '__main__':
    print("Starting the server... This may take a moment as the data is being processed.")
    print("Once the server is running, open a web browser and go to http://127.0.0.1:5000/")
    app.run(debug=True)