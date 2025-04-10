import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
with open("data/embeddings.json", "r") as f:
    data = json.load(f)

G = nx.Graph()

for item in data:
    img_node = f"IMG::{item['image_name']}"
    img_vec = np.array(item['image_embedding']).reshape(1, -1)

    # Add image node
    G.add_node(img_node, type='image')

    for i, text in enumerate(item['text_candidates']):
        text_node = f"TXT::{text}"
        text_vec = np.array(item['text_embeddings'][i]).reshape(1, -1)

        # Add text node if not already added
        if text_node not in G:
            G.add_node(text_node, type='text')

        # Compute similarity and add edge
        similarity = cosine_similarity(img_vec, text_vec)[0][0]
        if similarity > 0.25:  # you can tune this threshold
            G.add_edge(img_node, text_node, weight=similarity)

print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Optional: Visualize graph
pos = nx.spring_layout(G, seed=42)
labels = {node: node.replace("IMG::", "").replace("TXT::", "") for node in G.nodes}
colors = ['skyblue' if G.nodes[n]['type'] == 'image' else 'lightgreen' for n in G.nodes]

plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, edge_color='gray', font_size=8)
plt.title("CLIP-Based Knowledge Graph")
# plt.show()
plt.savefig('output_plot.png')

