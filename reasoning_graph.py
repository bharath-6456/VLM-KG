import networkx as nx
import matplotlib.pyplot as plt

# Visual + Text Concepts (replace with your output)
visual_concepts = ["cat", "sofa", "laptop"]
text_concepts = ["cat", "person", "working", "laptop"]

# Step 1: Merge concepts and init graph
all_concepts = set(visual_concepts + text_concepts)
G = nx.Graph()

# Step 2: Add all concepts as nodes
for concept in all_concepts:
    G.add_node(concept)

# Step 3: Add sample reasoning edges (custom logic)
# In real use case, you'd use dependency parsing or CLIP attention weights

G.add_edge("cat", "sofa", label="sitting on")
G.add_edge("person", "laptop", label="using")
G.add_edge("cat", "person", label="near")
G.add_edge("laptop", "sofa", label="on")

# Step 4: Visualize
pos = nx.spring_layout(G, seed=42)
edge_labels = nx.get_edge_attributes(G, 'label')

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1500, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
plt.title("Reasoning Graph from Concepts")
plt.savefig('put.png')
