from pyvis.network import Network
import networkx as nx

# Reuse the graph from previous step
G = nx.Graph()
G.add_node("cat")
G.add_node("sofa")
G.add_node("laptop")
G.add_node("person")

G.add_edge("cat", "sofa", label="sitting on")
G.add_edge("person", "laptop", label="using")
G.add_edge("cat", "person", label="near")
G.add_edge("laptop", "sofa", label="on")

# Create PyVis network
net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=False)

# Convert from NetworkX to PyVis
net.from_nx(G)

# Show labels on edges
for edge in net.edges:
    edge["title"] = edge["label"]

# Save and view
net.show("reasoning_graph.html")
