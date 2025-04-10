from pyvis.network import Network

net = Network(height="400px", width="100%", notebook=False)
net.add_node("CLIP", label="CLIP Model")
net.add_node("Concept", label="Extracted Concept")
net.add_edge("CLIP", "Concept", label="predicts")

net.show("test_graph.html")
