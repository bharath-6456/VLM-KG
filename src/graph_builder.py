from pyvis.network import Network
import networkx as nx

def build_reasoning_graph(image_path):
    G = nx.Graph()
    G.add_node("cat")
    G.add_node("sofa")
    G.add_node("laptop")
    G.add_edge("cat", "sofa", label="sitting on")
    G.add_edge("laptop", "sofa", label="on")

    net = Network(height="600px", width="100%", notebook=False)
    net.from_nx(G)
    for edge in net.edges:
        edge["title"] = edge["label"]
    net.show("reasoning_graph.html")
