import networkx as nx
from pyvis.network import Network

def build_graph_and_html(objects, question):
    G = nx.DiGraph()

    for obj in objects:
        G.add_node(obj)

    # Example: link all to "person" for now
    if "person" in objects:
        for obj in objects:
            if obj != "person":
                G.add_edge("person", obj, label="interacts_with")

    net = Network(height="400px", width="100%", directed=True)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    net_html = net.generate_html()

    return net_html
