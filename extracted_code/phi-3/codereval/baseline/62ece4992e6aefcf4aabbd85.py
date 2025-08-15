import rdflib

def find_roots(graph: rdflib.Graph, prop: rdflib.URIRef, roots: Optional[set] = None) -> set:
    if roots is None:
        roots = set()

    for s, p, o in graph.triples((None, prop, None)):
        roots.add(s)
        find_roots(graph, prop, roots)

    return roots