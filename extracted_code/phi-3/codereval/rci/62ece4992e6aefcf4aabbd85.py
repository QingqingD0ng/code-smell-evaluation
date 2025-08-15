import rdflib
from typing import Set, Optional

def find_roots(graph: rdflib.Graph, prop: rdflib.URIRef, roots: Optional[Set[rdflib.URIRef]] = None) -> Set[rdflib.URIRef]:
    if roots is None:
        roots = set()

    for s, p, o in graph.triples((None, prop, None)):
        if s not in roots:
            roots.add(s)
            find_roots(graph, prop, roots)

    return roots