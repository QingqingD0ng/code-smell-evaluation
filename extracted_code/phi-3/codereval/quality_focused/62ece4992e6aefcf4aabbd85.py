from typing import Optional, Set
from rdflib import Graph, URIRef

def find_roots(graph: Graph, prop: URIRef, roots: Optional[Set[str]] = None) -> Set[str]:
    if roots is None:
        roots = set()

    for _, _, subject in graph.triples((None, prop, None)):
        if subject not in roots:
            roots.add(subject)
            find_roots(graph, prop, roots)

    return roots