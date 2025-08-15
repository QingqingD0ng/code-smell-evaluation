import rdflib
from typing import Set, Optional

def find_roots(graph: rdflib.Graph, prop: rdflib.URIRef, roots: Optional[Set[rdflib.term.Node]] = None) -> Set[rdflib.term.Node]:
    if roots is None:
        roots = set()

    for s, p, o in graph.triples((None, prop, None)):
        roots.add(o)
        roots.update(find_roots(graph, prop, roots))

    return roots