from rdflib import Graph, URIRef

from typing import Set, Optional


def find_roots(graph: Graph, prop: URIRef, roots: Optional[Set["Node"]] = None) -> Set["Node"]:
	if roots is None:
		roots = set()

	for s, p, o in graph.triples((None, prop, None)):
		if s not in roots:
			roots.add(s)
			find_roots(graph, prop, roots)

	return roots