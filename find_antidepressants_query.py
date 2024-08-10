from rdflib import Graph, URIRef

g = Graph()
g.parse('C:/Users/david.bogdan/master/disertatie/oregano/oregano-master/oregano-master/Data_OREGANO/Graphs/oreganov2'
        '.1.rdf')

results = g.query(
    "PREFIX oregano: <http://erias.fr/oregano/#> "
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema/#>"
    "SELECT DISTINCT ?compound ?name_compound WHERE { ?disease rdfs:label \"pgkb:depressive disorder\". ?compound "
    "oregano:is_substance_that_treats ?disease. ?compound rdfs:label ?name_compound.}")

print(len(results))

for row in results:
    print(row)
