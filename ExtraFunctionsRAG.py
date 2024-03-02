#ExtraFunctionsRAG
import requests
from rdflib import Graph, Literal, URIRef, Namespace, BNode


def create_sub_graph(node_FromGraph_tups,json_data):
  for binding in json_data['results']['bindings']:
    for var, value in binding.items():
      if value['type'] == 'literal':
        tup = (binding['title']['value'], var, value['value'])
      elif value['type'] == 'uri':
        tup = (binding['title']['value'], var, f"URI: {value['value']}")
      elif value['type'] == 'typed-literal':
        tup = (binding['title']['value'], var, f"{value['datatype']}: {value['value']}")
      else:
        tup = (binding['title']['value'], var, str(value))
      node_FromGraph_tups.append(tup)


