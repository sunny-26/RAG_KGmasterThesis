#ExtraFunctionsRAG
import requests
from rdflib import Graph, Literal, URIRef, Namespace, BNode
from openai import OpenAI

def create_sub_graph(node_FromGraph_tups,json_data):
  for binding in json_data['results']['bindings']:
    for var, value in binding.items():
      if value['type'] == 'literal':
        tup = (binding['publication']['value'], var, value['value'])
      elif value['type'] == 'uri':
        tup = (binding['publication']['value'], var, f"URI: {value['value']}")
      elif value['type'] == 'typed-literal':
        tup = (binding['publication']['value'], var, f"{value['datatype']}: {value['value']}")
      else:
        tup = (binding['publication']['value'], var, str(value))
      node_FromGraph_tups.append(tup)




def AccessModelAPI(model_name,api_key,prompt):
  client=OpenAI(
    api_key=api_key,
    base_url="https://api.llama-api.com"
  )

  response = client.chat.completions.create(
    model=model_name,

    messages=[
      {"role": "system",
       "content": "You extract value and variable from the sentence. You can provide a response only in the following format: {\"variable\": <variable_name>, \"value\": <search_keyword>}"},
      {"role": "user", "content": prompt}
    ],
    #force model output in json format
    functions=[
      {
        "name": "retrieve_metadata",
        "description": "Retrieve variable and value from the sentence based on DCMI Metadata Terms properties",
        "parameters": {
          "type": "object",
          "properties": {
            "variable": {
              "type": "string",
              "description": "Property form the list:publication, title, available, abstract, accessRights, accrualMethod, accrualPeriodicity, accrualPolicy, alternative, audience, bibliographicCitation, conformsTo, contributor, coverage, created, creator, date, dateAccepted, dateCopyrighted, dateSubmitted, description, educationLevel, extent, format, hasFormat, hasPart, hasVersion, identifier, instructionalMethod, isFormatOf, isPartOf, isReferencedBy, isReplacedBy, isRequiredBy, issued, isVersionOf, language, license, mediator, medium, modified, provenance, publisher, references, relation, replaces, requires, rights, rightsHolder, source, spatial, subject, tableOfContents, temporal, type, valid",

            },
            "value": {
              "type": "string",
              "description": "Corresponding to the variable values, e.g Who is author of Lost Love: variable: Lost love"
            }

          },
          "required": ["variable", "value"],
        },
      }
    ],
    function_call="retrieve_metadata",

  )

  return response.choices[0].message.content



