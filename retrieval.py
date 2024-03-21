# 1) the functions for retrieval data from Knowledge graph are stored
# 2) the functions for subgraph construction

from openai import OpenAI as openai
import os
import requests
from urllib.parse import urlencode
from support_functions import api_model
from dotenv import load_dotenv
#region Preparation to query main KG

def LLM_via_API(question): # extraction of  entity and entity_type from the sentence
    client = api_model()

    #load_dotenv()
    # The system message (instruction) sets the behavior of the assistant.
    instruction = """Perform entity extraction based on a given property list,
     extracting values corresponding to specific entity types. "

    Property list: [
                        "publication",
                        "title",
                        "available",
                        "abstract",
                        "bibliographicCitation",
                        "issn",
                        "issued",
                        "type",
                        "publisher",
                        "creator",
                        "subject",
                        "description",
                        "language",

                    ]

    Output JSON in format: {"variable": "entity type","value": "entity"}
    Use provided examples to understand the logic:
    Example 1: Question: Tell me who is a publisher of the publication with the abstract 'This is about nature'
                    Output:{
          "variable": "abstract",
          "value": "This is about nature"
        }

          Example 2:
        Question: 'Please provide a list of publications related to Web Composition. '
        Output:
        {
          "variable": "subject",
          "value": "Web Composition"
        }
        Example 3:
        Question: 'tell me the title of the most recent publication, created by John Doe'
        Output:
        {
          "variable": "creator",
          "value": "John Doe"
        }
        Example 4:
        Question: 'tell me when the book 'Supporting the Evolution' was published'
        Output:
        {
          "variable": "title",
          "value": "Supporting the Evolution"
        }
        Example 5:
        Question: 'List all the names of all authors who contributed to the book written by John Smith'
        Output:
        {
          "variable": "creator",
          "value": "John Smith"
        }
        Example 5:
        Question: 'name the authors of Supporting the Evolution'
        Output:
        {
          "variable": "title",
          "value": "Supporting the Evolution"
        }
Return only the answers where both variable and value are not empty.
    """

    completion = client.chat.completions.create(
        model=os.getenv('MODEL_API'),
        response_format={"type": "json_object"},
        seed=12345,
        max_tokens=50,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": question}
        ]

    )

    return completion.choices[0].message.content

#endregion


#region Query main Knowledge graph

def put_data_into_query_template(json_pair): # function to get data from main Knowledge graph
    query_template = """
    PREFIX dct: <http://purl.org/dc/terms/>
PREFIX dce: <http://purl.org/dc/elements/1.1/>
PREFIX bibo: <http://purl.org/ontology/bibo/>
PREFIX tucbib: <https://tucbib.tu-chemnitz.de/schema/>

SELECT DISTINCT ?publication (GROUP_CONCAT(DISTINCT ?creator; SEPARATOR=";") as ?authors) (GROUP_CONCAT(DISTINCT ?title; SEPARATOR="<!|!>") as ?title) ?issn  ?available ?issued ?abstract ?type ?publisher
(GROUP_CONCAT(DISTINCT ?subject ; SEPARATOR=";") as ?keywords)
?description ?language
WHERE {
  ?publication dct:title ?title.
OPTIONAL{?publication tucbib:orderedAuthor ?creator.}
 OPTIONAL { ?publication dct:abstract ?abstract }
  OPTIONAL {?publication bibo:doi ?doi.}

  OPTIONAL {?publication dce:subject ?subject.}
  OPTIONAL {?publication dce:issn ?issn.}
OPTIONAL {  ?publication dct:issued ?issued.}
OPTIONAL {?publication tucbib:orderedAuthor ?creator.}
OPTIONAL { ?publication dct:publisher ?publisher.}
OPTIONAL {?publication dct:type ?type.}
OPTIONAL {?publication dct:available ?available .}
OPTIONAL { ?publication dct:description ?description }
OPTIONAL { ?publication dct:language ?language }
    """
    filter_part = 'FILTER (REGEX(?{}, "{}"))'

    sparql_query = query_template + filter_part.format(json_pair["variable"], json_pair["value"]) + "}"
    """
    quotation_mark = '"'
    variable = json_pair["variable"]
    #value = quotation_mark + json_pair["value"] + quotation_mark
    value =  json_pair["value"]
    sparql_query = query_template.format(variable=variable, value=value)
    """
    return sparql_query


def get_from_kg(sparql_query, sparql_endpoint):#retrieve response from KG as json based on the created query
  encoded_query = urlencode(
    {'default-graph-uri': '', 'query': sparql_query, 'format': 'application/json', 'timeout': '0', 'signal_void': 'on'})

  complete_url = f"{sparql_endpoint}?{encoded_query}"

  try:

    response = requests.get(complete_url)

    if response.status_code == 200:
      content = response.json()  # Convert response to JSON
      return content
    else:
      return None
  except Exception as e:
    print(f"Error: {e}")
    return None


#endregion