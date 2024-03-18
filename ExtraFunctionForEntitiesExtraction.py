#ExtraFunctionForEntitiesExtraction

from llama_cpp import Llama,LlamaGrammar
from ExtraFunction import read_file_to_string
from langchain.chains import create_tagging_chain
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain_openai import OpenAI

#query template for put_data_into_query_template function
query_full_template="""
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX dce: <http://purl.org/dc/elements/1.1/>
PREFIX bibo: <http://purl.org/ontology/bibo/>
PREFIX tucbib: <https://tucbib.tu-chemnitz.de/schema/>

SELECT DISTINCT ?publication (GROUP_CONCAT(DISTINCT ?creator; SEPARATOR=";") as ?authors) (GROUP_CONCAT(DISTINCT ?title; SEPARATOR="<!|!>") as ?titles) ?issn  ?available ?issued ?abstract ?type ?publisher
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

def extract_json(model_path,grammar_path,question,n_gpu_layers=64,n_ctx=4096,max_tokens=2096):
    #initialize

    llama2_model = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx
    )
    # proccess grammar file
    grammar_text = read_file_to_string(grammar_path)
    grammar = LlamaGrammar.from_string(grammar_text)
    context = """
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT  ?title ?available ?abstract ?accessRights ?accrualMethod ?accrualPeriodicity ?accrualPolicy ?alternative ?audience ?bibliographicCitation ?conformsTo ?contributor ?coverage ?created ?creator ?date ?dateAccepted ?dateCopyrighted ?dateSubmitted ?description ?educationLevel ?extent ?format ?hasFormat ?hasPart ?hasVersion ?identifier ?instructionalMethod ?isFormatOf ?isPartOf ?isReferencedBy ?isReplacedBy ?isRequiredBy ?issued ?isVersionOf ?language ?license ?mediator ?medium ?modified ?provenance ?publisher ?references ?relation ?replaces ?requires ?rights ?rightsHolder ?source ?spatial ?subject ?tableOfContents ?temporal ?type ?valid

    WHERE {
      ?publication dct:title ?title ;
                   dct:available ?available .

      OPTIONAL { ?publication dct:abstract ?abstract }
      OPTIONAL { ?publication dct:accessRights ?accessRights }
      OPTIONAL { ?publication dct:accrualMethod ?accrualMethod }
      ...

      FILTER (REGEX(?variable, "value"))
    }


    """
    json_format = """
   {
      "variable": "<variable_name>",
      "value": "<search_keyword>"
    }
    if there are several values:
      {
      "variable": "<variable_name>",
      "value": "<search_keyword>"
    }
    {
      "variable": "<variable_name>",
      "value": "<search_keyword>"
    }
    ...

    """
    prompt_for_query = (
            "Your are SPARQL-query expert. Based on the provided context find the variable and value in the question." +
            "Question: " + question + "SPARQL-template: " + context +
            "Note: output the results in JSON format:" + json_format +
            "Note: variable can only contain properties from DCMI Metadata Terms."+
            """Take the examples of output into account: Example 1: {
  "variable": "abstract",
  "value": "keyword"
}

Example 2:
{
  "variable": "creator",
  "value": "John Doe"
}


"""

    )

    #calling LLM
    response = llama2_model(prompt=prompt_for_query, grammar=grammar, max_tokens=max_tokens)

    return response['choices'][0]['text']


def put_data_into_query_template(json_pair):
    query_template = """
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT ?publication ?title ?available ?abstract ?accessRights ?accrualMethod ?accrualPeriodicity ?accrualPolicy ?alternative ?audience ?bibliographicCitation ?conformsTo ?contributor ?coverage ?created ?creator ?date ?dateAccepted ?dateCopyrighted ?dateSubmitted ?description ?educationLevel ?extent ?format ?hasFormat ?hasPart ?hasVersion ?identifier ?instructionalMethod ?isFormatOf ?isPartOf ?isReferencedBy ?isReplacedBy ?isRequiredBy ?issued ?isVersionOf ?language ?license ?mediator ?medium ?modified ?provenance ?publisher ?references ?relation ?replaces ?requires ?rights ?rightsHolder ?source ?spatial ?subject ?tableOfContents ?temporal ?type ?valid

    WHERE {
      ?publication dct:title ?title ;
                   dct:available ?available .

      OPTIONAL { ?publication dct:abstract ?abstract }
      OPTIONAL { ?publication dct:accessRights ?accessRights }
      OPTIONAL { ?publication dct:accrualMethod ?accrualMethod }
      OPTIONAL { ?publication dct:accrualPeriodicity ?accrualPeriodicity }
      OPTIONAL { ?publication dct:accrualPolicy ?accrualPolicy }
      OPTIONAL { ?publication dct:alternative ?alternative }
      OPTIONAL { ?publication dct:audience ?audience }
      OPTIONAL { ?publication dct:bibliographicCitation ?bibliographicCitation }
      OPTIONAL { ?publication dct:conformsTo ?conformsTo }
      OPTIONAL { ?publication dct:contributor ?contributor }
      OPTIONAL { ?publication dct:coverage ?coverage }
      OPTIONAL { ?publication dct:created ?created }
      OPTIONAL { ?publication dct:creator ?creator }
      OPTIONAL { ?publication dct:date ?date }
      OPTIONAL { ?publication dct:dateAccepted ?dateAccepted }
      OPTIONAL { ?publication dct:dateCopyrighted ?dateCopyrighted }
      OPTIONAL { ?publication dct:dateSubmitted ?dateSubmitted }
      OPTIONAL { ?publication dct:description ?description }
      OPTIONAL { ?publication dct:educationLevel ?educationLevel }
      OPTIONAL { ?publication dct:extent ?extent }
      OPTIONAL { ?publication dct:format ?format }
      OPTIONAL { ?publication dct:hasFormat ?hasFormat }
      OPTIONAL { ?publication dct:hasPart ?hasPart }
      OPTIONAL { ?publication dct:hasVersion ?hasVersion }
      OPTIONAL { ?publication dct:identifier ?identifier }
      OPTIONAL { ?publication dct:instructionalMethod ?instructionalMethod }
      OPTIONAL { ?publication dct:isFormatOf ?isFormatOf }
      OPTIONAL { ?publication dct:isPartOf ?isPartOf }
      OPTIONAL { ?publication dct:isReferencedBy ?isReferencedBy }
      OPTIONAL { ?publication dct:isReplacedBy ?isReplacedBy }
      OPTIONAL { ?publication dct:isRequiredBy ?isRequiredBy }
      OPTIONAL { ?publication dct:issued ?issued }
      OPTIONAL { ?publication dct:isVersionOf ?isVersionOf }
      OPTIONAL { ?publication dct:language ?language }
      OPTIONAL { ?publication dct:license ?license }
      OPTIONAL { ?publication dct:mediator ?mediator }
      OPTIONAL { ?publication dct:medium ?medium }
      OPTIONAL { ?publication dct:modified ?modified }
      OPTIONAL { ?publication dct:provenance ?provenance }
      OPTIONAL { ?publication dct:publisher ?publisher }
      OPTIONAL { ?publication dct:references ?references }
      OPTIONAL { ?publication dct:relation ?relation }
      OPTIONAL { ?publication dct:replaces ?replaces }
      OPTIONAL { ?publication dct:requires ?requires }
      OPTIONAL { ?publication dct:rights ?rights }
      OPTIONAL { ?publication dct:rightsHolder ?rightsHolder }
      OPTIONAL { ?publication dct:source ?source }
      OPTIONAL { ?publication dct:spatial ?spatial }
      OPTIONAL { ?publication dct:subject ?subject }
      OPTIONAL { ?publication dct:tableOfContents ?tableOfContents }
      OPTIONAL { ?publication dct:temporal ?temporal }
      OPTIONAL { ?publication dct:title ?title }
      OPTIONAL { ?publication dct:type ?type }
      OPTIONAL { ?publication dct:valid ?valid }



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


def Call_LLM_via_API(question):
    #initialize prompt
    # send a request to LLM to construct query
    context = """
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT ?publication ?title ?available ?abstract ?accessRights ?accrualMethod ?accrualPeriodicity ?accrualPolicy ?alternative ?audience ?bibliographicCitation ?conformsTo ?contributor ?coverage ?created ?creator ?date ?dateAccepted ?dateCopyrighted ?dateSubmitted ?description ?educationLevel ?extent ?format ?hasFormat ?hasPart ?hasVersion ?identifier ?instructionalMethod ?isFormatOf ?isPartOf ?isReferencedBy ?isReplacedBy ?isRequiredBy ?issued ?isVersionOf ?language ?license ?mediator ?medium ?modified ?provenance ?publisher ?references ?relation ?replaces ?requires ?rights ?rightsHolder ?source ?spatial ?subject ?tableOfContents ?temporal ?type ?valid

    WHERE {
      ?publication dct:title ?title ;
                   dct:available ?available .

      OPTIONAL { ?publication dct:abstract ?abstract }
      OPTIONAL { ?publication dct:accessRights ?accessRights }
      OPTIONAL { ?publication dct:accrualMethod ?accrualMethod }
      ...

      FILTER (REGEX(?variable, "value"))
    }


    """
    jsonFormat = """
    Answer: {
      "variable": "<variable_name>",
      "value": "<search_keyword>"
    }
    if there are several values:
      Answer: {
      "variable": "<variable_name>",
      "value": "<search_keyword>"
    }
    {
      "variable": "<variable_name>",
      "value": "<search_keyword>"
    }

    """
    examples = """Take the examples of output into account: Example 1: Question: Tell me who is a publisher of the publication with the abstract 'This is about nature'
                Output:{
      "variable": "abstract",
      "value": "This is about nature"
    }

    Example 2:
    Question: 'tell me the title of the most recent publication, created by John Doe'
    Output:
    {
      "variable": "creator",
      "value": "John Doe"
    }
    Example 3:
    Question: 'tell me when the book Web Engineering was published'
    Output:
    {
      "variable": "title",
      "value": "Web Engineering"
    }
    Example 4:
    Question: 'List all the names of all authors who contributed to the book written by James Wang'
    Output:
    {
      "variable": "creator",
      "value": "James Wang"
    }
    Example 5:
    Question: 'what the main focus of 'Evolution of Web Science' '
    Output:
    {
      "variable": "creator",
      "value": "James Wang"
    }

    """

    prompt_For_Query = "Your are SPARQL-query expert. Based on the provided context find the variable and value in the question." + "Question:" + question + "SPARQL-template:" + context + "Note: output the results in JSON format:" + jsonFormat + "Note: variable can only contain properties from DCMI Metadata Terms." + examples
    generate_schema = {
        "type": "object",
        "properties": {
            "variable": {
                "type": "string",
                "enum": [
                    "publication",
                    "title",
                    "available",
                    "abstract",
                    "accessRights",
                    "accrualMethod",
                    "accrualPeriodicity",
                    "accrualPolicy",
                    "alternative",
                    "audience",
                    "bibliographicCitation",
                    "conformsTo",
                    "contributor",
                    "coverage",
                    "created",
                    "creator",
                    "date",
                    "dateAccepted",
                    "dateCopyrighted",
                    "dateSubmitted",
                    "description",
                    "educationLevel",
                    "extent",
                    "format",
                    "hasFormat",
                    "hasPart",
                    "hasVersion",
                    "identifier",
                    "instructionalMethod",
                    "isFormatOf",
                    "isPartOf",
                    "isReferencedBy",
                    "isReplacedBy",
                    "isRequiredBy",
                    "issued",
                    "isVersionOf",
                    "language",
                    "license",
                    "mediator",
                    "medium",
                    "modified",
                    "provenance",
                    "publisher",
                    "references",
                    "relation",
                    "replaces",
                    "requires",
                    "rights",
                    "rightsHolder",
                    "source",
                    "spatial",
                    "subject",
                    "tableOfContents",
                    "temporal",
                    "type",
                    "valid"
                ],
                "description": " Relevant variable from the DCMI Metadata Terms (DCMI Metaterms) properties list within user queries."
            },
            "value": {
                "type": "string",
                "description": "Value correlated to the variable without any extra quotes."
            }
        },
        "required": ["variable", "value"]
    }



    #call api

    api_key = "LL-4nklQdzbKARmXLBB8sQg5RM8jtw3lfms7sSe7OnwOHztA2FmtBwwgE0Q274FzeGy"
    llama = LlamaAPI(api_key)
    model = ChatLlamaAPI(client=llama)

    chain = create_tagging_chain(generate_schema, model)
    response=chain.run(prompt_For_Query)
    return response


