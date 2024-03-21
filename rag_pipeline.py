from huggingface_hub import hf_hub_download
import json

import os
from enrich.retrieval import LLM_via_API,put_data_into_query_template, get_from_kg
from enrich.augumentation import create_sub_graph, extract_keywords,\
    extract_synonyms_based_on_graph_predicates,construct_mini_graph
from enrich.response_generation import adapt_extracted_data,construct_system_prompt_for_rag,execute_rag
from enrich.support_functions import extract_json_from_llm_response
from dotenv import load_dotenv

def RAG_Pipeline(question):
    #load_dotenv()
    #1. Entity and Entity type extraction
    success = False  # value to ensure successful retrival of data fromKG
    counter = 0
    while success is False and counter < 6:
    #1.1 Retrieve data from the sentence using LLM
         entity_and_entity_type=LLM_via_API(question)
         print(entity_and_entity_type)
    #1.2 Ensure that the result in json format: if no-convert to json
         entities_in_json= json.dumps(extract_json_from_llm_response(entity_and_entity_type)[0])
         print(entities_in_json)
    #2 Construct SPARQL-Query sing template
         sparql_query = put_data_into_query_template(json.loads(entities_in_json))
         #print(sparql_query)
    #3 Send request to KG
         sparql_endpoint = "https://triplestore1.informatik.tu-chemnitz.de/sparql/"
         results_from_kg = get_from_kg(sparql_query, sparql_endpoint)

    # 4.Convert data to subgraph
         node_FromGraph_tups = []
         create_sub_graph(node_FromGraph_tups, results_from_kg)
         if node_FromGraph_tups:
              success = True

         else:
              counter += 1
         print(node_FromGraph_tups)
         if counter > 5:
             response = "It seems like I do not possess this knowledge."
             return

    # 5. Extract keywords from Subgraph
    keywords = extract_keywords(question)
    #print(keywords)
    properties = ["title", "authors", "issn", "available", "issued", "abstract", "type", "publisher", "keywords",
                  "description", "language"]
    extracted_keywords_for_kg = extract_synonyms_based_on_graph_predicates(keywords, properties)
    #print(extracted_keywords_for_kg)

    extracted_data_for_context = construct_mini_graph(node_FromGraph_tups, extracted_keywords_for_kg)
   #print(extracted_data_for_context)
    # 6. Construct Prompt

    adjusted_data = adapt_extracted_data(extracted_data_for_context)
    #print(adjusted_data)
    prompt_for_rag = construct_system_prompt_for_rag(f"""{adjusted_data}""")
    print(prompt_for_rag)

    # 7. RAG with question
    GENERATIVE_AI_MODEL_REPO = os.getenv('GENERATIVE_AI_MODEL_REPO')
    GENERATIVE_AI_MODEL_FILE = os.getenv('GENERATIVE_AI_MODEL_FILE')

    model_path = hf_hub_download(
        repo_id=GENERATIVE_AI_MODEL_REPO,
        filename=GENERATIVE_AI_MODEL_FILE
    )

    final_response=execute_rag(model_path,question,prompt_for_rag)
    print("User:"+question)
    print(final_response)
