extraInstruction = "If you do not know, say, that you do not know."

#Set LLM


question="Tell me about publication with the title Thomas Manns"
parameters_for_query = """{
  "variable": "title",
  "value": "Thomas Manns"
}"""
#extract_json(model_path, f_path, question)
print(parameters_for_query)

# Step 2: send request to Knowledge graph

# 2. Create query using template
sparql_query = put_data_into_query_template(json.loads(parameters_for_query))

# 3. Sneding request
sparql_endpoint = "https://triplestore1.informatik.tu-chemnitz.de/sparql/"
result = get_from_kg(sparql_query, sparql_endpoint)

# Step 3: creating sub graph
node_FromGraph_tups = []
create_sub_graph(node_FromGraph_tups, result)

# Step 4: RAG implamentation

    # Initialize LLM
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

GENERATIVE_AI_MODEL_REPO_RAG = "TheBloke/stablelm-zephyr-3b-GGUF"
GENERATIVE_AI_MODEL_FILE_RAG = "./stablelm-zephyr-3b.Q4_K_M.gguf"

llm_model_RAG = setup_llama_model(GENERATIVE_AI_MODEL_REPO_RAG, GENERATIVE_AI_MODEL_FILE_RAG,
                                  callback_handler=StreamingStdOutCallbackHandler())

# initialize Graph to connect with llm
graph_store = SimpleGraphStore()
for tup in node_FromGraph_tups:
    subject, predicate, obj = tup
    graph_store.upsert_triplet(subject, predicate, obj)

service_context = ServiceContext.from_defaults(llm=llm_model_RAG, embed_model='local')

storage_context = StorageContext.from_defaults(graph_store=graph_store)


index = KnowledgeGraphIndex(
        [],
        service_context=service_context,
        storage_context=storage_context,
    )

query_engine = index.as_query_engine(
        include_text=False, response_mode="tree_summarize"
    )

response = query_engine.query(
        question + extraInstruction,

    )







#question = "What is the publisher of the book with the title 'Human-Factors Taxonomy'?"
question = "What does 'Human-Factors Taxonomy' address?"
#chat_with_llmAPI(question)
#print(ExtractEntitesViaAPILLM(question))
GENERATIVE_AI_MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
GENERATIVE_AI_MODEL_FILE = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"

model_path = hf_hub_download(
    repo_id=GENERATIVE_AI_MODEL_REPO,
    filename=GENERATIVE_AI_MODEL_FILE
)

f_path = r"C:\Users\daria\PycharmProjects\SendRequest\venv\Resources\json.gbnf"

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
            """Take the examples of output into account: Example 1: Question: Tell me who is a publisher of the publication with the abstract 'This is about nature'
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
  "variable": "title",
  "value": "Evolution of web science"
}

"""

    )
extraInstruction = "If you do not know, say, that you do not know."
#parameters_for_query = extract_json(model_path, f_path, prompt_for_query)
#print(parameters_for_query)


def ExtractEntitesViaAPILLM(question):
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
    {
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
    # "Question:"+ question+

    promptForQuery = "Based on the provided context find the variable and value in the question." + "Question:" + question + "SPARQL-template:" + context + "Note: output the results in JSON format without any extra notes:" + jsonFormat + "Note: variable can only contain properties from DCMI Metadata Terms.In the variable_name output only the name of the property from DCMI Metadata Terms"
    return AccessModelAPI("mistral-7b-instruct-v0.2",
                          "LL-4nklQdzbKARmXLBB8sQg5RM8jtw3lfms7sSe7OnwOHztA2FmtBwwgE0Q274FzeGy",
                          promptForQuery)


def chat_with_llmAPI(question):
    extraInstruction = "If you do not know, say, that you do not know."
    # Step 1: Retrieve variable and its value

    parameters_for_query = ExtractEntitesViaAPILLM(question)
    print(parameters_for_query)
    # Step 2: send request to Knowledge graph

    # 2. Create query using template
    sparql_query = put_data_into_query_template(json.loads(parameters_for_query))

    # 3. Sneding request
    sparql_endpoint = "https://triplestore1.informatik.tu-chemnitz.de/sparql/"
    result = get_from_kg(sparql_query, sparql_endpoint)

    # Step 3: creating sub graph
    node_FromGraph_tups = []
    create_sub_graph(node_FromGraph_tups, result)

    # Step 4: RAG implamentation

    # Initialize LLM
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    GENERATIVE_AI_MODEL_REPO = "TheBloke/stablelm-zephyr-3b-GGUF"
    GENERATIVE_AI_MODEL_FILE = "./stablelm-zephyr-3b.Q4_K_M.gguf"

    # initialize Graph to connect with llm
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.llama_api import LlamaAPI
    llm_model = LlamaAPI(
        api_key="LL-4nklQdzbKARmXLBB8sQg5RM8jtw3lfms7sSe7OnwOHztA2FmtBwwgE0Q274FzeGy",
        model="mistral-7b-instruct-v0.2"
    )
    embed_model = OpenAIEmbedding(model="text-embedding-3-small"),
    service_context = ServiceContext.from_defaults(llm=llm_model, embed_model='local')

    graph_store = SimpleGraphStore()
    for tup in node_FromGraph_tups:
        subject, predicate, obj = tup
        graph_store.upsert_triplet(subject, predicate, obj)

    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex(
        [],
        service_context=service_context,
        storage_context=storage_context,
    )

    query_engine = index.as_query_engine(
        include_text=False, response_mode="tree_summarize"
    )

    response = query_engine.query(
        question + extraInstruction,

    )
    return str(response)
# function which uses local models to retrieve data from Sentence


from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from ExtraFunction import setup_llama_model
from ExtraFunctionKGCommunication import get_from_kg

from ExtraFunctionsRAG import create_sub_graph, AccessModelAPI

from ExtraFunctionForEntitiesExtraction import extract_json,put_data_into_query_template, AccessModelAPILammaCpp
import json


from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore

from flask import Flask, request, jsonify

from openai import OpenAI

app = Flask(__name__)


def chat_with_llm(question):

    # Step 1: Retrieve variable and its value
    GENERATIVE_AI_MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    GENERATIVE_AI_MODEL_FILE = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    model_path = hf_hub_download(
        repo_id=GENERATIVE_AI_MODEL_REPO,
        filename=GENERATIVE_AI_MODEL_FILE
    )

    f_path = r"C:\Users\daria\PycharmProjects\SendRequest\venv\Resources\json.gbnf"


    extraInstruction = "If you do not know, say, that you do not know."
    parameters_for_query = extract_json(model_path, f_path, question)
    print(parameters_for_query)

    # Step 2: send request to Knowledge graph

    # 2. Create query using template
    sparql_query = put_data_into_query_template(json.loads(parameters_for_query))

    # 3. Sneding request
    sparql_endpoint = "https://triplestore1.informatik.tu-chemnitz.de/sparql/"
    result = get_from_kg(sparql_query, sparql_endpoint)


    # Step 3: creating sub graph
    node_FromGraph_tups = []
    create_sub_graph(node_FromGraph_tups, result)

    # Step 4: RAG implamentation

    # Initialize LLM
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    GENERATIVE_AI_MODEL_REPO = "TheBloke/stablelm-zephyr-3b-GGUF"
    GENERATIVE_AI_MODEL_FILE = "./stablelm-zephyr-3b.Q4_K_M.gguf"

    llm_model = setup_llama_model(GENERATIVE_AI_MODEL_REPO, GENERATIVE_AI_MODEL_FILE,
                                  callback_handler=StreamingStdOutCallbackHandler())

    # initialize Graph to connect with llm
    service_context = ServiceContext.from_defaults(llm=llm_model, embed_model='local')

    graph_store = SimpleGraphStore()
    for tup in node_FromGraph_tups:
        subject, predicate, obj = tup
        graph_store.upsert_triplet(subject, predicate, obj)

    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex(
        [],
        service_context=service_context,
        storage_context=storage_context,
    )

    query_engine = index.as_query_engine(
        include_text=False, response_mode="tree_summarize"
    )

    response = query_engine.query(
        question + extraInstruction,

    )
    return str(response)




@app.route('/chat', methods=['POST'])
def ask():
    input_question = request.json.get('question')
    if input_question:
        answer = chat_with_llm(input_question)
        return answer
    else:
        return 'No question provided', 400



question = "tell me what 'Human-Factors Taxonomy' is about?"
chat_with_llm(question)
"""
if __name__ == '__main__':
    app.run(debug=True)
"""


from openai import OpenAI
def AccessModelAPILammaCpp(model_name,api_key,prompt,grammar_path):
  client=OpenAI(
    api_key=api_key,
    base_url="https://api.llama-api.com"
  )
  grammar_text = read_file_to_string(grammar_path)
  grammar = LlamaGrammar.from_string(grammar_text)

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