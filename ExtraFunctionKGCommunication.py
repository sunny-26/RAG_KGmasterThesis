#retrieve json object from KG based on the created query
import requests
from urllib.parse import urlencode





def get_from_kg(sparql_query, sparql_endpoint):
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
    """ + "LIMIT 20"
    return sparql_query

