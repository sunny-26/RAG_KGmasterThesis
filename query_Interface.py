# set the communication with the LLM via api, to access via other UIs (local server establishment)
import requests
import json

def ask_llm_via_https_requests(question):
    url="http://127.0.0.1:5000/chat"

    request_body = {"question": question}

    json_body = json.dumps(request_body)


    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json_body,headers=headers,timeout=600)
    if response.ok:
        print(response.text)
    else:
        print(f"""Error: {response.status_code}""")








