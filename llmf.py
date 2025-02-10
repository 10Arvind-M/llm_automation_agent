import os
import requests
from dotenv import load_dotenv  

#  Load environment variables
load_dotenv()  

#  Correct AI Proxy URL & Model
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_MODEL = "gpt-4o-mini"  #  Ensure correct model name
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

#  Ensure API Key is loaded
if not AIPROXY_TOKEN:
    raise ValueError("❌ AIPROXY_TOKEN is missing! Ensure it is set correctly in .env.")

def call_llm(prompt):
    """Calls AI Proxy LLM API and returns response."""
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
    "model": "gpt-4o-mini",  # ✅ Ensure this is the ONLY supported model
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.0
    }

    
    #  Debugging: Print request payload (Optional)
    print(f"🔹 Sending Request to AI Proxy: {payload}")
    
    try:
        response = requests.post(AIPROXY_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()  # ✅ Raise error if HTTP status code is not 200
        
        # ✅ Try to parse JSON response safely
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            raise Exception("❌ API returned invalid JSON response.")

        # ✅ Ensure "choices" key exists
        if "choices" not in data or not data["choices"]:
            raise Exception("❌ API Response is missing 'choices'. Full response: " + str(data))

        # ✅ Debugging: Print API Response (Optional)
        print(f"🔹 AI Proxy Response: {data}")

        return data["choices"][0]["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Error: {e}")
        raise Exception(f"LLM call failed: {str(e)}")
