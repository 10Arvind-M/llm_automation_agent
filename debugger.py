import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("AIPROXY_TOKEN"))  # Should print your actual key
