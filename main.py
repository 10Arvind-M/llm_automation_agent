from fastapi import FastAPI, HTTPException, Query
import sys
from tasks import execute_task
import utilsf
import os
from dotenv import load_dotenv
from pydantic import BaseModel  # ✅ Ensure request body parsing

# ✅ Load environment variables
load_dotenv()
if not os.getenv("AIPROXY_TOKEN"):
    raise ValueError("AIPROXY_TOKEN is missing in .env")

app = FastAPI()

# ✅ Define request body schema
class TaskRequest(BaseModel):
    task: str
    body: str | None = None


@app.post("/run")
async def run_task(request: TaskRequest):
    """
    Parse and execute a given task.
    """
    #print(f"Request: {request}")  # ✅ Print the request
    #task = request.get("task")  # ✅ Extract task from JSON body
    #if request.get("body"):
        #body = request.get("body")  # ✅ Extract body from JSON body
    #print(f"Executing Task: {task}")
    #print(f"Body: {body}")
    try:        
        result = execute_task(request)
        return {"status": "success", "result": result}
    except ValueError as e:
        print(f"Task Error: {e}")  # ✅ Print the error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_message = traceback.format_exc()  # ✅ Get full traceback
        print(f"Internal Server Error: {error_message}")  # ✅ Print error details
        raise HTTPException(status_code=500, detail=error_message)  # ✅ Send full error in response

@app.get("/read")
async def read_file(path: str = Query(..., description="File path to read")):
    """
    Retrieve file content if allowed.
    Ensures the file path is safe before reading.
    """
    if not path:
        raise HTTPException(status_code=400, detail="File path is required")

    if not utilsf.is_safe_path(path):
        raise HTTPException(status_code=403, detail="Access denied")

    content = utilsf.read_file(path)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")

    return {"content": content}
