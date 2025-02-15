import os
import subprocess
import json
import sqlite3
import requests
import pandas as pd
import markdown
import csv
from datetime import datetime
from dateutil import parser
from pathlib import Path
from difflib import get_close_matches
from dotenv import load_dotenv
from fastapi import HTTPException
from flask import Flask, request, jsonify
from PIL import Image,ImageEnhance
import speech_recognition as sr
from llmf import call_llm
import re
from typing import Optional, Dict, Any
import sys
import pytesseract
import numpy as np
from dotenv import load_dotenv
import duckdb
from bs4 import BeautifulSoup
import shutil

# ✅ Load environment variables
load_dotenv()

try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT = os.getcwd()
print("PROJECT_ROOT:", PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
print("DATA_DIR:", DATA_DIR)

### **TASK IDENTIFICATION**
def identify_task(task_description):
    """Use LLM to map a user task description to a predefined task ID."""
    prompt = f"""
    You are an AI automation assistant responsible for mapping user-provided task descriptions to predefined task IDs.

    Below is a list of task IDs and their corresponding descriptions:

    {{"A1": "Install uv and run the script from a remote URL with user email."}},
    {{"A2": "Format markdown file using Prettier."}},
    {{"A3": "Count the number of Wednesdays in a text file."}},
    {{"A4": "Sort contacts in JSON by last and first name."}},
    {{"A5": "Extract first lines from the most recent log files."}},
    {{"A6": "Extract H1 titles from markdown files and create an index."}},
    {{"A7": "Extract the sender's email from an email file."}},
    {{"A8": "Extract credit card number from an image using LLM."}},
    {{"A9": "Find the most similar comments using embeddings."}},
    {{"A10": "Compute total sales for 'Gold' ticket type in a database."}},
    {{"B1": "Ensure data outside '/data' is never accessed."}},
    {{"B2": "Ensure no file deletion occurs."}},
    {{"B3": "Fetch and save data from an API."}},
    {{"B4": "Clone a Git repository and make a commit."}},
    {{"B5": "Execute a SQL query on a SQLite/DuckDB database."}},
    {{"B6": "Extract (scrape) data from a website."}},
    {{"B7": "Compress or resize an image."}},
    {{"B8": "Transcribe audio from an MP3 file."}},
    {{"B9": "Convert Markdown to HTML."}},
    {{"B10": "Filter a CSV file and return JSON data."}}

    Match the following task description to its **best** predefined task ID.

    User Task Description:
    "{task_description}"

    Return only a valid JSON object that is strictly parsable with following sample structure:
    {{
        "task_id": "<Matching Task ID or 'UNKNOWN'>"
    }}
    Strictly do not include any extra charecters in the output JSON other than the defined structure.
    """

    response = call_llm(prompt)
    
    # ✅ Ensure response is valid JSON
    try:
        task_data = json.loads(response)
        return task_data.get("task_id")
    except json.JSONDecodeError:
        return "UNKNOWN"


### **TASK EXECUTION HANDLER**
def execute_task(request):
    task = request.task  # ✅ Extract task from JSON body    
    print(f"Executing Task: {task}")

    task_id = identify_task(task)
    print(f"Task ID identified by LLM: {task_id}")  # ✅ Debugging line

    if task_id == "A1":
        USER_EMAIL = request.body
        return install_uv_and_run_datagen(USER_EMAIL)
    elif task_id == "A2":
        return format_markdown()
    elif task_id == "A3":
        return count_weekdays()
    elif task_id == "A4":
        return sort_contacts()
    elif task_id == "A5":
        return extract_recent_logs()
    elif task_id == "A6":
        return extract_markdown_titles()
    elif task_id == "A7":
        return extract_email()
    elif task_id == "A8":
        return extract_credit_card_number()
    elif task_id == "A9":
        return find_similar_comments()
    elif task_id == "A10":
        return calculate_gold_ticket_sales()
    elif task_id == "B3":
        if not request.url or not request.output_file or not request.generated_prompt:
            return {"error": "Missing required parameters: 'url', 'output_file', or 'generated_prompt'"}

        # Ensure output file is saved in DATA_DIR
        output_file_path = os.path.join(DATA_DIR, request.output_file)

        return fetch_api_data(request.url, output_file_path, request.generated_prompt, request.params)
    
    elif task_id == "B4":
        if not request.repo_url or not request.commit_message:
            return {"error": "Missing required parameters: 'repo_url' or 'commit_message'"}

        # Set the repository output directory inside DATA_DIR
        repo_name = os.path.basename(request.repo_url).replace(".git", "")
        output_dir = os.path.join(DATA_DIR, repo_name)

        return clone_and_commit(request.repo_url, output_dir, request.commit_message)

    elif task_id == "B5":
        if not request.database_file or not request.query or not request.output_file:
            return {"error": "Missing required parameters: 'database_file', 'query', or 'output_file'"}

        # Ensure database and output file paths are inside DATA_DIR
        database_path = os.path.join(DATA_DIR, request.database_file)
        output_path = os.path.join(DATA_DIR, request.output_file)

        return execute_sql_query(database_path, request.query, output_path, request.is_sqlite)

    elif task_id == "B6":
        url = request.url  
        output_file = request.output_file  
        if not url or not output_file:
            return {"error": "Missing required parameters: 'url' and 'output_file'"}
        
        return scrape_website(url, output_file)
    elif task_id == "B7":
        if not request.input_file or not request.output_file:
            return {"error": "Missing required parameters: 'input_file' or 'output_file'"}
        input_path = os.path.join(DATA_DIR, request.input_file)
        output_path = os.path.join(DATA_DIR, request.output_file)
        return compress_image(input_path, output_path, request.quality)
    elif task_id == "B8":
        if not request.input_file or not request.output_file:
            return {"error": "Missing required parameters: 'input_file' or 'output_file'"}
        input_path = os.path.join(DATA_DIR, request.input_file)
        output_path = os.path.join(DATA_DIR, request.output_file)
        return transcribe_audio(input_path, output_path)
    elif task_id == "B9":
        if not request.input_file or not request.output_file:
            return {"error": "Missing required parameters: 'input_file' or 'output_file'"}
        input_path = os.path.join(DATA_DIR, request.input_file)
        output_path = os.path.join(DATA_DIR, request.output_file)
        return convert_markdown_to_html(input_path, output_path)
    elif task_id == "B10":
        if not request.input_file or not request.output_file or not request.column or not request.value:
            return {"error": "Missing required parameters: 'input_file', 'output_file', 'column', or 'value'"}
        input_path = os.path.join(DATA_DIR, request.input_file)
        output_path = os.path.join(DATA_DIR, request.output_file)
        return filter_csv(input_path, request.column, request.value, output_path)

    else:
        return f"Error: Task ID {task_id} not recognized."



### **SECURITY FUNCTIONS (B1 & B2)**
def is_safe_path(path):
    """Ensure the path is within /data directory to prevent unauthorized access."""
    abs_path = os.path.realpath(path)
    abs_data_dir = os.path.realpath(DATA_DIR)
    return abs_path.startswith(abs_data_dir)


def safe_open(path, mode="r"):
    """Safely open a file, ensuring it exists within the /data directory."""
    if not is_safe_path(path):
        raise PermissionError(f"Access outside {DATA_DIR} is not allowed: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")
    return open(path, mode, encoding="utf-8")


def safe_remove(path):
    """Prevent file deletion by raising an error."""
    raise PermissionError("File deletion is not allowed.")

# ✅ Prevent file deletion globally
os.remove = safe_remove


### **PHASE A TASKS (Operations)**
import subprocess
import requests
import os

import subprocess
import requests
import os

import subprocess
import requests
import os
import sys

def install_uv_and_run_datagen(user_email):
    if not user_email:
        raise ValueError("User email is required to run datagen.py")
    
    try:
        # Install uv globally
        subprocess.run(["pip", "install", "--quiet", "uv"], check=True)

        # Download datagen.py into the project root
        url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        script_path = os.path.join(PROJECT_ROOT, "datagen.py")
        print("script_path:", script_path)
        response = requests.get(url)
        response.raise_for_status()
        with open(script_path, "wb") as f:
            f.write(response.content)

        # Determine which Python executable to use:
        python_executable = None
        CUR_ENV = os.getenv("CUR_ENV", "0")  # Default to "0" if CUR_ENV is not set
        if CUR_ENV == "1":
            # Running in Docker: use system Python
            python_executable = "python"
            # Optionally, set DATA_DIR to /app/data if needed
            DATA_DIR = os.path.join("/app", "data")
        else:
            # Running locally: use virtual environment if available.
            # Try to get VIRTUAL_ENV; if not present, default to PROJECT_ROOT/venc.
            venv_path = os.environ.get("VIRTUAL_ENV") or os.path.join(PROJECT_ROOT, "venc")
            print("venv_path:", venv_path)
            if os.name == "nt":
                python_executable = os.path.join(venv_path, "Scripts", "python.exe")
            else:
                python_executable = os.path.join(venv_path, "bin", "python")
        
        print("python_executable:", python_executable)
        
        # When not running in Docker, verify that the python_executable exists.
        if (CUR_ENV != "1") and (not os.path.exists(python_executable)):
            return ("Error: Virtual environment not found. "
                    "Ensure that your virtual environment 'venc' is activated or exists at "
                    f"{os.path.join(PROJECT_ROOT, 'venc')}.")

        # Install dependencies using the chosen Python executable
        subprocess.run([python_executable, "-m", "pip", "install", "--quiet", "faker", "numpy", "pandas"], check=True)

        # Run the datagen.py script using the selected Python executable
        result = subprocess.run(
            [python_executable, script_path, user_email],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )

        if result.returncode == 0:
            return f"Data generation completed: {result.stdout.strip()}"
        else:
            return f"Error: {result.stderr.strip()}"

    except Exception as e:
        return f"Error: {e}"





def format_markdown():
    """Formats 'format.md' using Prettier"""
    md_file = os.path.join(DATA_DIR, "format.md")
    
    # Ensure the file exists
    if not os.path.exists(md_file):
        return {"error": f"File not found: {md_file}"}
    
    try:
        # Use shutil.which to find npx on the system's PATH
        npx_path = shutil.which("npx")
        if not npx_path:
            return {"error": "npx not found. Ensure Node.js is installed and npx is in your PATH."}
        
        # Run Prettier using npx (version 3.4.2)
        subprocess.run([npx_path, "prettier@3.4.2", "--write", md_file], check=True)
        return {"message": f"Formatted {md_file} successfully"}
    
    except subprocess.CalledProcessError as e:
        return {"error": f"Prettier formatting failed: {e}"}

def count_weekdays():
    """Counts the number of Wednesdays in 'dates.txt' and writes the result to 'dates-wednesdays.txt'."""
    dates_file = os.path.join(DATA_DIR, "dates.txt")
    output_file = os.path.join(DATA_DIR, "dates-wednesdays.txt")

    try:
        with open(dates_file, "r") as f:
            count = sum(1 for line in f if parser.parse(line.strip()).weekday() == 2)

        with open(output_file, "w") as f:
            f.write(str(count))

        return {"message": "Count of Wednesdays written successfully", "count": count}

    except FileNotFoundError:
        return {"error": f"File not found: {dates_file}"}

    except ValueError as e:
        return {"error": f"Invalid date format in file: {str(e)}"}

def sort_contacts():
    with open(f"{DATA_DIR}/contacts.json") as f:
        contacts = json.load(f)
    contacts.sort(key=lambda x: (x["last_name"], x["first_name"]))
    with open(f"{DATA_DIR}/contacts-sorted.json", "w") as f:
        json.dump(contacts, f, indent=2)
    return "Task A4 completed"

def extract_recent_logs():
    log_files = sorted(os.listdir(f"{DATA_DIR}/logs"), key=lambda f: os.path.getmtime(f"{DATA_DIR}/logs/{f}"), reverse=True)[:10]
    with open(f"{DATA_DIR}/logs-recent.txt", "w") as f:
        for log in log_files:
            with open(f"{DATA_DIR}/logs/{log}") as lf:
                f.write(lf.readline())
    return "Task A5 completed"

def extract_markdown_titles():
    """Extracts H1 titles from all Markdown files in /data/docs/** and saves to index.json"""
    docs_dir = DATA_DIR / "docs"
    index = {}

    if not docs_dir.exists():
        return {"error": f"Directory not found: {docs_dir}"}

    for md_file in docs_dir.rglob("*.md"):  # Recursively find .md files in all subfolders
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("# "):  # First H1 found
                        relative_path = md_file.relative_to(docs_dir)  # e.g., "above/file.md"
                        index[str(relative_path)] = line.strip("# ").strip()
                        break  # Stop reading after first H1

        except Exception as e:
            return {"error": f"Failed to read {md_file}: {str(e)}"}

    # Save output to /data/docs/index.json
    index_file = docs_dir / "index.json"
    try:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        return {"message": f"Index file created at {index_file}"}

    except Exception as e:
        return {"error": f"Failed to write index.json: {str(e)}"}


def extract_email():
    with open(f"{DATA_DIR}/email.txt", "r") as f:
        email_content = f.read()
    sender_email = call_llm(f"Extract the sender’s email from:\n\n{email_content}")
    with open(f"{DATA_DIR}/email-sender.txt", "w") as f:
        f.write(sender_email.strip())
    return "Task A7 completed"

# Define project root and data directory as Path objects.

# Define the image and output file paths.

IMAGE_PATH = os.path.join(DATA_DIR, "credit_card.png")
OUTPUT_FILE = os.path.join(DATA_DIR, "credit_card.txt")

# Optionally set the Tesseract command if needed (e.g., on Windows).
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_credit_card_number():
    try:
        # Open the image and convert it to grayscale.
        image = Image.open(IMAGE_PATH).convert("L")
    except Exception as e:
        return f"Error: Image not found or cannot be read. {e}"
    
    # Enhance contrast.
    image = ImageEnhance.Contrast(image).enhance(2)
    # Enhance sharpness.
    image = ImageEnhance.Sharpness(image).enhance(2)
    # Resize the image for better OCR accuracy.
    image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
    
    # Use Tesseract OCR with a custom configuration: 
    # psm 6 (assume a uniform block of text) and only whitelist digits.
    custom_config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    extracted_text = pytesseract.image_to_string(image, config=custom_config).strip()
    
    print("Extracted text:", extracted_text)  # Debug output
    
    # Use regex to find a 16-digit credit card number (allowing for spaces or dashes)
    pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    match = re.search(pattern, extracted_text)
    
    if match:
        card_number = match.group().replace(" ", "").replace("-", "")
    else:
        return f"Error: No valid credit card number found. Extracted text: '{extracted_text}'"
    
    # Write the extracted card number to the output file.
    try:
        with open ("credit_card.txt", "w") as f:
            f.write(card_number)
    except Exception as e:
        return f"Error writing to output file: {e}"
    
    return f"Credit card number extracted successfully: {card_number}"


   


### **IMPROVED TASK A9: Find Similar Comments**
def find_similar_comments():
    """Finds the two most similar comments using embeddings."""
    file_path = f"{DATA_DIR}/comments.txt"
    
    # ✅ Check if file exists
    if not os.path.exists(file_path):
        return "Error: comments.txt not found."
    
    with open(file_path, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]

    if len(comments) < 2:
        return "Not enough comments to compare."

    # ✅ Generate embeddings using OpenAI API
    payload = {
        "model": "text-embedding-3-small",
        "input": comments
    }
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
        headers={"Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}",
                 "Content-Type": "application/json"},
        json=payload
    )
    
    if response.status_code != 200:
        return f"Embedding API failed: {response.json()}"

    # ✅ Ensure embeddings are extracted correctly
    embedding_data = response.json()
    embeddings = [entry["embedding"] for entry in embedding_data.get("data", [])]

    if len(embeddings) != len(comments):
        return "Error: Embedding count mismatch with comments."

    # ✅ Find the most similar pair
    similarity_scores = []
    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            similarity = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
            similarity_scores.append((comments[i], comments[j], similarity))

    # ✅ Sort by similarity (highest first)
    similarity_scores.sort(key=lambda x: x[2], reverse=True)

    # ✅ Get the most similar pair
    most_similar_pair = similarity_scores[0][:2]

    # ✅ Write the most similar comments to a file
    with open(f"{DATA_DIR}/comments-similar.txt", "w", encoding="utf-8") as f:
        f.writelines("\n".join(most_similar_pair))

    return "Task A9 completed"
def calculate_gold_ticket_sales():
    conn = sqlite3.connect(f"{DATA_DIR}/ticket-sales.db")
    cur = conn.cursor()
    cur.execute("SELECT SUM(units * price) FROM tickets WHERE type='Gold'")
    total = cur.fetchone()[0] or 0
    conn.close()
    with open(f"{DATA_DIR}/ticket-sales-gold.txt", "w") as f:
        f.write(str(total))
    return "Task A10 completed"

### **PHASE B TASKS (Business & Security)**

### ✅ TASK B1 & B2: Enforce Security Constraints
def is_safe_path(path):
    """Ensure the path is within /data directory."""
    abs_path = os.path.abspath(path)
    abs_data_dir = os.path.abspath(DATA_DIR)
    return abs_path.startswith(abs_data_dir)

def safe_open(path, mode="r"):
    """Open a file safely within /data directory."""
    if not is_safe_path(path):
        raise PermissionError(f"Access outside {DATA_DIR} is not allowed: {path}")
    return open(path, mode, encoding="utf-8")

def safe_remove(path):
    """Prevent file deletion by raising an error."""
    raise PermissionError("File deletion is not allowed.")

# Monkey-patch os.remove to prevent deletion (B2)
os.remove = safe_remove

'''def fetch_api_data():
    response = requests.get("https://jsonplaceholder.typicode.com/posts")
    with open(f"{DATA_DIR}/api_data.json", "w") as f:
        f.write(response.text)
    return "Task B3 completed"


def clone_and_commit():
    repo_url = "https://github.com/your-username/your-repo.git"  # Change this
    repo_path = f"{DATA_DIR}/repo"

    # ✅ Clone only if the repo does not exist
    if not os.path.exists(repo_path):
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)

    # ✅ Change directory and configure git user
    subprocess.run(["git", "-C", repo_path, "config", "user.name", "Your Name"], check=True)
    subprocess.run(["git", "-C", repo_path, "config", "user.email", "your-email@example.com"], check=True)

    # ✅ Check if there are changes before committing
    status_output = subprocess.run(["git", "-C", repo_path, "status", "--porcelain"], capture_output=True, text=True)
    
    if not status_output.stdout.strip():
        return "No changes to commit."

    # ✅ Add changes, commit, and push
    subprocess.run(["git", "-C", repo_path, "add", "."], check=True)
    subprocess.run(["git", "-C", repo_path, "commit", "-m", "Automated commit"], check=True)

    # ✅ Ensure `origin` is set correctly
    subprocess.run(["git", "-C", repo_path, "remote", "set-url", "origin", repo_url], check=True)

    # ✅ Push changes (Ensure GitHub credentials are configured)
    subprocess.run(["git", "-C", repo_path, "push", "origin", "main"], check=True)

    return "Task B4 completed"

def execute_sql_query():
    conn = sqlite3.connect(f"{DATA_DIR}/sample.db")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    result = cur.fetchone()[0]
    conn.close()
    with open(f"{DATA_DIR}/sql_output.txt", "w") as f:
        f.write(str(result))
    return "Task B5 completed"

def scrape_website():
    response = requests.get("https://example.com")
    with open(f"{DATA_DIR}/scraped_data.html", "w") as f:
        f.write(response.text)
    return "Task B6 completed"

def compress_image():
    img = Image.open(f"{DATA_DIR}/input.jpg")
    img.save(f"{DATA_DIR}/compressed.jpg", "JPEG", quality=50)
    return "Task B7 completed"

import speech_recognition as sr

def transcribe_audio():
    recognizer = sr.Recognizer()
    
    # Load the audio file
    with sr.AudioFile("data/test.wav") as source:
        audio_data = recognizer.record(source)
    
    try:
        # Perform speech recognition
        text = recognizer.recognize_google(audio_data)  # Uses Google's API
        with open("data/audio-transcript.txt", "w") as f:
            f.write(text)
        return "Task B8 completed"
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "Could not request results, check your internet connection"



def convert_markdown_to_html():
    with open(f"{DATA_DIR}/format.md") as f:
        md_text = f.read()
    html_text = markdown.markdown(md_text)
    with open(f"{DATA_DIR}/document.html", "w") as f:
        f.write(html_text)
    return "Task B9 completed"

def filter_csv():
    df = pd.read_csv(f"{DATA_DIR}/data.csv")
    filtered_df = df[df["ide"] < 0.08]
    filtered_df.to_json(f"{DATA_DIR}/filtered_data.json", orient="records", indent=2)
    return "Task B10 completed"
'''
# Fetch data from an API and save it
def fetch_api_data(url: str, output_file: str,generated_prompt: str ,params: Optional[Dict[str, Any]] = None):
    """
    This tool function fetches data from an API using a GET request and saves the response to a JSON file. It also tries POST if GET fails with some params. Example 1: URL: "https://api.example.com/users" Output File: "users.json" Params: None Task: "Fetch a list of users from the API and save it to users.json." Task: Fetch a list of users from the API and save it to users.json. Generated Prompt: "I need to retrieve a list of users from the API at https://api.example.com/users and save the data in JSON format to a file named users.json.  Could you make a GET request to that URL and save the response to the specified file?" Example 2: URL: "https://api.example.com/products" Output File: "products.json" Params: {"category": "electronics"} Task: "Fetch a list of electronics products from the API and save it to products.json." Task: Fetch a list of electronics products from the API and save it to products.json. Generated Prompt: "I'm looking for a list of electronics products. The API endpoint is https://api.example.com/products.  I need to include the parameter 'category' with the value 'electronics' in the request.  Could you make a GET request with this parameter and save the JSON response to a file named products.json?" Example 3: URL: "https://api.example.com/items" Output File: "items.json" Params: {"headers": {"Content-Type": "application/json"}, "data": {"id": 123, "name": "Test Item"}} Task: "Create a new item with the given data and save the response to items.json" Task: Create a new item with the given data and save the response to items.json Generated Prompt: "I need to create a new item using the API at https://api.example.com/items.  The request should be a POST request. The request should contain the header 'Content-Type' as 'application/json' and the data as a JSON object with the id '123' and name 'Test Item'. Save the JSON response to a file named items.json." Args: url (str): The URL of the API endpoint. output_file (str): The path to the output file where the data will be saved. params (Optional[Dict[str, Any]]): The parameters to include in the request. Defaults to None. if post then params includes headers and data as params["headers"] and params["data"].
    Args:
        url (str): The URL of the API endpoint.
        output_file (str): The path to the output file where the data will be saved.
        generated_prompt (str): The prompt to generate from the task.
        params (Optional[Dict[str, Any]]): The parameters to include in the request. Defaults to None. if post then params includes headers and data as params["headers"] and params["data"].
        
    """   
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)
        return {"message": f"POST request successful. Data saved to {output_file}"}

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
    try:
        response = requests.post(url, params["headers"], params["data"])
        response.raise_for_status()
        data = response.json()
        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)
        return {"message": f"GET request successful. Data saved to {output_file}"}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")

#Clone a git repo and make a commit
def clone_and_commit(repo_url: str, output_dir: str, commit_message: str):
    """
    Clones a Git repository into DATA_DIR, makes a commit, and pushes the changes.
    """
    try:
        # Clone the repository
        subprocess.run(["git", "clone", repo_url, output_dir], check=True)

        # Add all files and make a commit
        subprocess.run(["git", "add", "."], cwd=output_dir, check=True)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=output_dir, check=True)
        
        return {"message": f"Repository cloned and committed successfully at {output_dir}"}

    except subprocess.CalledProcessError as e:
        return {"error": f"An error occurred: {e}"}

#Run a SQL query on a SQLite or DuckDB database
def execute_sql_query(database_file: str, query: str, output_file: str, is_sqlite: bool = True):
    """
    Executes a SQL query on a SQLite or DuckDB database and writes the result to an output file.
    """
    try:
        conn = sqlite3.connect(database_file) if is_sqlite else duckdb.connect(database_file)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()

        # Ensure output file is saved in DATA_DIR
        with open(output_file, "w") as file:
            for row in result:
                file.write(str(row) + "\n")

        return {"message": f"Query executed successfully. Results saved to {output_file}"}

    except (sqlite3.Error, duckdb.Error) as e:
        return {"error": f"An error occurred: {e}"}

    finally:
        conn.close()

#Extract data from (i.e. scrape) a website
def scrape_website(url: str, output_file: str):
    """Scrape the website and save the response to a file in DATA_DIR."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Save the file in DATA_DIR
    output_path = os.path.join(DATA_DIR, output_file)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(soup.prettify())

    return {"message": f"Scraped data saved to {output_path}"}
#Compress or resize an image
def compress_image(input_file: str, output_file: str, quality: int = 50):
    """
    Compresses an image and saves it.
    """
    try:
        img = Image.open(input_file)
        img.save(output_file, quality=quality, optimize=True)
        return {"message": f"Image compressed successfully. Saved to {output_file}"}
    except Exception as e:
        return {"error": f"Failed to compress image: {e}"}

#Transcribe audio from an MP3 file
def transcribe_audio(input_file: str, output_file: str):
    """
    Transcribes audio from an MP3 file (Placeholder function).
    """
    try:
        transcript = "Transcribed text (placeholder)"  # Replace with actual transcription logic
        with open(output_file, "w") as file:
            file.write(transcript)
        return {"message": f"Audio transcribed successfully. Saved to {output_file}"}
    except Exception as e:
        return {"error": f"Failed to transcribe audio: {e}"}
#Convert Markdown to HTML
def convert_markdown_to_html(input_file: str, output_file: str):
    """
    Converts a Markdown file to HTML.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            html_content = markdown.markdown(file.read())
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(html_content)
        return {"message": f"Markdown converted to HTML successfully. Saved to {output_file}"}
    except Exception as e:
        return {"error": f"Failed to convert Markdown to HTML: {e}"}

#Write an API endpoint that filters a CSV file and returns JSON data
def filter_csv(input_file: str, column: str, value: str, output_file: str):
    """
    Filters a CSV file based on a specific column value and saves the result in JSON format.
    """
    try:
        results = []
        with open(input_file, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get(column) == value:
                    results.append(row)

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        return {"message": f"CSV filtered successfully. Results saved to {output_file}"}
    except Exception as e:
        return {"error": f"Failed to filter CSV: {e}"}