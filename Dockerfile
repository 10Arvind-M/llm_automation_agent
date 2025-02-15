# Use the official Python 3.9 slim base image (we're targeting Python 3.9.21)
FROM python:3.9-slim

# Set environment variables
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LANG=C.UTF-8
ENV PYTHON_VERSION=3.9.21
ENV PYTHON_SHA256=3126f59592c9b0d798584755f2bf7b081fa1ca35ce7a6fea980108d752a05bb1
ENV GPG_KEY=E3FF2839C048B25C084DEBE9B26995E310250568

# Set working directory inside the container
WORKDIR /app

# Update package lists and install required system dependencies:
# - python3-distutils: for building wheels
# - curl: for downloading files
RUN apt-get update && apt-get install -y \
    python3-distutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment in /app/venv
RUN python -m venv /app/venv

# Set the PATH to use the virtual environment's Python
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip, setuptools, and wheel to support Python 3.9.21
RUN pip install --upgrade pip setuptools wheel

# Copy the requirements file first (to leverage Docker cache) and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code (including the data folder) into /app/
COPY . /app/

# (Optional) If your project is a Python package and needs installation from the source,
# this second pip install may trigger a large download if there are heavy dependencies.
RUN pip install --no-cache-dir .

# Expose port 8000 so the container can be accessed on this port
EXPOSE 8000

# Set the default command to run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
