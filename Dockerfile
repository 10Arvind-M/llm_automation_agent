# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install flask

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the API server
CMD ["python", "app.py"]
