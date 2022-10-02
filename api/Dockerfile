# Base image
# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip3 install --upgrade pip
RUN pip install -r dependencies/requirements.txt

# Run the application
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]