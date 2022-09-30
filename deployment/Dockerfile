
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]