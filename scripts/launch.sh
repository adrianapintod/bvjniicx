#!/bin/bash

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi


# Create the directories if it doesn't exist
echo "Creating directories..."	
mkdir -p ./data/models/output && mkdir -p ./data/volumes

# Download the model from the Hugging Face Hub
echo "Downloading the model from the Hugging Face Hub..."
uv run huggingface-cli download $HF_REPO_NAME --local-dir ./data/models/output/$HF_REPO_NAME

# Create the Modelfile for Ollama
echo "Creating the Modelfile for Ollama..."
uv run ./src/scripts/create_modelfile.py --fine-tuned-name $HF_REPO_NAME

# Start the Docker Compose
echo "Starting the Docker Compose..."
docker compose -f ./build/docker-compose.yml up -d

# Create the model in Ollama
echo "Creating the model in Ollama..."
docker exec ollama ollama create $HF_REPO_NAME -f /root/.ollama/host/models/$HF_REPO_NAME/Modelfile

# Restart the OpenWebUI
echo "Restarting the OpenWebUI..."
docker restart open-webui

# Print the OpenWebUI URL
echo "OpenWebUI is running at http://localhost:3000"

# Print the Ollama URL
echo "Ollama is running at http://localhost:11434"
