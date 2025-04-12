# Fine-tuning Llama for SQL

A project for fine-tuning a Llama model to generate SQL query from natural language.

## Description

This project provides tools and utilities for fine-tuning Llama-3.1 models using Unsloth to improve their ability to generate SQL based on natural language requests. The fine-tuning process uses LoRA (Low-Rank Adaptation) to efficiently adapt the model for the SQL generation task.

## Project Structure

```
.
├── data/               # Data storage
│   ├── models/         # Saved checkpointys and fine-tuned models
│   └── volumes/        # Docker volumes
├── notebooks/          # Jupyter notebooks for exploration and testing
├── build/              # Docker compose 
├── src/                # Source code
│   ├── services/       # Core services (data loading and formatting, training)
│   ├── templates/      # Template prompts and model configurations
│   ├── utils/          # Utility functions and application settings
│   └── main.py         # Main entry point
```

## Requirements

- Python 3.12+
- uv
- docker
- Dependencies as listed in `pyproject.toml`
- x86_64 architecture


## How to Install Locally for Development

### Setup

1. Clone the repository:

   ```
   git clone git@github.com:adrianapintod/bvjniicx.git
   cd bvjniicx
   ```

2. Create a virtual environment and Install dependencies:

   ```
   uv sync
   ```

3. Activate vitual environment
   
   ```
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Create a `.env` file based on `.env.dist` with your configuration:
   ```
   cp .env.dist .env
   # Edit .env with your Hugging Face token and other settings
   ```

### Fine-tuning

Run the fine-tuning process:

```
uv run ./src/main.py --output-dir=./data/models/output --fine-tuned-name=my-sql-model
```

## Configuration

The project uses environment variables for configuration:

- `BASE_MODEL`: The base Llama model to fine-tune (default: unsloth/Llama-3.1-8B-unsloth-bnb-4bit)
- `DATASET_NAME`: The dataset to use for fine-tuning (default: gretelai/synthetic_text_to_sql)
- `HF_TOKEN`: Your Hugging Face token for model upload
- `HF_REPO_NAME`: Repository name for pushing your model to Hugging Face

See `.env.dist` for all available configuration options.

## How to Contribute

1. Fork the repository
2. Create a feature branch:
   ```
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Push to your branch and create a pull request

### How to run OpenWebUI uisng the fine-tuned model

To run please first install the `huggingface-hub[cli]`

```Python
uv add "huggingface-hub[cli]"
```

Then give execution permissions to the file `scripts/launch.sh`

```bash
chmod +x scripts/launch.sh
```

Finally, run the script

```bash
./scripts/launch.sh
```

Then you can access the OpenWebUI at `http://localhost:3000`

## Contact

Adriana Pinto luzadriana.pin@gmail.com
