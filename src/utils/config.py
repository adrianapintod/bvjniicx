"""
This module contains the Settings class, which is used to load the application settings.
"""

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.

    Attributes:
        hf_token (SecretStr): The token to use for the Hugging Face API.
        hf_repo_name (str): The name of the Hugging Face repository.
        ollama_host (HttpUrl): The URL of the Ollama server.
        base_model (str): The base model to use.
        dataset_name (str): The name of the dataset to use.
        max_seq_length (int): The maximum context length the model can learn.
        lora_rank (int): Controls the number of low-rank factors used for adaptation. Number > 0.
        load_in_4bit (bool): Whether to load the model in 4-bit mode.
        lora_alpha (int): Scaling factor for weight updates.
        lora_dropout (int): The dropout of the LoRA matrices.
        bias (str): The bias of the LoRA matrices.
        use_gradient_checkpointing (str): Whether to use gradient checkpointing. Enable long context fine-tuning.
        random_state (int): The random state to use. Ensure reproducibility of results.
        use_rslora (bool): Whether to use RSLoRA.
        per_device_train_batch_size (int): The batch size per device accelerator core/CPU for training.
        gradient_accumulation_steps (int): The number of updates steps to accumulate the gradients for, before updating a backward/update pass.
        warmup_steps (int): The number of steps used for a linear warmup from 0 to learning_rate.
        max_steps (int): If set to a positive number, the total number of training steps to perform.
        learning_rate (float): The initial learining rate for AdamW optimizer.
        logging_steps (int): The number of updates steps between two logs.
        optimizer (str): The optimizer to use.
        weight_decay (float): The weight decay to apply to all layers except bias and LayerNorm weights in AdamW optimizer.
        lr_scheduler_type (str): The learning rate scheduler to use.
        seed (int): Random seed that will be set at the beginning of training to ensure reproducibility across runs.
    """

    # Hugging Face settings
    hf_token: SecretStr = Field(description="The token to use for the Hugging Face API")
    hf_repo_name: str = Field(description="The name of the Hugging Face repository")

    # Ollama settings
    ollama_host: HttpUrl = Field(description="The URL of the Ollama server")

    # Unsloth settings
    base_model: str = Field(
        default="unsloth/Llama-3.1-8B-unsloth-bnb-4bit",
        description="The base model to use",
    )

    # Dataset settings
    dataset_name: str = Field(
        default="gretelai/synthetic_text_to_sql",
        description="The name of the dataset to use",
    )

    # Model settings
    max_seq_length: int = Field(
        default=2048,
        description="The maximum context length the model can learn.",
    )
    lora_rank: int = Field(
        default=16,
        description="Controls the number of low-rank factors used for adaptation. Number > 0.",
    )
    load_in_4bit: bool = Field(
        default=True,
        description="Whether to load the model in 4-bit mode",
    )

    # Peft settings
    lora_alpha: int = Field(
        default=16,
        description="Scaling factor for weight updates.",
    )
    lora_dropout: int = Field(
        default=0,
        description="The dropout of the LoRA matrices",
    )
    bias: str = Field(
        default="none",
        description="The bias of the LoRA matrices",
    )
    use_gradient_checkpointing: str = Field(
        default="unsloth",
        description="Whether to use gradient checkpointing. Enable long context fine-tuning.",
    )
    random_state: int = Field(
        default=3407,
        description="The random state to use. Ensure reproducibility of results.",
    )
    use_rslora: bool = Field(
        default=False,
        description="Whether to use RSLoRA",
    )

    # Trainer settings
    per_device_train_batch_size: int = Field(
        default=2,
        description="The batch size per device accelerator core/CPU for training.",
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        description="The number of updates steps to accumulate the gradients for, before updating a backward/update pass.",
    )
    warmup_steps: int = Field(
        default=5,
        description="The number of steps used for a linear warmup from 0 to learning_rate.",
    )
    max_steps: int = Field(
        default=60,
        description="If set to a positive number, the total number of training steps to perform.",
    )
    learning_rate: float = Field(
        default=2e-4,
        description="The initial learining rate for AdamW optimizer.",
    )
    logging_steps: int = Field(
        default=1,
        description="The number of updates steps between two logs.",
    )
    optimizer: str = Field(
        default="adamw_8bit",
        description="The optimizer to use",
    )
    weight_decay: float = Field(
        default=0.01,
        description="The weight decay to apply to all layers except bias and LayerNorm weights in AdamW optimizer.",
    )
    lr_scheduler_type: str = Field(
        default="linear",
        description="The learning rate scheduler to use",
    )
    seed: int = Field(
        default=3407,
        description="Random seed that will be set at the beginning of training to ensure reproducibility across runs.",
    )


settings = Settings()  # type: ignore
