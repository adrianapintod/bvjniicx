from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
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
        description="The maximum sequence length to use",
    )
    lora_rank: int = Field(
        default=16,
        description="The rank of the LoRA matrices",
    )
    load_in_4bit: bool = Field(
        default=True,
        description="Whether to load the model in 4-bit mode",
    )

    # Peft settings
    lora_rank: int = Field(
        default=16,
        description="The rank of the LoRA matrices",
    )
    lora_alpha: int = Field(
        default=16,
        description="The alpha of the LoRA matrices",
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
        description="Whether to use gradient checkpointing",
    )
    random_state: int = Field(
        default=3407,
        description="The random state to use",
    )
    use_rslora: bool = Field(
        default=False,
        description="Whether to use RSLoRA",
    )

    # Trainer settings
    per_device_train_batch_size: int = Field(
        default=2,
        description="The batch size to use for training",
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        description="The number of gradient accumulation steps to use",
    )
    warmup_steps: int = Field(
        default=5,
        description="The number of warmup steps to use",
    )
    max_steps: int = Field(
        default=60,
        description="The maximum number of steps to use",
    )
    learning_rate: float = Field(
        default=2e-4,
        description="The learning rate to use",
    )
    logging_steps: int = Field(
        default=1,
        description="The number of logging steps to use",
    )
    optimizer: str = Field(
        default="adamw_8bit",
        description="The optimizer to use",
    )
    weight_decay: float = Field(
        default=0.01,
        description="The weight decay to use",
    )
    lr_scheduler_type: str = Field(
        default="linear",
        description="The learning rate scheduler to use",
    )
    seed: int = Field(
        default=3407,
        description="The seed to use",
    )


settings = Settings()  # type: ignore
