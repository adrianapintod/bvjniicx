from typing import Any

from pydantic import BaseModel, PrivateAttr
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

from utils.config import settings


class ModelTrainer(BaseModel):
    __model: PreTrainedModel | None = PrivateAttr(default=None)
    __trainer: SFTTrainer | None = PrivateAttr(default=None)

    def initialize_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: Any,
        dataset: Any,
        output_dir: str = "./data/models/checkpoints",
        report_to: str = "none",
    ) -> SFTTrainer:
        self.__model = FastLanguageModel.get_peft_model(
            model,
            r=settings.lora_rank,
            lora_alpha=settings.lora_alpha,  # Scaling factor for weight updates
            lora_dropout=settings.lora_dropout,
            bias=settings.bias,
            use_gradient_checkpointing=settings.use_gradient_checkpointing,  # type: ignore - Enable long context finetuning
            random_state=settings.random_state,  # Ensure reproducibility of results
            use_rslora=settings.use_rslora,
        )

        training_args = TrainingArguments(
            per_device_train_batch_size=settings.per_device_train_batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            warmup_steps=settings.warmup_steps,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=settings.max_steps,
            learning_rate=settings.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=settings.logging_steps,
            optim=settings.optimizer,
            weight_decay=settings.weight_decay,
            lr_scheduler_type=settings.lr_scheduler_type,
            seed=settings.seed,
            output_dir=output_dir,
            report_to=report_to,
        )

        trainer = SFTTrainer(
            model=self.__model,  # type: ignore
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
        self.__trainer = trainer
        return self.__trainer

    def get_model(self) -> PreTrainedModel:
        return self.__model  # type: ignore

    def get_trainer(self) -> SFTTrainer:
        return self.__trainer  # type: ignore
