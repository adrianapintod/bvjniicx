"""
This module contains the main.py file, which is used to fine-tune a model.
"""

import os
from string import Template

import click

from services.data import FineTuneDataset
from services.train import ModelTrainer
from utils.config import settings
from utils.models import loaded_model, tokenizer


@click.command()
@click.option(
    "--output-dir",
    help="Where to save the fine-tuned model",
    type=click.Path(exists=True),
)
@click.option(
    "--fine-tuned-name",
    help="Name of the model to fine-tune",
    type=str,
)
def main(
    output_dir: str,
    fine_tuned_name: str,
):
    """
    Main function to fine-tune a model.

    Args:
        output_dir (str): The directory to save the fine-tuned model.
        fine_tuned_name (str): The name of the model to fine-tune.
    """
    # Load the dataset
    dataset_loader = FineTuneDataset(settings.dataset_name, "train")
    dataset = dataset_loader.load_dataset()
    dataset = dataset.map(dataset_loader.format_for_fine_tuning, batched=True)

    # Load the model
    model_trainer = ModelTrainer()
    trainer = model_trainer.initialize_trainer(
        loaded_model,
        tokenizer,
        dataset,
    )

    # Train the model
    trainer_stats = trainer.train()
    model = model_trainer.get_model()
    model_path = os.path.join(output_dir, fine_tuned_name)

    # Save the model
    model.save_pretrained_gguf(
        model_path,
        tokenizer,
        quantization_method="f16",
    )  # type: ignore
    print("Trainer stats: ", trainer_stats)

    # Generate the Modelfile based on the template in ./templates/modelfile_template.txt
    with open(
        os.path.join(os.path.dirname(__file__), "./templates/modelfile_template.txt"),
        "r",
    ) as f:
        modelfile_template = Template(f.read())  # type: ignore
    modelfile = modelfile_template.substitute(
        fine_tuned_name=fine_tuned_name,
    )
    with open(os.path.join(model_path, "Modelfile"), "w") as f:
        f.write(modelfile)

    # Push the model to the model hub
    model.push_to_hub_gguf(
        settings.hf_repo_name,
        tokenizer,
        quantization_method="f16",
        token=settings.hf_token.get_secret_value(),
    )  # type: ignore


if __name__ == "__main__":
    main()
