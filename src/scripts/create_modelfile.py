#!/usr/bin/env python3

import os
from string import Template

import click


@click.command()
@click.option(
    "--fine-tuned-name",
    help="Name of the fine-tuned model.",
    required=True,
    type=str,
)
def create_modelfile(fine_tuned_name):
    """
    Creates a modelfile for the specified model name using the template.
    The modelfile will be stored in ./data/models/output/{model_name}/Modelfile
    """
    # Get the template file path
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "templates",
        "modelfile_template.txt",
    )

    # Output modelfile path
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "models",
        "output",
        fine_tuned_name,
        "Modelfile",
    )

    # Read the template
    with open(template_path, "r") as f:
        template_content = Template(f.read())

    # Replace placeholder with model name
    content = template_content.substitute(
        fine_tuned_name=fine_tuned_name,
    )

    # Write the modified content to the output file
    with open(output_path, "w") as f:
        f.write(content)

    click.echo(f"Modelfile created at: {output_path}")


if __name__ == "__main__":
    create_modelfile()
