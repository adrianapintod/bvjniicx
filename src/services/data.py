import os
from string import Template

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from pydantic import BaseModel, PrivateAttr

from utils.models import tokenizer


class FineTuneDataset(BaseModel):
    name: str
    split: str
    __loaded_dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = (
        PrivateAttr()
    )

    def __init__(self, name: str, split: str):
        super().__init__(
            name=name,
            split=split,
        )

    def load_dataset(
        self,
    ) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
        self.__loaded_dataset = load_dataset(self.name, split=self.split)
        return self.__loaded_dataset

    @staticmethod
    def format_for_fine_tuning(
        examples: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    ) -> dict[str, list[str]]:
        with open(
            os.path.join(os.path.dirname(__file__), "../templates/alpaca_template.txt"),
            "r",
        ) as f:
            alpaca_template = Template(f.read())
        sql_contexts = examples["sql_context"]  # type: ignore
        sql_prompts = examples["sql_prompt"]  # type: ignore
        sqls = examples["sql"]  # type: ignore
        sql_explanations = examples["sql_explanation"]  # type: ignore
        texts = []
        for sql_context, sql_prompt, sql, sql_explanation in zip(
            sql_contexts, sql_prompts, sqls, sql_explanations
        ):
            text = alpaca_template.substitute(
                sql_context=sql_context,
                sql_prompt=sql_prompt,
                sql=sql,
                sql_explanation=sql_explanation,
            )
            # Add EOS_TOKEN
            text += tokenizer.eos_token
            texts.append(text)
        return {"text": texts}
