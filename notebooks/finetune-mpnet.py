import ctypes
ctypes.CDLL("libstdc++.so.6", mode=ctypes.RTLD_GLOBAL)

import json
import logging
import os

# Third party imports
import pandas as pd
import wandb

# Local application imports
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, TrainingArguments, Trainer)
from transformers.integrations import WandbCallback
from torch import nn

# Local application imports
import evaluate


def compute_spearman(eval_pred):
    metric = evaluate.load("spearmanr")
    try:
        logits, labels = eval_pred
    except TypeError:
        try:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        except AttributeError:
            logging.error("Error getting logits and labels")
            exit()
    result = metric.compute(predictions=logits, references=labels)
    # Check if result is nan
    if result["spearmanr"] != result["spearmanr"]:
        logging.warn("Spearmanr is nan")
        # Print predictions and labels
        logging.warn("Predictions: {}".format(logits))
        logging.warn("Labels: {}".format(labels))
    return result


def main():
    # Load data
    ROOT = "/mnt/rufus_A/multimemo/data"

    # Specify wandb to log all (including weights and gradients)
    os.environ["WANDB_WATCH"] = "all"

    # data = pd.read_json(os.path.join(ROOT, 'memento_data_recaption.json'),
    # orient='records')
    train = pd.read_json(
        os.path.join(ROOT, "raw/memento_train_data.json"), orient="records"
    )
    val = pd.read_json(os.path.join(ROOT, "raw/memento_val_data.json"),
                       orient="records")

    data = pd.concat([train, val], ignore_index=True)
    print(data.shape)
    data.head()

    RESPONSES_PATH = "/mnt/rufus_A/Memento10k/responses_mem_exp/"
    # Load responses from the path. There is a response per caption,
    # and it is identified with the filename followed by _{i}
    # with i going from 0 to 4 and substituting .mpy with .json

    def load_json(filename):
        # Print the filename if it does not exist
        try:
            json_data = json.load(open(filename, "r"))
        except FileNotFoundError:
            print(filename)
            json_data = {}
        return json_data

    data["responses"] = data["filename"].apply(
        lambda x: [
            load_json(os.path.join(RESPONSES_PATH, x + "_" + str(i) + ".json"))
            for i in range(5)
        ]
    )
    data["responses"][:5]

    # Count number of videos with no responses (list of empty dicts)
    print(data["responses"].apply(
        lambda x: len([y for y in x if y != {}])).value_counts())

    data_exp = data.explode(["responses", "captions"]).reset_index(drop=True)

    data_exp["responses"][0]

    # The following rows have no response because they were content filtered
    data_exp[
        data_exp["responses"].apply(
            lambda x: "content" not in x["choices"][0]["message"].keys()
        )
    ].responses.values

    # Add content to these rows as a "Filtered response" string
    data_exp.loc[
        data_exp["responses"].apply(
            lambda x: "content" not in x["choices"][0]["message"].keys()
        ),
        "responses",
    ] = data_exp.loc[
        data_exp["responses"].apply(
            lambda x: "content" not in x["choices"][0]["message"].keys()
        ),
        "responses",
    ].apply(
        lambda x: {"choices": [{"message": {"content": "Filtered response"}}]}
    )

    # Extract text from responses
    data_exp["mem_exp"] = data_exp["responses"].apply(
        lambda x: x["choices"][0]["message"]["content"]
    )

    # Distribution of number of words in captions and responses
    data_exp["captions_n_words"] = data_exp["captions"].apply(
        lambda x: len(x.split(" ")))
    data_exp["mem_exp_n_words"] = data_exp["mem_exp"].apply(
        lambda x: len(x.split(" ")))

    df = data_exp

    # ## Fine-Tune MPNet with mem_exp

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2")
    df_prepared = df[["mem_exp", "mem_score"]].rename(
        columns={"mem_exp": "text", "mem_score": "label"}
    )
    train_dataset = Dataset.from_pandas(df_prepared[: 7000 * 5]).map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length"),
        batched=True
    )
    eval_dataset = Dataset.from_pandas(df_prepared[7000 * 5:]).map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length"),
        batched=True
    )

    # Reduce dataset size for testing
    # train_dataset = train_dataset.select(range(100))
    # eval_dataset = eval_dataset.select(range(100))

    model = AutoModelForSequenceClassification.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2", num_labels=1
    )
    # Append sigmoid to regressor
    model.classifier = nn.Sequential(model.classifier, nn.Sigmoid())

    BASE_DIR = "/mnt/rufus_A/multimemo"
    new_model_name = "finetuned-mpnet-mem-exp"
    training_args = TrainingArguments(
        new_model_name,
        remove_unused_columns=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-05,
        auto_find_batch_size=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # gradient_accumulation_steps=16,
        # tf32=True,
        # warmup_ratio=0.05,
        # Do not reduce learning rate in scheduler
        lr_scheduler_type="constant",
        logging_steps=10,
        logging_dir=os.path.join(BASE_DIR, "logs", new_model_name),
        load_best_model_at_end=True,
        metric_for_best_model="spearmanr",
        report_to="wandb",
        run_name=new_model_name,
        num_train_epochs=50,
    )

    wandb.login()
    wandb.init(project="multimemo", name=new_model_name, config=training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_spearman,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3,
                                  early_stopping_threshold=0.01)
        ],
    )
    
    trainer.train()
    wandb.summary["train_batch_size"] = trainer._train_batch_size

    wandb.finish()


if __name__ == "__main__":
    main()
