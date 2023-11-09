import ctypes
ctypes.CDLL("libstdc++.so.6", mode=ctypes.RTLD_GLOBAL)

import json
import logging
import os

# Third party imports
from datasets import Dataset
import evaluate
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, TrainingArguments, Trainer)
from transformers.integrations import WandbCallback
from torch import nn
import wandb

# Local application imports
from src.tools.kfolds import compute_test_fold_indices


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
    ROOT = "/home/imartinf/multimemo/data"

    # Specify wandb to log all (including weights and gradients)
    # os.environ["WANDB_WATCH"] = "all"

    # data = pd.read_json(os.path.join(ROOT, 'memento_data_recaption.json'),
    # orient='records')
    # train = pd.read_json(
    #     os.path.join(ROOT, "raw/memento_train_data.json"), orient="records"
    # )
    # val = pd.read_json(os.path.join(ROOT, "raw/memento_val_data.json"),
    #                    orient="records")

    # data = pd.concat([train, val], ignore_index=True)
    data = pd.read_json(os.path.join(ROOT, "processed/memento_blip_captions.json"), orient="records")
    print(data.shape)
    print(data.head())

    compute_test_fold_indices(
        data.rename(columns={"mem_score": "y", "filename": "groups"}), 5, random_state=42
    )

    df = data[["blip_caption", "mem_score"]].rename(
        columns={"blip_caption": "text", "mem_score": "label"}
    )

    # ## Fine-Tune MPNet with mem_exp

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2")
    # df_prepared = df[["mem_exp", "mem_score"]].rename(
    #     columns={"mem_exp": "text", "mem_score": "label"}
    # )
    kf = PredefinedSplit(compute_test_fold_indices(
        data.rename(columns={"mem_score": "y", "filename": "groups"}), 5, random_state=42
    ))
    for i, (train_index, test_index) in enumerate(kf.split(df)):
        print(f"Fold {i}")
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        train_dataset = Dataset.from_pandas(train_df).map(
            lambda x: tokenizer(x["text"], truncation=True, padding="max_length"),
            batched=True
        )
        eval_dataset = Dataset.from_pandas(test_df).map(
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
        new_model_name = "finetuned-mpnet-blip-captions-fold-{}".format(i)
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
    print("Done")


if __name__ == "__main__":
    main()
