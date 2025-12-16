from collections import Counter
from getpass import getpass
from pathlib import Path

import datasets
import evaluate
from datasets import Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    TrainingArguments,
    Trainer
)

# Input
MDA_MANUAL_FOLDER_PATH = "private/mda_manual"
MDA_TONE_DF_PATH = "private/mda_manual_tone_with_fold.csv"
test_fold_index = 0

def gen():
    tone_df = pd.read_csv(MDA_TONE_DF_PATH)
    folder_path = MDA_MANUAL_FOLDER_PATH

    for _, row in tone_df.iterrows():
        ticker = row["Ticker"][:4]
        year = row["Year"]
        tone = row["YoY_Close_Category"]
        ticker_fold = row["Ticker_Fold"]
        filename = f"{ticker}_MDA_{year}.txt"

        with Path(folder_path, filename).open(encoding="utf-8") as fp:
            text = fp.read()

        yield {
            "Ticker": ticker,
            "Ticker_Fold": ticker_fold,
            "Year": year,
            "Text": text,
            "YoY_Close_Category": tone
        }

dataset = Dataset.from_generator(gen)
assert isinstance(dataset, Dataset)

splitted_dataset = DatasetDict({
    "train": dataset.filter(
        lambda example: example["Ticker_Fold"] != test_fold_index),
    "test": dataset.filter(
        lambda example: example["Ticker_Fold"] == test_fold_index),
})

ft_model_folder_path = Path("private/models")
# ft_model_folder_path.mkdir(exist_ok=True, parents=True)

# Cache model belum bisa diimplementasikan. Nantilah.

label2id = {
    "tinggi": 0,
    "rendah": 1,
    "netral": 2
}
id2label = {id:label for label, id in label2id.items()}
best_model = AutoModelForSequenceClassification.from_pretrained(
    Path(ft_model_folder_path, f"finbert_fold_{test_fold_index}"),
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    # Default dropout
)
base_model_path = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# model_folder_path.mkdir(exist_ok=True)




# model_path = 'ProsusAI/finbert'
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# def preprocess_function(example):
#     text = example["Text"]
#     tone = example["YoY_Close_Category"]

#     example = tokenizer(text, truncation=True)
#     example['labels'] = [label2id[tone]]

#     return example

# tokenized_dataset = splitted_dataset.map(preprocess_function)


# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# clf_metrics = evaluate.combine(["accuracy"])
# clf_2_metrics = evaluate.combine(["precision", "recall", "f1"])

# def compute_metrics(eval_pred):
#     raw_predictions, raw_labels = eval_pred
#     tone_predictions = raw_predictions.argmax(axis=1)
#     predictions = tone_predictions

#     labels = raw_labels[:,0]
#     first = clf_metrics.compute(predictions=predictions, references=labels)
#     second = clf_2_metrics.compute(predictions=predictions, references=labels, average="macro")

#     return first | second

# model = AutoModelForSequenceClassification.from_pretrained(
#     model_path,
#     num_labels=len(id2label),
#     id2label=id2label,
#     label2id=label2id,
#     # Default dropout
# )

# # need to login with wandb

# training_args = TrainingArguments(
#     run_name=f"FinBERT Cross Validation Inference - Index {test_fold_index}",

#     learning_rate=3e-6,  # Also learning rate for 1e-2 sucks. 2e-6 lebih buruk dari 3e-6.
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=3,
#     num_train_epochs=50,  # Max, I guess. I hope it's enough.
#     weight_decay=0.01,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     seed=120,

#     push_to_hub=False,
#     report_to="wandb",

#     logging_strategy="epoch",
#     metric_for_best_model="eval_accuracy",
#     greater_is_better=True,
#     save_total_limit=1,  # This is important.
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,

#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["test"],

#     processing_class=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=15, early_stopping_threshold=0.005)],
# )

# trainer.train()
# trainer.evaluate()

# best_model = trainer.model
# assert isinstance(best_model, PreTrainedModel)
# best_model.save_pretrained("/home/my_name/Desktop/t5small")

result_folder_path = Path("private/FinBERT_Result")
result_folder_path.mkdir(exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

df_data = []
sm = nn.Softmax(dim=0)

for example in tqdm(splitted_dataset["test"], total=len(splitted_dataset["test"])):
    if not isinstance(example, dict):
        # This is just for debug
        print(example)
        raise ValueError(f"Unexpected type: {type(example)}")
    
    ticker = example["Ticker"]
    ticker_fold = example["Ticker_Fold"]
    year = example["Year"]
    text = example["Text"]
    tone = example["YoY_Close_Category"]

    inputs = tokenizer(text, truncation=True, return_tensors="pt")  # Intinya pakai ini!
    for k in inputs.keys():
        inputs[k] = inputs[k].to(device)

    result = best_model(**inputs)
    result = result.logits.cpu().detach()
    result = result[0]
    tone_result = sm(result)

    positive_prob = tone_result[0].item()
    negative_prob = tone_result[1].item()
    neutral_prob = tone_result[2].item()

    if tone == "tinggi":
        tone_label_prob = positive_prob
    elif tone == "rendah":
        tone_label_prob = negative_prob
    elif tone == "netral":
        tone_label_prob = neutral_prob
    else:
        raise ValueError(f"Unknown tone: {tone}")

    df_data.append({
        "Ticker": ticker,
        "Ticker_Fold": ticker_fold,
        "Year": year,
        "YoY_Close_Category": tone,
        "Tone_Label_Prob": tone_label_prob,
        "Positive_Prob": positive_prob,
        "Negative_Prob": negative_prob,
        "Neutral_Prob": neutral_prob
    })

df = pd.DataFrame(df_data)
df = df.sort_values(["Ticker", "Year"])

df.to_csv(Path(
    result_folder_path,
    f"Fold_{test_fold_index}_Result.csv"
), index=False)
