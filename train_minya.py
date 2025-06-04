# train_minya.py

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
from tqdm.auto import tqdm

# ====== Параметры обучения ======
NUM_BASE_EPOCHS = 7    # число эпох
BATCH_SIZE = 16
LR = 2e-5
SEED = 42
MAX_LENGTH = 128
TEST_SIZE = 0.2        # 80/20 train/test split
FREEZE_LAYERS = 2      # сколько первых слоёв DistilBERT заморозить
MODEL_ROOT = "models"  # корневая папка для сохранения каждой модели intent

# ====== Сид для воспроизводимости ======
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(path_data: str, path_ant: str):
    """
    path_data: Excel с колонками ['Phrase','Intentionality'] (или ['Example','Intension']).
    path_ant:  Excel с колонками ['Intension','Antonym of Intension'].
    Возвращает df_data и df_ant.
    """
    df = pd.read_excel(path_data, engine="openpyxl")
    # Если колонки названы ['Example','Intension'], переименуем
    if 'Example' in df.columns:
        df.rename(columns={'Example': 'Phrase'}, inplace=True)
    if 'Intension' in df.columns:
        df.rename(columns={'Intension': 'Intentionality'}, inplace=True)
    if 'Phrase' not in df.columns or 'Intentionality' not in df.columns:
        raise KeyError("В файле с данными должны быть колонки 'Phrase' и 'Intentionality' (или Example/Intension).")

    df_ant = pd.read_excel(path_ant, engine="openpyxl")
    if 'Intension' not in df_ant.columns or 'Antonym of Intension' not in df_ant.columns:
        raise KeyError("В файле path_ant должны быть колонки 'Intension' и 'Antonym of Intension'.")

    return df[['Phrase', 'Intentionality']], df_ant[['Intension', 'Antonym of Intension']]


class TorchSet(Dataset):
    """
    Обёртка для токенизированных данных:
    - encodings: {'input_ids': tensor, 'attention_mask': tensor}
    - labels: tensor shape=[N]
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    """
    Для бинарной классификации: accuracy, f1, roc_auc, recall, precision.
    eval_pred: object с logits (shape=[N,2]) и labels (shape=[N]).
    """
    logits, labels = eval_pred
    probs = nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)

    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    roc = evaluate.load("roc_auc")
    rec = evaluate.load("recall")
    prec = evaluate.load("precision")

    return {
        "accuracy": acc.compute(predictions=preds, references=labels)['accuracy'],
        "f1": f1.compute(predictions=preds, references=labels)['f1'],
        "roc_auc": roc.compute(prediction_scores=probs, references=labels)['roc_auc'],
        "recall": rec.compute(predictions=preds, references=labels)['recall'],
        "precision": prec.compute(predictions=preds, references=labels)['precision'],
    }


def train_for_intent(intent: str, anti: str, df_pair: pd.DataFrame, output_dir: str):
    """
    Балансирует df_pair, делает train/test split, запускает Trainer, 
    сохраняет лучшую модель в output_dir/<Intent>/best_model/, и возвращает metrics на test.
    """
    texts = df_pair['Phrase'].tolist()
    labels = df_pair['Label'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=SEED
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(
        X_train, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='pt'
    )
    test_encodings = tokenizer(
        X_test, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='pt'
    )

    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_labels = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TorchSet(train_encodings, train_labels)
    test_dataset = TorchSet(test_encodings, test_labels)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )

    # Заморозим первые FREEZE_LAYERS слоёв DistilBERT
    for i, layer in enumerate(model.distilbert.transformer.layer):
        if i < FREEZE_LAYERS:
            for p in layer.parameters():
                p.requires_grad = False

    intent_folder = os.path.join(output_dir, intent.replace(" ", "_"))
    os.makedirs(intent_folder, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=intent_folder,
        num_train_epochs=NUM_BASE_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs", intent.replace(" ", "_")),
        logging_steps=50,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print(f"\n========== Fine-tuning for intent: '{intent}' vs '{anti}' ==========")
    trainer.train()

    metrics = trainer.evaluate(test_dataset)
    print(f"Metrics on TEST for intent '{intent}': {metrics}\n")

    best_output = os.path.join(intent_folder, "best_model")
    trainer.save_model(best_output)
    tokenizer.save_pretrained(best_output)

    return metrics


def main(args):
    df_data, df_ant = load_data(path_data=args.data, path_ant=args.antonyms)

    pairs = []
    for _, row in df_ant.iterrows():
        intent = row['Intension']
        anti = row['Antonym of Intension']
        subset = df_data[df_data['Intentionality'].isin([intent, anti])].copy()
        if subset.empty:
            continue
        subset['Label'] = (subset['Intentionality'] == intent).astype(int)
        counts = subset['Label'].value_counts()
        if counts.min() == 0:
            continue
        min_cnt = counts.min()
        pos = subset[subset['Label'] == 1].sample(min_cnt, random_state=SEED)
        neg = subset[subset['Label'] == 0].sample(min_cnt, random_state=SEED)
        balanced = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        pairs.append((intent, anti, balanced))

    if not os.path.isdir(MODEL_ROOT):
        os.makedirs(MODEL_ROOT)

    all_results = []
    for intent, anti, balanced_df in pairs:
        balanced_df['Label'] = (balanced_df['Intentionality'] == intent).astype(int)
        metrics = train_for_intent(intent, anti, balanced_df, MODEL_ROOT)
        row = {
            "Intent": intent,
            "Antonym": anti,
            "Test_accuracy": metrics["eval_accuracy"],
            "Test_f1": metrics["eval_f1"],
            "Test_roc_auc": metrics["eval_roc_auc"],
            "Test_recall": metrics["eval_recall"],
            "Test_precision": metrics["eval_precision"]
        }
        all_results.append(row)

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n=== All done. Summary saved to {args.output} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT binary for each intent (Minya’s approach)")
    parser.add_argument(
        "--data", type=str, default="correct_data.xlsx",
        help="Excel с колонками ['Phrase','Intentionality'] (или Example/Intension)."
    )
    parser.add_argument(
        "--antonyms", type=str, default="correct_ant.xlsx",
        help="Excel с колонками ['Intension','Antonym of Intension']."
    )
    parser.add_argument(
        "--output", type=str, default="metrics_summary.csv",
        help="Куда сохранить итоговый CSV с метриками для каждого intent."
    )
    args = parser.parse_args()
    main(args)
