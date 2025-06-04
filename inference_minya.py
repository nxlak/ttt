# inference_minya.py

import os
import argparse
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ====== Параметры ======
MODEL_ROOT = "models"           # папка, где лежат подпапки <Intent>/best_model
MAX_LENGTH = 128


def load_all_models(model_root: str):
    """
    Проходит по model_root, находит подпапки вида <Intent>/best_model/,
    и загружает DistilBertForSequenceClassification.from_pretrained(...) для каждой.
    Возвращает dict: intent_name -> модель (в eval() на GPU/CPU).
    """
    intent_to_model = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for intent_dir in os.listdir(model_root):
        full_dir = os.path.join(model_root, intent_dir, "best_model")
        if not os.path.isdir(full_dir):
            continue
        try:
            model = DistilBertForSequenceClassification.from_pretrained(full_dir, num_labels=2)
            model.to(device)
            model.eval()
            intent_name = intent_dir.replace("_", " ")
            intent_to_model[intent_name] = model
        except Exception as e:
            print(f"Warning: не удалось загрузить модель для '{intent_dir}': {e}")
    if not intent_to_model:
        raise FileNotFoundError("В папке models/ не найдено ни одной подпапки '<Intent>/best_model'.")
    print(f"Loaded {len(intent_to_model)} intent-models into memory ({'GPU' if torch.cuda.is_available() else 'CPU'}).")
    return intent_to_model


def inference(texts: list[str], intent_models: dict[str, torch.nn.Module], tokenizer: DistilBertTokenizerFast):
    """
    texts: список строк (1 или несколько). 
    intent_models: dict intent_name -> модель DistilBertForSequenceClassification.
    tokenizer: DistilBertTokenizerFast.

    Возвращает List[List[(intent_name, probability)]]: 
      для каждого текста по отсортированному списку из 125 пар (intent, prob).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Токенизируем batch из всех texts
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    num_texts = len(texts)
    intent_names = list(intent_models.keys())
    total_intents = len(intent_names)

    # 2) Буфер для вероятностей [total_intents × num_texts]
    all_probs = [[0.0] * num_texts for _ in range(total_intents)]

    with torch.no_grad():
        for i, intent in enumerate(intent_names):
            model = intent_models[intent]
            logits = model(ids, mask).logits  # [num_texts, 2]
            # probability of class “1”:
            probs = nn.functional.softmax(logits, dim=1)[:, 1].cpu().tolist()
            all_probs[i] = probs

    # 3) Собираем для каждого текста словарь intent->prob
    per_text_dicts = [dict() for _ in range(num_texts)]
    for i, intent in enumerate(intent_names):
        for j in range(num_texts):
            per_text_dicts[j][intent] = all_probs[i][j]

    # 4) Сортируем для каждого текста список (intent, prob) по убыванию prob
    sorted_results = []
    for j in range(num_texts):
        items = list(per_text_dicts[j].items())
        items.sort(key=lambda x: x[1], reverse=True)
        sorted_results.append(items)

    return sorted_results


def main(args):
    print("Loading all intent-models…")
    intent_models = load_all_models(MODEL_ROOT)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    texts = args.texts
    if not texts:
        print("Укажите хотя бы одну фразу через --texts.")
        return

    print(f"Performing inference for {len(texts)} sentence(s)…")
    results = inference(texts, intent_models, tokenizer)

    for idx, text in enumerate(texts):
        print(f"\n=== Text #{idx+1}: «{text}» ===")
        print(f"{'Intent':<50} {'Probability':<10}")
        print("-" * 62)
        for intent_name, prob in results[idx]:
            print(f"{intent_name:<50} {prob:.4f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for all 125 intents (Minya’s style) on GPU.")
    parser.add_argument(
        "--texts", nargs="+", default=[],
        help="One or more sentences, e.g.: --texts \"Sentence one.\" \"Sentence two.\""
    )
    args = parser.parse_args()
    main(args)
