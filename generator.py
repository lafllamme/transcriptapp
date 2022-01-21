import torch
import pandas as pd
import nlpaug.augmenter.word as nlpaw

from augment import *

pd.options.mode.chained_assignment = None
# default='warn'


# reads the the labels files


labels_df = pd.read_csv("labels.csv", header=None, index_col=0)
labels_df.index.name = "id"
labels_df.columns = ["name"]


def create_embedding(tokens, model):
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    model.eval()
    with torch.no_grad():
        embedding = model(input_ids, attention_mask)[0][:, 0, :]

    return embedding


def tokenize_text(text: str, tokenizer):
    print(text)
    tokens = tokenizer(
        text,
        add_special_tokens=True,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",

    )
    print(tokens)
    return tokens


def get_k_most_similar_keywords(text, label_embeddings, model, tokenizer, k=10):
    tokens = tokenize_text(text, tokenizer)
    embedding = create_embedding(tokens, model)
    print('TOKENS: ', tokens, 'EMBDEDDINGS: ', embedding)

    # calculate similarities
    similarities = torch.cdist(torch.tensor(embedding).float(
    ), torch.tensor(label_embeddings).float()) * (-1)
    top_k_values, top_k_indices = similarities.topk(k)
    top_k_keywords = labels_df.iloc[top_k_indices.squeeze()]
    top_k_keywords["Ähnlichkeit"] = top_k_values.squeeze() * -1

    print('SIMILARITIES: ', similarities, 'TOP K :', top_k_values,
          'TOP K IND: ', top_k_indices, 'KEYW : ', top_k_keywords)

    return top_k_keywords


def retrainModel(input_df, model_path, num_threads, num_times):
    aug10p = nlpaw.ContextualWordEmbsAug(
        model_path=model_path, aug_min=1, aug_p=0.1, action="substitute")
    balanced_df = augment_text(
        input_df, aug10p, num_threads=num_threads, num_times=num_times)

    return balanced_df
