import torch
import pandas as pd
import streamlit as st

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
    top_k_keywords.sort_values(by='Ähnlichkeit', ascending=False)

    print('SIMILARITIES: ', similarities, 'TOP K :', top_k_values,
          'TOP K IND: ', top_k_indices, 'KEYW : ', top_k_keywords)
    print('last', labels_df.iloc[top_k_indices.squeeze()])
    print("LABEL EMBEDDINGS: ", label_embeddings)
    return top_k_keywords


def retrainModel(wordList, text, label_embeddings, model, tokenizer, k=10):
    df = pd.read_csv("labels.csv", header=None, index_col=0)
    df.index.name = "id"
    df.columns = ["name"]

   # frame where wordList ∉ labels
    frame = df[~df.name.isin(wordList)]
    pd_frame = pd.DataFrame(frame)
    cleaned_pd_frame = pd_frame.query('name != {}'.format(wordList))
    df.head()
    tokens = tokenize_text(text, tokenizer)
    embedding = create_embedding(tokens, model)
    print('TOKENS: ', tokens, "\n", 'EMBDEDDINGS: ', embedding, "\n")
    print('CLEANED: ', cleaned_pd_frame, "\n")

    # calculate new similarities
    similarities = torch.cdist(torch.tensor(embedding).float(
    ), torch.tensor(label_embeddings).float()) * (-1)
    top_k_values, top_k_indices = similarities.topk(k)
    top_k_keywords = cleaned_pd_frame.iloc[top_k_indices.squeeze()]
    top_k_keywords["Ähnlichkeit"] = top_k_values.squeeze(
    ) * (-1 - ((float(len(wordList) * 0.5)) / 25))
    top_k_keywords.sort_values(by='Ähnlichkeit', ascending=False)

    
    print('Changed by', (-1 - ((float(len(wordList) * 0.5)) / 25)))
    print('SIMILARITIES: ', similarities, "\n", 'TOP K :', top_k_values, "\n",
          'TOP K IND: ', top_k_indices, "\n", 'KEYW : ', top_k_keywords, "\n")
    print('last', cleaned_pd_frame.iloc[top_k_indices.squeeze()])
    print("LABEL EMBEDDINGS: ", label_embeddings, "\n")
    return top_k_keywords
