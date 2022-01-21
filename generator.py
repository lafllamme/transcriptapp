import torch
import pandas as pd
#reads the the labels files


labels_df = pd.read_csv("labels.csv", header=None, index_col=0)
labels_df.index.name = "id"
labels_df.columns = ["name"]

def create_embedding(tokens, model):
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    model.eval()
    with torch.no_grad():
        embedding = model(input_ids, attention_mask)[0][:,0,:]
    
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
    print(embedding)

    # calculate similarities
    similarities = torch.cdist(torch.tensor(embedding).float(), torch.tensor(label_embeddings).float()) * (-1)
    # similarities = torch.cdist(torch.tensor(embedding).clone().detach(), torch.tensor(embedding).clone().detach())
    top_k_values, top_k_indices = similarities.topk(k)
    top_k_keywords = labels_df.iloc[top_k_indices.squeeze()]
    top_k_keywords["Ähnlichkeit"] = top_k_values.squeeze() * -1
 
    return top_k_keywords


