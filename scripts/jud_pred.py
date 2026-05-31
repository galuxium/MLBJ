import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer

CHUNK_SIZE = 510    
MAX_CHUNKS = 8      
IS_ROBERTA_STYLE = True

MODEL_NAME='law-ai/InCaseLawBERT'
TOKENIZER=AutoTokenizer.from_pretrained(MODEL_NAME)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class HierarchicalClassifier(nn.Module):
    def __init__(self, base_model_name, num_labels=2, dropout_prob=0.1):
        super().__init__()
        # AutoModel returns just the encoder (no classification head)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for BERT-base
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.base_model_name = base_model_name

        # Attention pooling: learn a scalar attention score per chunk
        self.chunk_attention = nn.Linear(hidden_size, 1)

        # Classification head (same shape as BertForSequenceClassification's)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, chunk_mask):
        """
        input_ids:      (B, K, L)  — batched chunked token IDs
        attention_mask: (B, K, L)  — per-chunk token attention (1=real, 0=pad)
        chunk_mask:     (B, K)     — per-doc chunk validity (1=real chunk, 0=pad chunk)
        Returns: logits (B, num_labels)
        """
        B, K, L = input_ids.shape

        # Flatten chunk dim into batch dim so encoder processes B*K sequences
        flat_input_ids = input_ids.view(B * K, L)
        flat_attention_mask = attention_mask.view(B * K, L)

        # Encode all chunks. AutoModel returns BaseModelOutput with last_hidden_state.
        outputs = self.encoder(flat_input_ids, attention_mask=flat_attention_mask)

        # Take CLS embedding from each chunk: (B*K, H)
        chunk_cls = outputs.last_hidden_state[:, 0, :]

        # Reshape back to (B, K, H)
        chunk_embeds = chunk_cls.view(B, K, self.hidden_size)

        # Compute attention scores per chunk: (B, K)
        attn_scores = self.chunk_attention(chunk_embeds).squeeze(-1)

        # Mask out padding chunks before softmax — set their scores to -inf
        # so they get zero weight in the softmax.
        attn_scores = attn_scores.masked_fill(chunk_mask == 0, float('-inf'))

        # Softmax over chunks → attention weights (B, K)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Weighted sum of chunk embeddings → document embedding (B, H)
        doc_embed = (chunk_embeds * attn_weights.unsqueeze(-1)).sum(dim=1)

        # Classification head
        doc_embed = self.dropout(doc_embed)
        logits = self.classifier(doc_embed)
        return logits

model = HierarchicalClassifier(base_model_name=MODEL_NAME, num_labels=2)
model.to(DEVICE)


def make_chunks(text, tokenizer):
    """
    Tokenize a document and split into chunks. Returns:
        input_ids: list of [chunk_size+2] int lists (padded to MAX_CHUNKS slots)
        attention_mask: list of [chunk_size+2] int lists (1 for real, 0 for pad)
        n_real_chunks: int — actual chunk count (rest are padding chunks)
    """
    # Tokenize the full text
    if IS_ROBERTA_STYLE:
        tokens = tokenizer.tokenize(text, add_prefix_space=True)
    else:
        tokens = tokenizer.tokenize(text)

    pad_id = tokenizer.pad_token_id
    cls = tokenizer.cls_token
    sep = tokenizer.sep_token
    target_len = CHUNK_SIZE + 2  # CLS + content + SEP

    chunks_ids = []
    chunks_mask = []

    # Walk through tokens in stride of CHUNK_SIZE (no overlap)
    # Cap at MAX_CHUNKS * CHUNK_SIZE tokens (rest of doc is discarded)
    capped_tokens = tokens[:MAX_CHUNKS * CHUNK_SIZE]
    n_real_chunks = 0

    for start in range(0, len(capped_tokens), CHUNK_SIZE):
        chunk = capped_tokens[start:start + CHUNK_SIZE]
        # Add special tokens and convert to IDs
        chunk_with_special = [cls] + chunk + [sep]
        ids = tokenizer.convert_tokens_to_ids(chunk_with_special)
        mask = [1] * len(ids)
        # Pad each chunk to target_len
        pad_amount = target_len - len(ids)
        ids = ids + [pad_id] * pad_amount
        mask = mask + [0] * pad_amount
        chunks_ids.append(ids)
        chunks_mask.append(mask)
        n_real_chunks += 1

    # If document was empty or too short, ensure at least 1 chunk
    if n_real_chunks == 0:
        chunks_ids.append([tokenizer.convert_tokens_to_ids(cls), tokenizer.convert_tokens_to_ids(sep)] + [pad_id] * (target_len - 2))
        chunks_mask.append([1, 1] + [0] * (target_len - 2))
        n_real_chunks = 1

    # Pad chunk LIST to MAX_CHUNKS slots (with all-padding chunks)
    while len(chunks_ids) < MAX_CHUNKS:
        chunks_ids.append([pad_id] * target_len)
        chunks_mask.append([0] * target_len)

    return chunks_ids, chunks_mask, n_real_chunks

def predict_judgment(text, model=model, tokenizer=TOKENIZER, device=DEVICE, use_amp=True):
    """
    Predict accepted/rejected for a single legal document using the
    hierarchical model. Returns dict with prediction, confidence, attention
    weights showing which chunks the model focused on.
    """
    model.eval()

    # Chunk the input text using the same logic as training
    chunks_ids, chunks_mask, n_real = make_chunks(text, tokenizer)

    # Build chunk-validity mask (1 for real chunks, 0 for padding chunks)
    chunk_mask = [1] * n_real + [0] * (MAX_CHUNKS - n_real)

    # Convert to tensors — batch dim = 1
    input_ids_t = torch.tensor([chunks_ids], dtype=torch.long).to(device)
    attention_mask_t = torch.tensor([chunks_mask], dtype=torch.long).to(device)
    chunk_mask_t = torch.tensor([chunk_mask], dtype=torch.long).to(device)

    with torch.no_grad(), torch.amp.autocast(enabled=use_amp and torch.cuda.is_available(), dtype=torch.float16, device_type=device.type, cache_enabled=True):
        logits = model(input_ids_t, attention_mask=attention_mask_t, chunk_mask=chunk_mask_t)
        probabilities = torch.softmax(logits, dim=-1)

    logits_list = logits[0].float().cpu().tolist()
    probs_list = probabilities[0].float().cpu().tolist()
    pred_idx = int(np.argmax(logits_list))

    return {
        'prediction': 'accepted' if pred_idx == 1 else 'rejected',
        'confidence': round(probs_list[pred_idx], 4),
        'logits': [round(x, 4) for x in logits_list],
        'probabilities': [round(x, 4) for x in probs_list],
        'n_chunks_used': n_real,
    }


if __name__ == "__main__":
    with open("samples/sample_judgment.txt", "r") as f:
        text = f.read()
    
    result = predict_judgment(text)
    print(f"Prediction: \n{result}")