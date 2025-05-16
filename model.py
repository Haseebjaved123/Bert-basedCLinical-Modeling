import torch
import torch.nn as nn
from transformers import BertModel

class HybridPooling(nn.Module):
    def __init__(self):
        super(HybridPooling, self).__init__()

    def forward(self, hidden_states):
        # [CLS] token
        cls_token = hidden_states[:, 0]  # shape: [batch, hidden]
        # Mean Pooling
        avg_pool = torch.mean(hidden_states, dim=1)
        # Max Pooling
        max_pool, _ = torch.max(hidden_states, dim=1)
        # Concatenate
        return torch.cat((cls_token, avg_pool, max_pool), dim=1)  # shape: [batch, 3*hidden]

class ModifiedTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ModifiedTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Self-attention + residual + norm
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        # Feed-forward + residual + norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class FineTunedBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super(FineTunedBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        self.hybrid_pool1 = HybridPooling()
        self.hybrid_pool2 = HybridPooling()

        # Project hybrid-pooled features back to hidden size
        self.encoder1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.encoder2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder3 = nn.Linear(self.hidden_size, self.hidden_size)

        self.transformer_block = ModifiedTransformerBlock(self.hidden_size)

        # Final classifier
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state  # shape: [batch, seq_len, hidden]

        # First hybrid pooling
        pooled1 = self.hybrid_pool1(hidden_states)  # shape: [batch, 3*hidden]
        encoded1 = self.encoder1(pooled1)           # shape: [batch, hidden]

        # Second hybrid pooling
        pooled2 = self.hybrid_pool2(hidden_states)  # shape: [batch, 3*hidden]
        encoded2 = self.encoder1(pooled2)           # reuse encoder1 for simplicity

        # Reshape and stack (simulate additional encoder layers)
        x = self.encoder2(encoded2)
        x = self.encoder3(x)
        x = x.unsqueeze(1)  # shape: [batch, 1, hidden]

        # Apply transformer block
        x = self.transformer_block(x)

        # Final prediction (use first token output)
        logits = self.classifier(x[:, 0])
        return logits
