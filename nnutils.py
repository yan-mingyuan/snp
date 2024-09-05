import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.utils import softmax
from torch_geometric.nn import Linear, HGTConv
from torch_geometric.nn.dense import HeteroDictLinear


class DiseaseEncoder(nn.Module):
    def __init__(self, in_channels=768, hidden_channels=256, out_channels=256, pooling_type='mean', dropout_prob=0.3):
        super(DiseaseEncoder, self).__init__()

        # Shared MLP
        self.shared_mlp = nn.Sequential(
            # First Linear
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            # Second Linear
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

        # Residual projection (in case input_dim != output_dim)
        self.projection = nn.Linear(
            in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        # Set pooling type
        if pooling_type == "attention":
            self.attention_layer = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, 1)
            )
        self.pooling_type = pooling_type

    def forward(self, x, batch_ids):
        # Apply the shared MLP transformation
        out = self.shared_mlp(x)

        # Residual connection
        out = out + self.projection(x)

        # Apply the selected pooling mechanism
        if self.pooling_type == 'mean':
            return torch_scatter.scatter_mean(out, batch_ids, dim=0)
        elif self.pooling_type == 'max':
            # [0] because scatter_max returns both values and indices
            return torch_scatter.scatter_max(out, batch_ids, dim=0)[0]
        elif self.pooling_type == 'sum':
            return torch_scatter.scatter_sum(out, batch_ids, dim=0)
        elif self.pooling_type == 'attention':
            # Compute attention scores for each node
            attention_scores = self.attention_layer(
                out).squeeze(-1)  # Shape: [num_nodes]
            # Apply softmax within each batch
            attention_weights = softmax(attention_scores, batch_ids)
            # Compute the weighted sum of features using scatter_add
            # Shape: [num_batches, output_dim]
            return torch_scatter.scatter_sum(out * attention_weights.unsqueeze(-1), batch_ids, dim=0)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels,
                           metadata, num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        x_dict['gene'] = self.lin(x_dict['gene'])
        # x_dict['trait'] = self.lin(x_dict['trait'])
        return x_dict['gene'], x_dict['trait']


class OurModel(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_graph_layers, data, variant_input_dim, pooling_type='attention', dropout_prob=0.3):
        super(OurModel, self).__init__()

        disease_channels = data['trait'].num_node_features
        metadata = data.metadata()

        # Disease Encoder
        self.disease_encoder = DiseaseEncoder(
            in_channels=disease_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            pooling_type=pooling_type,
            dropout_prob=dropout_prob
        )

        # Heterogeneous Graph Transformer
        self.hgt = HGT(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            num_layers=num_graph_layers,
            metadata=metadata
        )

        # Projection for variant embeddings
        self.variant_transform = Linear(
            in_channels=variant_input_dim, out_channels=hidden_channels)
        self.hetero_transform = HeteroDictLinear(
            in_channels={
                node_type: data[node_type].num_node_features for node_type in data.node_types
            },
            out_channels=hidden_channels
        )

    def forward(self, data, variant_x, variant_id, gene_id, trait_ids, batch_ids):
        # Diseases Encoding
        disease_embedding = self.disease_encoder(
            x=data['trait'].x[trait_ids],
            batch_ids=batch_ids
        )

        # Variant Encoding
        transformed_variant_x = self.variant_transform(
            variant_x[variant_id].to(data['trait'].x.device))
        transformed_x_dict = self.hetero_transform(data.x_dict)
        transformed_x_dict['gene'][gene_id] += transformed_variant_x

        # Graph Message Passing
        gene_embeddings, trait_embeddings = self.hgt(
            x_dict=transformed_x_dict,
            edge_index_dict=data.edge_index_dict
        )
        gene_embedding = gene_embeddings[gene_id]

        return gene_embedding, disease_embedding


def multi_positive_info_nce_loss(gene_embedding, disease_embedding, num_positives, temperature=0.7):
    # Compute similarity (dot product) between gene_embedding and all disease embeddings
    # Optionally, you can use cosine similarity. Here we use simple dot product as logits.
    # Shape: [num_positives + num_negatives]
    similarity = torch.matmul(disease_embedding, gene_embedding) / temperature

    # Create labels: the first `num_positives` positions correspond to positive diseases
    labels = torch.zeros(similarity.size(
        0), dtype=torch.float).to(gene_embedding.device)
    labels[:num_positives] = 1.0 / num_positives  # Mark the positive examples

    # Apply cross-entropy loss (negative log-likelihood over logits)
    loss = F.cross_entropy(similarity.unsqueeze(0), labels.unsqueeze(0))

    return loss


def cosine_margin_loss(gene_embedding, disease_embedding, num_positives, margin=0.5):
    # Normalize embeddings to use cosine similarity
    gene_embedding = F.normalize(gene_embedding, dim=-1)
    disease_embedding = F.normalize(disease_embedding, dim=-1)

    # Cosine similarity between gene and all disease embeddings
    # Shape: [num_positives + num_negatives]
    similarity = torch.matmul(disease_embedding, gene_embedding)

    # Extract the positive sample similarity
    positive_similarity = similarity[:num_positives].mean()
    positive_loss = 1.0 - positive_similarity  # We want this to be close to 0

    # For negative samples: similarity[num_positives:] gives the negatives
    negative_similarity = similarity[num_positives:]
    # Compute the margin loss for negatives (if similarity > margin, penalize)
    # Encourage negative similarity < margin
    negative_loss = F.relu(negative_similarity - margin).mean()

    # Total loss is the combination of the positive and negative losses
    loss = positive_loss + negative_loss

    return loss
