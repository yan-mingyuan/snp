import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.utils import softmax
from torch_geometric.nn import HGTConv
from torch_geometric.nn.dense import HeteroDictLinear


class DiseaseEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pooling_type, dropout_prob=0.3):
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


class VariantEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_prob=0.3):
        super(VariantEncoder, self).__init__()
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
        )

        # Residual projection (in case input_dim != output_dim)
        self.projection = nn.Linear(
            in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, variant_x, gene_x):
        x = torch.cat((variant_x, gene_x), 0)

        # Apply the shared MLP transformation with residual connections
        out = self.shared_mlp(x)
        out = out + self.projection(x)

        variant_out, gene_out = torch.chunk(out, chunks=2, dim=0)
        return variant_out - gene_out


class HeteroDictBatchNorm(nn.Module):
    def __init__(self, num_features, node_types, **kwargs):
        super(HeteroDictBatchNorm, self).__init__()

        # Create a BatchNorm1d layer for each node type
        self.batch_norms = nn.ModuleDict({
            node_type: nn.BatchNorm1d(num_features, **kwargs) for node_type in node_types
        })

    def forward(self, x_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.batch_norms[node_type](x)
        return x_dict


class HeteroDictLayerNorm(nn.Module):
    def __init__(self, num_features, node_types, **kwargs):
        super(HeteroDictLayerNorm, self).__init__()

        # Create a BatchNorm1d layer for each node type
        self.batch_norms = nn.ModuleDict({
            node_type: nn.LayerNorm(num_features, **kwargs) for node_type in node_types
        })

    def forward(self, x_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.batch_norms[node_type](x)
        return x_dict


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        # self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels,
                           metadata, num_heads)
            self.convs.append(conv)
            # norm = HeteroDictBatchNorm(hidden_channels, metadata[0])
            # self.norms.append(norm)

        self.act = nn.GELU()
        self.proj = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            # nn.BatchNorm1d(out_channels)
        )

    def forward(self, x_dict, edge_index_dict):
        # for num, conv in enumerate(self.convs):
        for num in range(self.num_layers):
            # HGTConv layer with residual connection
            # new_x_dict = self.convs[num](x_dict, edge_index_dict)
            # x_dict = {key: new_x_dict[key] + x_dict[key] for key in x_dict}

            x_dict = self.convs[num](x_dict, edge_index_dict)

            # x_dict = self.norms[num](x_dict)
            if num != self.num_layers:
                x_dict = {key: self.act(x) for key, x in x_dict.items()}
        x_dict['gene'] = self.proj(x_dict['gene'])
        # x_dict['trait'] = self.proj(x_dict['trait'])
        return x_dict['gene'], x_dict['trait']


class OurModel(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_graph_layers, data, variant_x, gene_local_x, pooling_type, dropout_prob=0.3):
        super(OurModel, self).__init__()

        disease_channels = data['trait'].num_node_features
        metadata = data.metadata()
        variant_input_dim = variant_x.size(1)

        self.data = data
        self.register_buffer('variant_x', variant_x)
        self.register_buffer('gene_local_x', gene_local_x)

        # Disease Encoder
        self.disease_encoder = DiseaseEncoder(
            in_channels=disease_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            pooling_type=pooling_type,
            dropout_prob=dropout_prob
        )

        # Variant Encoder
        self.variant_encoder = VariantEncoder(
            in_channels=variant_input_dim,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout_prob=dropout_prob
        )

        # Heterogeneous Graph Transformer
        self.hetero_transform = HeteroDictLinear(
            in_channels={
                node_type: data[node_type].num_node_features for node_type in data.node_types
            },
            out_channels=hidden_channels
        )
        self.hetero_norms = HeteroDictBatchNorm(
            hidden_channels, data.node_types)
        self.hgt = HGT(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            num_layers=num_graph_layers,
            metadata=metadata
        )

    def forward(self, variant_id, gene_id, trait_ids, batch_ids):
        # Diseases Encoding
        disease_embedding = self.disease_encoder(
            x=self.data['trait'].x[trait_ids],
            batch_ids=batch_ids
        )

        # Variant Encoding
        transformed_variant_x = self.variant_encoder(
            variant_x=self.variant_x[variant_id].unsqueeze(0),
            gene_x=self.gene_local_x[gene_id].unsqueeze(0),
        )

        # Graph Projection
        transformed_x_dict = self.hetero_transform(self.data.x_dict)
        transformed_x_dict = self.hetero_norms(transformed_x_dict)

        # Graph Information Mixing
        transformed_x_dict['gene'][[gene_id]] += transformed_variant_x

        # Graph Message Passing
        gene_embeddings, trait_embeddings = self.hgt(
            x_dict=transformed_x_dict,
            edge_index_dict=self.data.edge_index_dict
        )
        gene_embedding = gene_embeddings[gene_id]

        return gene_embedding, disease_embedding


def multi_positive_info_nce_loss(gene_embedding, disease_embedding, num_positives, temperature=0.7, norm=False):
    if norm:
        gene_embedding = F.normalize(gene_embedding, dim=-1)
        disease_embedding = F.normalize(disease_embedding, dim=-1)
    
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
