import ast
import os.path as osp
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb

from config import PATH_PROCESSED, DEVICE, SEED, set_seed_all
from datautils import create_graph, load_node_csv, split_dataset, LabelGenerator, sample_negative_trait_groups
from nnutils import *

set_seed_all(SEED)

# Step 1: Load graph data
# Creates the graph structure and loads node mappings
data, gene_mapping, trait_mapping = create_graph()
data = data.to(DEVICE)

# Step 2: Load variant features
# Reads variant data and feature matrix
variant_mapping, variant_x = load_node_csv(
    osp.join(PATH_PROCESSED, 'variant_x.csv'), index_col='SNPs'
)
_, gene_local_x = load_node_csv(
    osp.join(PATH_PROCESSED, 'gene_local_x.csv'), index_col=0
)
print(f"Variants features:   ({variant_x.size(0)}, {variant_x.size(1)})")
print(f"Gene local features: ({variant_x.size(0)}, {variant_x.size(1)})")

# Step 3: Load labels and split datasets
labels = pd.read_csv(
    osp.join(PATH_PROCESSED, 'labels.csv'),
    index_col='snps',
    converters={'hpo_id': ast.literal_eval}
)
train_labels, val_labels, test_labels = split_dataset(
    labels, random_state=SEED)
print(f"Variants associated with diseases and trait groups: "
      f"Total = {labels.shape[0]} (Train: {train_labels.shape[0]}, "
      f"Validation: {val_labels.shape[0]}, Test: {test_labels.shape[0]})")

# Step 4: Load disease-to-trait mapping for negative sampling
# Read disease-to-trait mappings, converting 'hpo_id' into list format
disease_to_traits = pd.read_csv(
    osp.join(PATH_PROCESSED, 'disease_to_traits.csv'),
    index_col='disease_index',
    converters={'hpo_id': ast.literal_eval}
)
print(f"Number of diseases: {disease_to_traits.shape[0]}")

hidden_channels = 200
out_channels = 128
num_heads = 2
num_graph_layers = 2
pooling_type = 'attention'
dropout_prob = 0.3
temperature = 0.7
norm = True

learning_rate = 1e-3
weight_decay = 1e-2
momentum_gradient = 0.9
momentum_square = 0.95

max_epochs = 100

# Metrics vary in different settings
batch_size = 2
num_train_samples = 1000
num_valid_samples = 1000
max_positives = None
max_num_negatives = None

# Initialize wandb
wandb.init(
    project="snp", 
    config={
        "hidden_channels": hidden_channels,
        "out_channels": out_channels,
        "num_heads": num_heads,
        "num_graph_layers": num_graph_layers,
        "pooling_type": pooling_type,
        "dropout_prob": dropout_prob,
        "temperature": temperature,
        "norm": norm,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum_gradient": momentum_gradient,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
})

def train(model, optimizer, label_loader, num_negatives, max_positives=None, temperature=0.7, norm=False):
    """Train the model with a given optimizer and label loader."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        label_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False, ncols=100)
    # progress_bar = tqdm(
        # label_loader, desc=f"Epoch [{epoch+1}/{max_epochs}] - Training", leave=False, ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for label_batch in progress_bar:
        optimizer.zero_grad()  # Reset the gradients before each batch
        batch_loss = 0.0

        # Process each label batch
        for variant, (gene, disease, traits) in label_batch.iterrows():
            variant_id = variant_mapping[variant]
            gene_id = gene_mapping[gene]

            # Get positive and negative samples
            positive_relations = labels.loc[labels.index == variant, [
                'disease_index', 'hpo_id']]
            if max_positives is not None:
                positive_relations = positive_relations.sample(
                    n=min(max_positives, len(positive_relations)))
            positive_diseases = positive_relations['disease_index'].to_list()
            positive_trait_groups = positive_relations['hpo_id'].to_list()

            # Sample negative trait groups
            negative_trait_groups = sample_negative_trait_groups(
                disease_to_traits, positive_diseases, num_negatives)

            # Combine positive and negative trait groups
            num_positives = len(positive_relations)
            trait_groups = positive_trait_groups + negative_trait_groups
            batch_ids = torch.cat([torch.full((len(group),), i)
                                   for i, group in enumerate(trait_groups)])
            trait_ids = torch.tensor(
                [trait_mapping[trait] for trait_group in trait_groups for trait in trait_group])

            # Move to device (GPU)
            batch_ids = batch_ids.to(DEVICE)
            trait_ids = trait_ids.to(DEVICE)

            # Forward pass
            gene_embedding, disease_embedding = model(
                variant_id=variant_id,
                gene_id=gene_id,
                trait_ids=trait_ids,
                batch_ids=batch_ids
            )

            # Compute loss (use InfoNCE loss)
            loss = multi_positive_info_nce_loss(
                gene_embedding, disease_embedding, num_positives, temperature, norm=True)

            # Backward pass
            loss.backward()
            batch_loss += loss.item()

        optimizer.step()
        # Accumulate loss and update progress bar
        running_loss += batch_loss / len(label_batch)
        num_batches += 1

        # Log and print batch loss
        wandb.log({"train_batch_loss": batch_loss / len(label_batch)})
        progress_bar.set_postfix({'loss': running_loss / num_batches})

    # wandb.log({"train_loss": running_loss / num_batches})
    # print(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {running_loss / num_batches:.4f}")


def test(model, label_loader, max_num_negatives, temperature, norm):
    """Test the model and compute AUC, AUPRC, and MRR."""
    model.eval()
    all_trues = []
    all_predictions = []
    all_ranks = []

    with torch.no_grad():
        for label_batch in label_loader:
            batch_loss = 0.0
            for variant, (gene, disease, traits) in label_batch.iterrows():
                variant_id = variant_mapping[variant]
                gene_id = gene_mapping[gene]

                positive_relations = labels.loc[labels.index == variant, [
                    'disease_index', 'hpo_id']]
                positive_diseases = positive_relations['disease_index'].to_list(
                )
                positive_trait_groups = positive_relations['hpo_id'].to_list()

                # Sample negative trait groups
                negative_trait_groups = sample_negative_trait_groups(
                    disease_to_traits, positive_diseases, max_num_negatives=max_num_negatives)
                num_negatives = len(negative_trait_groups)

                # Combine positive and negative trait groups
                num_positives = len(positive_relations)
                trait_groups = positive_trait_groups + negative_trait_groups
                batch_ids = torch.cat([torch.full((len(group),), i)
                                       for i, group in enumerate(trait_groups)])
                trait_ids = torch.tensor(
                    [trait_mapping[trait] for trait_group in trait_groups for trait in trait_group])

                batch_ids = batch_ids.to(DEVICE)
                trait_ids = trait_ids.to(DEVICE)

                # Forward pass
                gene_embedding, disease_embedding = model(
                    variant_id=variant_id,
                    gene_id=gene_id,
                    trait_ids=trait_ids,
                    batch_ids=batch_ids
                )

                loss = multi_positive_info_nce_loss(gene_embedding, disease_embedding, num_positives, temperature, norm)
                batch_loss += loss.item()

                # Cosine similarity (compute distance between gene embedding and disease embedding)
                similarities = torch.cosine_similarity(
                    gene_embedding, disease_embedding, dim=-1).cpu().numpy()

                # Construct trues (positives first)
                trues = [1] * num_positives + [0] * num_negatives

                # Save the results
                all_trues.extend(trues)
                all_predictions.extend(similarities)

                # Compute MRR (Mean Reciprocal Rank)
                rank = 1 / (1 + torch.argsort(torch.tensor(similarities),
                            descending=True).cpu().numpy().tolist().index(0))
                all_ranks.append(rank)

            wandb.log({"valid_batch_loss": batch_loss / len(label_batch)})
    # Calculate AUROC, AUPRC, and MRR
    auroc = roc_auc_score(all_trues, all_predictions)
    auprc = average_precision_score(all_trues, all_predictions)
    mrr = sum(all_ranks) / len(all_ranks)

    wandb.log({"AUROC": auroc, "AUPRC": auprc, "MRR": mrr})
    return auroc, auprc, mrr


# Define the model
set_seed_all(SEED)
model = OurModel(
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    num_heads=num_heads,
    num_graph_layers=num_graph_layers,
    data=data,
    variant_x=variant_x,
    gene_local_x=gene_local_x,
    pooling_type=pooling_type,
    dropout_prob=dropout_prob
).to(DEVICE)

# Define optimizer hyperparameters
# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=learning_rate,
#     weight_decay=weight_decay,
#     betas=(momentum_gradient, momentum_square),
# )
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum_gradient,
    weight_decay=weight_decay
)

set_seed_all(SEED)
train_loader = LabelGenerator(
    train_labels, batch_size, num_samples=num_train_samples, shuffle=True)
val_loader = LabelGenerator(
    val_labels, batch_size, num_samples=num_valid_samples, shuffle=False)

for epoch in range(max_epochs):
    # print(f"\nEpoch {epoch+1}/{max_epochs}")
    # wandb.log({"epoch": epoch + 1})

    if epoch % 5 == 0:
        # print(f"\nValidation after Epoch {epoch+1}")
        test(model, val_loader, max_num_negatives, temperature, norm)

    train(model, optimizer, train_loader,
          max_num_negatives, max_positives, temperature, norm)

# print(f"\nFinal Validation after Epoch {max_epochs}")
test(model, val_loader, max_num_negatives)

# Save the model
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")