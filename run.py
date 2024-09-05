import ast
import os.path as osp
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from config import PATH_PROCESSED, DEVICE, SEED, BATCH_SIZE
from datautils import *
from nnutils import *

# Step 1: Load graph data
# Creates the graph structure and loads node mappings
data, gene_mapping, trait_mapping = create_graph()
print("Graph structure and mappings successfully created.")

# Step 2: Load variant features
# Reads variant data and feature matrix
variant_mapping, variant_x = load_node_csv(
    osp.join(PATH_PROCESSED, 'variant_x.csv'), index_col='SNPs'
)
print("Variant features loaded from CSV.")

# Step 3: Load labels and split datasets
# Load SNP-to-trait labels for positive samples, convert 'hpo_id' to list
labels = pd.read_csv(
    osp.join(PATH_PROCESSED, 'labels.csv'), 
    index_col='snps', 
    converters={'hpo_id': ast.literal_eval}
)
# Split into training, validation, and test sets
train_labels, val_labels, test_labels = split_dataset(labels, random_state=SEED)
print("Labels loaded and dataset split into training, validation, and test sets.")

# Step 4: Load disease-to-trait mapping for negative sampling
# Read disease-to-trait mappings, converting 'hpo_id' into list format
disease_to_traits = pd.read_csv(
    osp.join(PATH_PROCESSED, 'disease_to_traits.csv'), 
    index_col='disease_index', 
    converters={'hpo_id': ast.literal_eval}
)
print("Disease-to-trait mappings loaded.")

# Define the model
hidden_channels = 256
out_channels = 128
num_heads = 2
num_graph_layers = 1
temperature = 0.7

model = OurModel(
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    num_heads=num_heads,
    num_graph_layers=num_graph_layers,
    data=data,
    variant_input_dim=variant_x.size(1),
    pooling_type='attention',
    dropout_prob=0.3
).to(DEVICE)

# Define optimizer hyperparameters
learning_rate = 1e-5
weight_decay = 1e-2
momentum_gradient = 0.9
momentum_square = 0.95
# optimizer = optim.AdamW(
#     model.parameters(),
#     lr=learning_rate,
#     weight_decay=weight_decay,
#     betas=(momentum_gradient, momentum_square),
# )
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum = momentum_gradient,
    weight_decay = weight_decay
)

# Load the data
num_test_samples = 30
num_negative_samples = 32
epochs = 30
data = data.to(DEVICE)

# Training loop with gradient descent and progress bar
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    num_batches = 0  # Counter for batches

    label_loader = label_generator(train_labels.iloc[:num_test_samples], BATCH_SIZE, gene_mapping, trait_mapping, variant_mapping)
    progress_bar = tqdm(label_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    
    for label_batch in progress_bar:
        optimizer.zero_grad()  # Reset the gradients before each batch
        
        # Process each label batch
        for variant, (gene, disease, traits) in label_batch.iterrows():
            variant_id = variant_mapping[variant]
            gene_id = gene_mapping[gene]
            trait_ids = [trait_mapping[trait] for trait in traits]
            
            # Get positive and negative samples
            positive_relations = labels.loc[labels.index == variant, ['disease_index', 'hpo_id']]
            positive_diseases = positive_relations['disease_index'].to_list()
            positive_trait_groups = positive_relations['hpo_id'].to_list()
            negative_trait_groups = sample_negative_trait_gropus(disease_to_traits, positive_diseases, num_negative_samples=num_negative_samples)
            
            # Combine positive and negative trait groups
            num_positives = len(positive_relations)
            trait_groups = positive_trait_groups + negative_trait_groups
            batch_ids = torch.cat([torch.full((len(group),), i) for i, group in enumerate(trait_groups)])
            trait_ids = torch.tensor([trait_mapping[trait] for trait_group in trait_groups for trait in trait_group])
            
            batch_ids = batch_ids.to(DEVICE)
            trait_ids = trait_ids.to(DEVICE)
            
            # Forward pass
            gene_embedding, disease_embedding = model(
                data=data,
                variant_x=variant_x,
                variant_id=variant_id,
                gene_id=gene_id,
                trait_ids=trait_ids,
                batch_ids=batch_ids
            )
            
            # Compute loss
            # loss = multi_positive_info_nce_loss(gene_embedding, disease_embedding, num_positives, temperature)
            loss = cosine_margin_loss(gene_embedding, disease_embedding, num_positives, margin=0.5)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss and update progress bar
            running_loss += loss.item()
            num_batches += 1  # Increment the batch counter
            progress_bar.set_postfix({'loss': running_loss / num_batches})
    
    # Print loss at the end of the epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/num_batches}")