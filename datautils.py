import os.path as osp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from config import PATH_PROCESSED, DEVICE, SEED, BATCH_SIZE
from datautils import *

def load_node_csv(path, index_col, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    assert len(df.index.unique()) == df.shape[0]
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = torch.tensor(df.values, dtype=torch.float32)
    return mapping, x

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    # TODO: edge attributions
    edge_attr = ...
    return edge_index, edge_attr

def create_graph():
    # Load vertices
    gene_mapping, gene_x = load_node_csv(osp.join(PATH_PROCESSED, 'gene_x.csv'), index_col=0)
    trait_mapping, trait_x = load_node_csv(osp.join(PATH_PROCESSED, 'trait_x.csv'), index_col=0)

    # Load edges
    # TODO: gene_to_traits
    gene_to_gene, _ = load_edge_csv(osp.join(PATH_PROCESSED, 'gene_to_gene.csv'), 'gene1', gene_mapping, 'gene2', gene_mapping)
    gene_to_trait, _ = load_edge_csv(osp.join(PATH_PROCESSED, 'gene_to_trait.csv'), 'Gene Name', gene_mapping, 'HPO', trait_mapping)
    trait_to_trait, _ = load_edge_csv(osp.join(PATH_PROCESSED, 'trait_to_trait.csv'), 'HPO 1', trait_mapping, 'HPO 2', trait_mapping)

    # Create a HeteroData
    data = HeteroData()
    data['gene'].x = gene_x    # [8,127 x 968]
    data['trait'].x = trait_x  # [8,526 x 768]
    data['gene', 'to', 'gene'].edge_index = gene_to_gene      # [2 x 171,534]
    data['gene', 'to', 'trait'].edge_index = gene_to_trait    # [2 x 221,916] and its reverse
    data['trait', 'to', 'trait'].edge_index = trait_to_trait  # [2 x 7,732] -> [2 x 15,464]

    # Validation and Transform
    data.validate(raise_on_error=True)
    data = T.ToUndirected(reduce='max', merge=True)(data)

    return data, gene_mapping, trait_mapping

def split_dataset(data, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    
    # Ensure the splits add up to 1.0
    assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must add up to 1.0"
    
    # First, split the data into training and remaining (validation + test)
    train_data, temp_data = train_test_split(data, test_size=(val_size + test_size), random_state=random_state, shuffle=True)
    
    # Now, split the remaining data into validation and test sets
    val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)), random_state=random_state, shuffle=True)
    
    return train_data, val_data, test_data

def label_generator(labels, batch_size, gene_mapping, trait_mapping, variant_mapping):
    # Get the total number of rows
    num_rows = len(labels)
    
    # Split the DataFrame into batches
    for i in range(0, num_rows, batch_size):
        label_batch = labels.iloc[i:i + batch_size]
        yield label_batch
        # rows = []
        # for variant, (gene, disease, traits) in label_batch.iterrows():
        #     rows.append([
        #         variant_mapping[variant],
        #         gene_mapping[gene],
        #         [trait_mapping[trait] for trait in traits],
        #     ])
        # yield rows

def sample_negative_trait_gropus(disease_to_traits, positive_diseases, num_negative_samples):
    # Use numpy.setdiff1d to exclude positive diseases more efficiently
    negative_candidates = np.setdiff1d(disease_to_traits.index, positive_diseases, assume_unique=True)
    
    # Use numpy.random.choice with replace=False for faster sampling
    negative_samples = np.random.choice(negative_candidates, size=min(num_negative_samples, len(negative_candidates)), replace=False)
    
    # Retrieve the hpo_id traits corresponding to the sampled negative diseases
    negative_trait_groups = disease_to_traits.loc[negative_samples, 'hpo_id'].tolist()
    
    return negative_trait_groups