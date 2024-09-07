import os.path as osp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from config import PATH_PROCESSED, DEVICE, SEED
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
    # Step 1: Load vertices (nodes)
    gene_mapping, gene_x = load_node_csv(
        osp.join(PATH_PROCESSED, 'gene_global_x.csv'), index_col=0)
    trait_mapping, trait_x = load_node_csv(
        osp.join(PATH_PROCESSED, 'trait_x.csv'), index_col=0)
    print(f"Gene features:       ({gene_x.size(0)}, {gene_x.size(1)})")
    print(f"Trait features:      ({trait_x.size(0)}, {trait_x.size(1)})")

    # Step 2: Load edges
    gene_to_gene, _ = load_edge_csv(
        osp.join(PATH_PROCESSED,
                 'gene_to_gene.csv'), 'gene1', gene_mapping, 'gene2', gene_mapping
    )
    gene_to_trait, _ = load_edge_csv(
        osp.join(PATH_PROCESSED,
                 'gene_to_trait.csv'), 'Gene Name', gene_mapping, 'HPO', trait_mapping
    )
    trait_to_trait, _ = load_edge_csv(
        osp.join(PATH_PROCESSED,
                 'trait_to_trait.csv'), 'HPO 1', trait_mapping, 'HPO 2', trait_mapping
    )
    print(f"Gene-to-gene edges:   {gene_to_gene.shape[1]} edges.")
    print(f"Gene-to-trait edges:  {gene_to_trait.shape[1]} edges.")
    print(f"Trait-to-trait edges: {trait_to_trait.shape[1]} edges.")

    # Step 3: Create a HeteroData graph
    data = HeteroData()
    data['gene'].x = gene_x    # Gene features
    data['trait'].x = trait_x  # Trait features
    data['gene', 'to', 'gene'].edge_index = gene_to_gene      # Gene-to-gene edges
    # Gene-to-trait edges
    data['gene', 'to', 'trait'].edge_index = gene_to_trait
    # Trait-to-trait edges
    data['trait', 'to', 'trait'].edge_index = trait_to_trait

    # Step 4: Validate and transform the graph
    data.validate(raise_on_error=True)
    data = T.ToUndirected(reduce='max', merge=True)(data)

    return data, gene_mapping, trait_mapping


def split_dataset(data, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    # Ensure the splits add up to 1.0
    assert train_size + val_size + \
        test_size == 1.0, "Train, validation, and test sizes must add up to 1.0"

    # First, split the data into training and remaining (validation + test)
    train_data, temp_data = train_test_split(data, test_size=(
        val_size + test_size), random_state=random_state, shuffle=True)

    # Now, split the remaining data into validation and test sets
    val_data, test_data = train_test_split(temp_data, test_size=(
        test_size / (val_size + test_size)), random_state=random_state, shuffle=True)

    return train_data, val_data, test_data


class LabelGenerator:
    def __init__(self, labels, batch_size, num_samples=None, shuffle=True, random_state=None):
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = num_samples if num_samples is not None else len(
            labels)
        self.shuffle = shuffle
        self.random_state = random_state

    def __iter__(self):
        labels = self.labels

        # Shuffle the DataFrame if shuffle is True
        if self.shuffle:
            labels = labels.sample(frac=1, random_state=self.random_state)

        # Limit the number of samples
        labels = labels[:self.num_samples]

        # Yield the DataFrame in batches
        for i in range(0, self.num_samples, self.batch_size):
            label_batch = labels.iloc[i:i + self.batch_size]
            yield label_batch


def sample_negative_trait_groups(disease_to_traits, positive_diseases, max_num_negatives=None):
    # Use numpy.setdiff1d to exclude positive diseases efficiently
    negative_candidates = np.setdiff1d(disease_to_traits.index, positive_diseases, assume_unique=True)
    
    # If num_negative_samples is None, use all negative candidates
    if max_num_negatives is None:
        negative_samples = negative_candidates
    else:
        # Otherwise, sample the specified number of negative samples
        negative_samples = np.random.choice(negative_candidates, size=min(max_num_negatives, len(negative_candidates)), replace=False)
    
    # Retrieve the hpo_id traits corresponding to the sampled negative diseases
    negative_trait_groups = disease_to_traits.loc[negative_samples, 'hpo_id'].tolist()

    return negative_trait_groups
