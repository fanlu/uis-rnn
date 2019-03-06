import glob
import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset

from uisrnn.utils import sample_permuted_segments

class ConversationDataset(Dataset):

  def __init__(self, path, shuffle=True, num_permutations=0):
    # data path
    self.path = path
    self.file_list = glob.glob(os.path.join(path, "train_sequence*.npy"))
    self.shuffle=shuffle
    self.num_permutations = num_permutations
    transit_num = 0
    bias_denominator = 0
    for cluster_ids_f in glob.glob(os.path.join(path, "train_cluster_id*.npy")):
      cluster_ids = np.load(cluster_ids_f)
      bias_denominator += len(cluster_ids)
      for entry in range(len(cluster_ids) - 1):
        transit_num += (cluster_ids[entry] != cluster_ids[entry+1])
    self.bias = (transit_num + 1) / bias_denominator
    self.bias_denominator = bias_denominator
    #self.bias = 0.07603031471933715

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, idx):
    if self.shuffle:
      selected_file = random.sample(self.file_list, 1)[0]  # select random speaker
    else:
      selected_file = self.file_list[idx]

    sequences = np.load(selected_file)        # load utterance spectrogram of selected speaker
    cluster_ids = np.load(selected_file.replace("sequence", "cluster_id"))
    seqs = []
    clusters = []
    seq_lengths = []
    if self.num_permutations > 0:
      unique_id = np.unique(cluster_ids)
      for i in unique_id:
        idx_set = np.where(cluster_ids == i)[0]
        sampled_idx_sets = sample_permuted_segments(idx_set, self.num_permutations)
        for j in range(self.num_permutations):
          seqs.append(sequences[sampled_idx_sets[j], :])
          seq_lengths.append(len(idx_set) + 1)
          clusters.append(cluster_ids[sampled_idx_sets[j]])

    return {'sequence': seqs, 'cluster_id': clusters, 'seq_lengths': seq_lengths}

def collate_fn(data):
  def merge(sequences):
    lengths = [seq['seq'].shape[0]+1 for seq in sequences]
    padded_seqs = np.zeros((max(lengths), len(sequences), sequences[0]['seq'].shape[1]), dtype=np.float32)
    for i, seq in enumerate(sequences):
      end = lengths[i]
      padded_seqs[1:end, i, :] = seq['seq'][:end-1, :]
    return padded_seqs, lengths
  data2 = []
  for d in data:
    for seq in d['sequence']:
      data2.append({"seq": seq})
  # sort a list by sequence length (descending order) to use pack_padded_sequence
  # data2.sort(key=lambda x: x['mel_db'].shape[0], reverse=True)

  seqs, seq_lengths = merge(data2)

  return torch.tensor(seqs), torch.tensor(seq_lengths)
