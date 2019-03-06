# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A demo script showing how to use the uisrnn package on toy data."""
import os
import glob
import numpy as np

import uisrnn
from uisrnn.data_load import ConversationDataset


SAVED_MODEL_NAME = 'saved_model.uisrnn'


def diarization_experiment(model_args, training_args, inference_args):
  """Experiment pipeline.

  Load data --> train model --> test model --> output result

  Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
  """

  test_record = []

  train_data = np.load('./data/toy_training_data.npz')
  test_data = np.load('./data/toy_testing_data.npz')
  train_sequence = train_data['train_sequence']
  train_cluster_id = train_data['train_cluster_id']
  test_sequences = test_data['test_sequences'].tolist()
  test_cluster_ids = test_data['test_cluster_ids'].tolist()
  #train_data = np.load('./data/training_data.npz')
  #test_data = np.load('./data/testing_data.npz')
  #train_sequence = train_data['train_sequence']
  #train_cluster_id = train_data['train_cluster_id']
  #test_sequences = test_data['test_sequences']
  #test_cluster_ids = test_data['test_cluster_ids']
  #train_sequence = np.load('/mnt/cephfs2/asr/users/fanlu/PyTorch_Speaker_Verification/dvector_data_4/train_sequence.npy')
  #train_cluster_id= np.load('/mnt/cephfs2/asr/users/fanlu/PyTorch_Speaker_Verification/dvector_data_4/train_cluster_id.npy')
  import pdb;pdb.set_trace()
  train_sequence, train_cluster_id = [], []
  for npy in glob.glob('/mnt/cephfs2/asr/users/fanlu/PyTorch_Speaker_Verification/%s/train_sequence*.npy' % training_args.dvector_dir):
    train_sequence.append(np.load(npy))
    train_cluster_id.append(np.load(npy.replace('sequence', 'cluster_id')))
  #train_dataset = ConversationDataset('/mnt/cephfs2/asr/users/fanlu/PyTorch_Speaker_Verification/dvector_data_6', num_permutations = training_args.num_permutations)

  model = uisrnn.UISRNN(model_args)


  # training
  model.fit(train_sequence, train_cluster_id, training_args)
  #model.fit2(train_dataset, training_args)
  os.makedirs(training_args.checkpoint_dir, exist_ok=True)
  model.save(os.path.join(training_args.checkpoint_dir, SAVED_MODEL_NAME))
  #model.save2(os.path.join(training_args.checkpoint_dir, SAVED_MODEL_NAME))

def test_experiment(model_args, training_args, inference_args):
  predicted_cluster_ids = []
  test_record = []
  #test_sequences = np.load('/mnt/cephfs2/asr/users/fanlu/PyTorch_Speaker_Verification/dvector_data_4/test_sequence.npy')
  #test_cluster_ids = np.load('/mnt/cephfs2/asr/users/fanlu/PyTorch_Speaker_Verification/dvector_data_4/test_cluster_id.npy')
  #test_sequences = glob.glob('../PyTorch_Speaker_Verification/dvector_data_200h_12_768.256_0.01_24_40_epoch_80/test_sequence_*.npy')
  #test_sequences = glob.glob('../PyTorch_Speaker_Verification/dvector_data_200h_test_6_768.256_0.01/test_sequence_*.npy')
  test_sequences = glob.glob('../PyTorch_Speaker_Verification/%s/test_sequence_*.npy' % inference_args.dvector_dir_inference)
  # we can also skip training by calling：
  #import pdb;pdb.set_trace()
  model = uisrnn.UISRNN(model_args)
  model.load(inference_args.model)
  # testing
  for test_seq in test_sequences:
    test_sequence = np.load(test_seq)
    test_cluster_id = list(np.load(test_seq.replace("sequence", "cluster_id")))
  #for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
  #for i in range(int(test_sequences.shape[0]/100)):
    #test_sequence = test_sequences[i*100:(i+1)*100]
    #test_cluster_id = list(test_cluster_ids[i*100:(i+1)*100])
    predicted_cluster_id = model.predict(test_sequence, inference_args)
    predicted_cluster_ids.append(predicted_cluster_id)
    accuracy = uisrnn.compute_sequence_match_accuracy(
        test_cluster_id, predicted_cluster_id)
    test_record.append((accuracy, len(test_cluster_id)))
    print('Ground truth labels:')
    print(test_cluster_id)
    print('Predicted labels:')
    print(predicted_cluster_id)
    print('-' * 80)

  output_string = uisrnn.output_result(model_args, training_args, test_record)

  print('Finished diarization experiment')
  print(output_string)


def main():
  """The main function."""
  model_args, training_args, inference_args = uisrnn.parse_arguments()
  if training_args.training:
    diarization_experiment(model_args, training_args, inference_args)
  else:
    test_experiment(model_args, training_args, inference_args)


if __name__ == '__main__':
  main()
