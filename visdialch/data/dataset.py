import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Dataset, HeteroData
import numpy as np
import os
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
from visdialch.data.readers import ImageFeaturesHdfReader, DialogsReader
from visdialch.data.vocabulary import Vocabulary

class VisDialDataset(Dataset):
    def __init__(self,
                 config: Dict[str, Any],
                 dialogs_jsonpath: str,
                 root,
                 filename,
                 test=False,
                 transform=None,
                 pre_transform=None,
                 overfit: bool = False,
                 in_memory: bool = False):
        self.test = test
        self.filename = filename
        self.config = config
        self.dialogs_reader = DialogsReader(dialogs_jsonpath)

        self.vocabulary = Vocabulary(
            config["word_counts_json"], min_count=config["vocab_min_count"]
        )

        # Initialize image features reader according to split.
#         image_features_hdfpath = config["image_features_train_h5"]
        image_features_hdfpath = config["image_features_train_h5"]
        if "val" in self.dialogs_reader.split:
            image_features_hdfpath = config["image_features_val_h5"]
        elif "test" in self.dialogs_reader.split:
            image_features_hdfpath = config["image_features_test_h5"]
        
        self.hdf_reader = ImageFeaturesHdfReader(image_features_hdfpath, in_memory)
        
        # Keep a list of image_ids as primary keys to access data.
        self.image_ids = list(self.dialogs_reader.dialogs.keys())
        if overfit:
            self.image_ids = self.image_ids[:10]

        # self.captions_reader = CaptionReader(caption_jsonpath)
        
        super(VisDialDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def split(self):
        return self.dialogs_reader.split
    
    @property
    def raw_file_names(self):
        return self.filename
    
    @property
    def processed_file_names(self):
#         self.image_ids, self.features = self.hdf_reader.get_node_features()
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.image_ids)]
        else:
            return [f'data_{i}.pt' for i in list(self.image_ids)]
    
    def download(self):
        pass

    def process(self):
        for index, image_id in tqdm(enumerate(self.image_ids)):
            data = HeteroData()
            # construct visual graph
            # Get node features
            image_features = torch.tensor(self.hdf_reader[image_id], dtype=torch.float)
            # Normalize image features at zero-th dimension (since there's no batch dimension).
            if self.config["img_norm"]:
                image_features = normalize(image_features, dim=0, p=2)
            # Get adjacency info
            v_edge_index = self._get_adjacency_info()
            # Get edge features
            v_edge_features = self._get_edge_features()
            
            # add visual graph to HeteroData 
            data['v_object'].x = image_features
            data['v_object', 'relates', 'v_object'].edge_index = v_edge_index
            data['v_object', 'relates', 'v_object'].edge_attr = v_edge_features

            
            # construct semantic graph
            # Retrieve instance for this image_id using json reader.
            visdial_instance = self.dialogs_reader[image_id]
            caption = visdial_instance["caption"]
            dialog = visdial_instance["dialog"]

            # Convert word tokens of caption, question, answer and answer options to integers.
            caption = self.vocabulary.to_indices(caption)
            for i in range(len(dialog)):
                dialog[i]["question"] = self.vocabulary.to_indices(dialog[i]["question"])
                dialog[i]["answer"] = self.vocabulary.to_indices(dialog[i]["answer"])

                for j in range(len(dialog[i]["answer_options"])):
                    dialog[i]["answer_options"][j] = self.vocabulary.to_indices(
                        dialog[i]["answer_options"][j]
                    )
            # pad questions to maximum length
            questions, question_lengths = self._pad_sequences(
                [dialog_round["question"] for dialog_round in dialog]
            )
            
            # get current history
            history, history_lengths = self._get_history(
                caption,
                [dialog_round["question"] for dialog_round in dialog],
                [dialog_round["answer"] for dialog_round in dialog]
            )

            answer_options = []
            answer_option_lengths = []
            for dialog_round in dialog:
                options, option_lengths = self._pad_sequences(dialog_round["answer_options"])
                answer_options.append(options)
                answer_option_lengths.append(option_lengths)
            answer_options = torch.stack(answer_options, 0)

            if "test" not in self.split:
                answer_indices = [dialog_round["gt_index"] for dialog_round in dialog]


            # captions_dic = self.captions_reader[image_id]
            # captions_mul = captions_dic["captions"]
            # captions_new = [] 
            # for i in range(len(captions_mul)):
            #     captions_each = self.vocabulary.to_indices(captions_mul[i])
            #     captions_new.append(captions_each)

            # captions_new ,captions_len = self._pad_captions(captions_new)

            
            
            data['dialogue_entity'].x = history
            data['dialogue_entity', 'relates', 'dialogue_entity'].edge_index = self._get_adjacency_info(num_of_nodes=10)
            data['dialogue_entity', 'relates', 'dialogue_entity'].edge_attr = self._get_edge_features(num_of_nodes=10)

            # add visual semantic edges
            # data['dialogue_entity', 'relates', 'v_object'].edge_index = self._get_adjacency_info(num_of_nodes=10)
            # data['dialogue_entity', 'relates', 'dialogue_entity'].edge_attr = self._get_edge_features(num_of_nodes=10)

            # add current question as global graph attr to HeteroData
            data.questions = questions

            
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))


    def _get_node_features(self, image_ids, features):
        pass

    def _get_edge_features(self, num_of_nodes=36, edge_feature_size = 512):
        """
        For now it generates a random matrix of the shape 
        [num_of_edges, edge_feature_size = 512]
        """
        num_of_edges = num_of_nodes*(num_of_nodes - 1)
        edge_features = torch.rand([num_of_edges, edge_feature_size])
        
        return edge_features
    
    def _get_adjacency_info(self, num_of_nodes=36):
        """
        Return an edge list of a fully connected graph with the 
        given number of nodes
        """
        # Initialize edge index matrix
        E = torch.zeros((2, num_of_nodes * (num_of_nodes - 1)), dtype=torch.long)

        # Populate 1st row
        for node in range(num_of_nodes):
            for neighbor in range(num_of_nodes - 1):
                E[0, node * (num_of_nodes - 1) + neighbor] = node

        # Populate 2nd row
        neighbors = []
        for node in range(num_of_nodes):
            neighbors.append(list(np.arange(node)) + list(np.arange(node+1, num_of_nodes)))
        E[1, :] = torch.Tensor([item for sublist in neighbors for item in sublist])

        return E
    
    
    def len(self):
        return len(self.image_ids)
    
    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data
    
    def _pad_sequences(self, sequences: List[List[int]]):
        """Given tokenized sequences (either questions, answers or answer options, tokenized
        in ``__getitem__``), padding them to maximum specified sequence length. Return as a
        tensor of size ``(*, max_sequence_length)``.

        This method is only called in ``__getitem__``, chunked out separately for readability.

        Parameters
        ----------
        sequences : List[List[int]]
            List of tokenized sequences, each sequence is typically a List[int].

        Returns
        -------
        torch.Tensor, torch.Tensor
            Tensor of sequences padded to max length, and length of sequences before padding.
        """

        for i in range(len(sequences)):
            sequences[i] = sequences[i][: self.config["max_sequence_length"] - 1]
        sequence_lengths = [len(sequence) for sequence in sequences]

        # Pad all sequences to max_sequence_length.
        maxpadded_sequences = torch.full(
            (len(sequences), self.config["max_sequence_length"]),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_sequences = pad_sequence(
            [torch.tensor(sequence) for sequence in sequences],
            batch_first=True, padding_value=self.vocabulary.PAD_INDEX
        )
        maxpadded_sequences[:, :padded_sequences.size(1)] = padded_sequences
        return maxpadded_sequences, sequence_lengths
    
    
    def _get_history(self,
                     caption: List[int],
                     questions: List[List[int]],
                     answers: List[List[int]]):
        # Allow double length of caption, equivalent to a concatenated QA pair.
        caption = caption[: self.config["max_sequence_length"] * 2 - 1]

        for i in range(len(questions)):
            questions[i] = questions[i][: self.config["max_sequence_length"] - 1]

        for i in range(len(answers)):
            answers[i] = answers[i][: self.config["max_sequence_length"] - 1]

        # History for first round is caption, else concatenated QA pair of previous round.
        history = []
        history.append(caption)
        for question, answer in zip(questions, answers):
            history.append(question + answer + [self.vocabulary.EOS_INDEX])
        # Drop last entry from history (there's no eleventh question).
        history = history[:-1]
        max_history_length = self.config["max_sequence_length"] * 2

        if self.config.get("concat_history", False):
            # Concatenated_history has similar structure as history, except it contains
            # concatenated QA pairs from previous rounds.
            concatenated_history = []
            concatenated_history.append(caption)
            for i in range(1, len(history)):
                concatenated_history.append([])
                for j in range(i + 1):
                    concatenated_history[i].extend(history[j])

            max_history_length = self.config["max_sequence_length"] * 2 * len(history)
            history = concatenated_history

        history_lengths = [len(round_history) for round_history in history]
        maxpadded_history = torch.full(
            (len(history), max_history_length),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_history = pad_sequence(
            [torch.tensor(round_history) for round_history in history],
            batch_first=True, padding_value=self.vocabulary.PAD_INDEX
        )
        maxpadded_history[:, :padded_history.size(1)] = padded_history
        return maxpadded_history, history_lengths


    # def _pad_captions(self,sequences:List[List[int]]):

    #     LEN_S = len(sequences) 
    #     if LEN_S > self.config["caption_round_num"]:
    #         for i in range(LEN_S - self.config["caption_round_num"]):
    #             sequences.pop(-1)
    #             #caption_len.pop(-1)


    #     caption_len = []
    #     for i in range(len(sequences)):
    #         LEN = len(sequences[i])
    #         if LEN < self.config["caption_maxlen_each"] :
    #             caption_len.append(len(sequences[i]))
    #             for j in range(self.config["caption_maxlen_each"] - LEN):
    #                 sequences[i].append(0)
    #         elif LEN > self.config["caption_maxlen_each"] :
    #             for j in range(LEN - self.config["caption_maxlen_each"]):
    #                 sequences[i].pop(-1)
    #             caption_len.append(len(sequences[i]))
    #         else:
    #             caption_len.append(len(sequences[i]))

    #     LEN_S = len(sequences)
    #     if LEN_S < self.config["caption_round_num"]:
    #         j = 0
    #         #LENS = len(sequences)
    #         for i in range(self.config["caption_round_num"] - LEN_S):
    #             if j >= LEN_S-1:
    #                 j = 0
    #             else:
    #                 j+=1
    #             sequences.append(sequences[j])
    #             length_new = caption_len[j]
    #             caption_len.append(length_new)

    #     sequences = torch.tensor(sequences).view(self.config["caption_round_num"] ,self.config["caption_maxlen_each"])
      
    #     return sequences,caption_len
