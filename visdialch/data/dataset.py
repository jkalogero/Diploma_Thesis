from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from visdialch.data.readers import DialogsReader, DenseAnnotationsReader, ImageFeaturesHdfReader, AdjacencyMatricesReader
from visdialch.data.vocabulary import Vocabulary

class VisDialDataset(Dataset):
    """
    A full representation of VisDial v1.0 (train/val/test) dataset. According to the appropriate
    split, it returns dictionary of question, image, history, ground truth answer, answer options,
    dense annotations etc.        
    """
    def __init__(self,
                 config: Dict[str, Any],
                 dialogs_jsonpath: str,
                 dialogs_adj: str,
                 dense_annotations_jsonpath: Optional[str] = None,
                 overfit: bool = False,
                 in_memory: bool = False,
                 num_workers: int = 1,
                 return_options: bool = True,
                 add_boundary_toks: bool = False,
                 sample_graph: bool = True,
                 load_dialog: bool = False):

        super().__init__()
        self.config = config
        self.return_options = return_options
        self.add_boundary_toks = add_boundary_toks
        self.load_dialog = load_dialog
        self.dialogs_reader = DialogsReader(
            dialogs_jsonpath,
            config,
            num_examples=(5 if overfit else None),
            num_workers=num_workers,
            load_dialog=True
        )

        if "val" in self.split and dense_annotations_jsonpath is not None:
            self.annotations_reader = DenseAnnotationsReader(dense_annotations_jsonpath)
        else:
            self.annotations_reader = None

        self.vocabulary = Vocabulary(
            config["word_counts_json"], min_count=config["vocab_min_count"]
        )

        # Initialize image features reader according to split.
        image_features_hdfpath = config["image_features_train_h5"]
        if "val" in self.dialogs_reader.split:
            image_features_hdfpath = config["image_features_val_h5"]
        elif "test" in self.dialogs_reader.split:
            image_features_hdfpath = config["image_features_test_h5"]

        self.hdf_reader = ImageFeaturesHdfReader(image_features_hdfpath, in_memory)

        # Keep a list of image_ids as primary keys to access data.
        self.image_ids = list(self.dialogs_reader.dialogs.keys())
        if overfit:
            self.image_ids = self.image_ids[:5]
        
        self.adj_reader = AdjacencyMatricesReader(dialogs_adj)


    @property
    def split(self):
        return self.dialogs_reader.split

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # Get image_id, which serves as a primary key for current instance.
        image_id = self.image_ids[index]
        print('image_id = ', image_id)

        # Get image features for this image_id using hdf reader.
        image_features,image_relation = self.hdf_reader[image_id]
        image_features = torch.tensor(image_features, dtype=torch.float)
        image_relation = torch.tensor(image_relation, dtype=torch.float)
        
        # Normalize image features at zero-th dimension (since there's no batch dimension).
        if self.config["img_norm"]:
            image_features = normalize(image_features, dim=0, p=2)

        # Retrieve instance for this image_id using json reader.
        visdial_instance = self.dialogs_reader[image_id]
        caption = visdial_instance["caption"]
        dialog = visdial_instance["dialog"]

        # Convert word tokens of caption, question, answer and answer options to integers.
        caption = self.vocabulary.to_indices(caption)
        for i in range(len(dialog)):
            dialog[i]["question"] = self.vocabulary.to_indices(dialog[i]["question"])
            
            if self.add_boundary_toks:
                dialog[i]["answer"] = self.vocabulary.to_indices(
                    [self.vocabulary.SOS_TOKEN]
                    + dialog[i]["answer"]
                    + [self.vocabulary.EOS_TOKEN]
                )
            else:
                dialog[i]["answer"] = self.vocabulary.to_indices(
                    dialog[i]["answer"]
                )

            if self.return_options:
                for j in range(len(dialog[i]["answer_options"])):
                    if self.add_boundary_toks:
                        dialog[i]["answer_options"][j] = self.vocabulary.to_indices(
                            [self.vocabulary.SOS_TOKEN]
                            + dialog[i]["answer_options"][j]
                            + [self.vocabulary.EOS_TOKEN])
                    else:
                        dialog[i]["answer_options"][j] = self.vocabulary.to_indices(
                            dialog[i]["answer_options"][j])

        questions, question_lengths = self._pad_sequences(
            [dialog_round["question"] for dialog_round in dialog]
        )
        history, history_lengths = self._get_history(
            caption,
            [dialog_round["question"] for dialog_round in dialog],
            [dialog_round["answer"] for dialog_round in dialog]
        )
        answers_in, answer_lengths = self._pad_sequences(
            [dialog_round["answer"][:-1] for dialog_round in dialog]
        )
        answers_out, _ = self._pad_sequences(
            [dialog_round["answer"][1:] for dialog_round in dialog]
        )
        # print("answers_in = ", answers_in)
        # print("answers_out = ", answers_out)
        # alt_history, alt_history_lengths = self._get_history_alt(
        #     caption,
        #     [dialog_round["question"] for dialog_round in dialog],
        #     [dialog_round["answer"] for dialog_round in dialog]
        # )
        # print("HISTORY ALT[0][0][0] = ", [self.vocabulary.index2word[int(index)] for index in alt_history[0][0]])
        # print("HISTORY ALT[0][1][0] = ", [self.vocabulary.index2word[int(index)] for index in alt_history[1][0]])
        # print("HISTORY ALT[0][1][1] = ", [self.vocabulary.index2word[int(index)] for index in alt_history[1][1]])

        # answer_options = []
        # answer_option_lengths = []
        # for dialog_round in dialog:
        #     options, option_lengths = self._pad_sequences(dialog_round["answer_options"])
        #     answer_options.append(options)
        #     answer_option_lengths.append(option_lengths)
        # answer_options = torch.stack(answer_options, 0)

        # if "test" not in self.split:
        #     answer_indices = [dialog_round["gt_index"] for dialog_round in dialog]
        
        # external knowledge
        col, row, data, shape, concepts = self.adj_reader[image_id]


        # Collect everything as tensors for ``collate_fn`` of dataloader to work seemlessly
        # questions, history, etc. are converted to LongTensors, for nn.Embedding input.
        item = {}
        item["img_ids"] = torch.tensor(image_id).long()
        item["img_feat"] = image_features
        item["relations"] = image_relation
        item["ques"] = questions.long()
        item["hist"] = history.long()
        item["ans_in"] = answers_in.long()
        item["ans_out"] = answers_out.long()
        item["ques_len"] = torch.tensor(question_lengths).long()
        item["hist_len"] = torch.tensor(history_lengths).long()
        item["ans_len"] = torch.tensor(answer_lengths).long()
        # item["opt_len"] = torch.tensor(answer_option_lengths).long()
        item["num_rounds"] = torch.tensor(visdial_instance["num_rounds"]).long()
        item['concept_ids'], item['adj_lengths'], _, item['n_rel'], item['adj_data'] = self.load_adj_data(col, row, data, shape, concepts)
        # item['concept_ids'], item['adj_lengths'], _, item['n_rel'] = self.load_adj_data(col, row, data, shape, concepts)

        print("len(item['adj_data')] = ", len(item['adj_data']))
        print("len(item['adj_data')[0][0]] = ", len(item['adj_data'][0][0]))        
        print("len(item['adj_data')[0][1]] = ", len(item['adj_data'][0][1]))

        if self.return_options:
            if self.add_boundary_toks:
                answer_options_in, answer_options_out = [], []
                answer_option_lengths = []
                for dialog_round in dialog:
                    options, option_lengths = self._pad_sequences(
                        [
                            option[:-1]
                            for option in dialog_round["answer_options"]
                        ]
                    )
                    answer_options_in.append(options)

                    options, _ = self._pad_sequences(
                        [
                            option[1:]
                            for option in dialog_round["answer_options"]
                        ]
                    )
                    answer_options_out.append(options)

                    answer_option_lengths.append(option_lengths)
                answer_options_in = torch.stack(answer_options_in, 0)
                answer_options_out = torch.stack(answer_options_out, 0)

                item["opt_in"] = answer_options_in.long()
                item["opt_out"] = answer_options_out.long()
                item["opt_len"] = torch.tensor(answer_option_lengths).long()
            else:
                answer_options = []
                answer_option_lengths = []
                for dialog_round in dialog:
                    options, option_lengths = self._pad_sequences(
                        dialog_round["answer_options"]
                    )
                    answer_options.append(options)
                    answer_option_lengths.append(option_lengths)
                answer_options = torch.stack(answer_options, 0)

                item["opt"] = answer_options.long()
                item["opt_len"] = torch.tensor(answer_option_lengths).long()

            if "test" not in self.split:
                answer_indices = [
                    dialog_round["gt_index"] for dialog_round in dialog
                ]
                item["ans_ind"] = torch.tensor(answer_indices).long()

        

        # Gather dense annotations.
        if "val" in self.split:
            dense_annotations = self.annotations_reader[image_id]
            item["gt_relevance"] = torch.tensor(dense_annotations["gt_relevance"]).float()
            item["round_id"] = torch.tensor(dense_annotations["round_id"]).long()

        print(list(item.keys()))
        for k,v in item.items():
            print('\n\n', k, end=' ')
            if k not in ['img_ids', 'num_rounds'] :
                print(' has len = ', len(v))

        return item

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
        # print("padded_sequences.shape = ", padded_sequences.shape)
        # print("maxpadded_sequences.shape = ", maxpadded_sequences.shape)
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
        # print("DATASET: padded_history.shape = ", padded_history.shape)
        # print("DATASET: maxpadded_history.shape = ", maxpadded_history.shape)
        maxpadded_history[:, :padded_history.size(1)] = padded_history
        return maxpadded_history, history_lengths


    def _get_history_alt(self,
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
        history_lengths = [len(round_history) for round_history in history] # delete later

        if self.config.get("concat_history", False):
            # Concatenated_history has similar structure as history, except it contains
            # concatenated QA pairs from previous rounds.
            concatenated_history = []
            for i in range(len(questions)):
                concatenated_history.append(history[:i+1])
            history = concatenated_history
            
            history_lengths = [len(round_history[-1]) for round_history in history]

        # print("alt history: history.shape = ", [len(el) for el in history])
        # padd rounds to max_sequence_length
        for el in history:
            el += [0]*(max_history_length-len(el))
        maxpadded_history = torch.full(
            (len(history),len(history), max_history_length),
            fill_value=self.vocabulary.PAD_INDEX,
        )
        padded_history = pad_sequence(
            [torch.tensor(round_history) for round_history in history],
            batch_first=True, padding_value=self.vocabulary.PAD_INDEX
        )
        # print("ALT DATASET: padded_history.shape = ", padded_history.shape)
        # print("ALT DATASET: maxpadded_history.shape = ", maxpadded_history.shape)
        maxpadded_history[:, :padded_history.size(1)] = padded_history
        return maxpadded_history, history_lengths


    def _pad_captions(self,sequences:List[List[int]]):

        LEN_S = len(sequences) 
        if LEN_S > self.config["caption_round_num"]:
            for i in range(LEN_S - self.config["caption_round_num"]):
                sequences.pop(-1)
                #caption_len.pop(-1)


        caption_len = []
        for i in range(len(sequences)):
            LEN = len(sequences[i])
            if LEN < self.config["caption_maxlen_each"] :
                caption_len.append(len(sequences[i]))
                for j in range(self.config["caption_maxlen_each"] - LEN):
                    sequences[i].append(0)
            elif LEN > self.config["caption_maxlen_each"] :
                for j in range(LEN - self.config["caption_maxlen_each"]):
                    sequences[i].pop(-1)
                caption_len.append(len(sequences[i]))
            else:
                caption_len.append(len(sequences[i]))

        LEN_S = len(sequences)
        if LEN_S < self.config["caption_round_num"]:
            j = 0
            #LENS = len(sequences)
            for i in range(self.config["caption_round_num"] - LEN_S):
                if j >= LEN_S-1:
                    j = 0
                else:
                    j+=1
                sequences.append(sequences[j])
                length_new = caption_len[j]
                caption_len.append(length_new)

        sequences = torch.tensor(sequences).view(self.config["caption_round_num"] ,self.config["caption_maxlen_each"])
      
        return sequences,caption_len

    def load_adj_data(self, col, row, data, shape, concepts, max_node_num=50, select_random_nodes=True):
        """
        Add inverse relations, pad matrices and keep max length.

        Parameters:
        ===========
        data = (col, row, data, shape, concepts)
        col, row, data: np.arrays of 10 lists each for each round
        shape: shape of coo matrix
        concepts: list of concepts
        """
        # with open(adj_pk_path, 'rb') as fin:
        #     adj_concept_pairs = pickle.load(fin)
        n_rounds = len(col)
        adj_data = []
        # 
        adj_lengths = torch.zeros((n_rounds,), dtype=torch.long)
        # max_node_num = len(concepts[-1]) // 2 # for each round keep only half of the nodes
        concept_ids = torch.zeros((n_rounds, max_node_num), dtype=torch.long)
        # node_type_ids = torch.full((n_rounds, max_node_num), 2, dtype=torch.long)

        # padding
        max_edges = max_node_num * 17
        # i, j = np.array([np.zeros(max_edges) for _ in range(n_rounds)]), np.array([np.zeros(max_edges) for _ in range(n_rounds)])

        adj_lengths_ori = adj_lengths.clone()   # get initial adj len
        n_relations = []
        buffer = []
        for _round, (_col, _row, _data, _shape, _concepts) in tqdm(enumerate(zip(col, row, data, shape, concepts)), total=n_rounds, desc='Loading adj matrices'):
            print('\t\tLEN _ROW = ', len(_row))
            print('\t\tLEN _COL = ', len(_col))
            num_concept = min(len(_concepts), max_node_num)
            adj_lengths_ori[_round] = len(_concepts)

            # select the concepts that will be kept
            concept_ids[_round, :num_concept] = torch.tensor(\
                np.random.choice(_concepts,num_concept, replace=False) if select_random_nodes\
                 else _concepts[:num_concept])

            adj_lengths[_round] = num_concept
            # node_type_ids[_round, :num_concept][torch.tensor(qm, dtype=torch.uint8)[:num_concept]] = 0
            # node_type_ids[_round, :num_concept][torch.tensor(am, dtype=torch.uint8)[:num_concept]] = 1
            _row = np.array(_row)
            _col = np.array(_col)
            # _shape is the shape of coo matrix: (RxN, N)
            n_node = _shape[1] #number of nodes
            # half of the relations, because it is undirected
            half_n_rel = _shape[0] // n_node
            # get the coordinates
            i = _row // n_node # i: number of relation
            # i[_round][:len(_row)] = _row // n_node
            j = _row % n_node # j: number of node
            # j[_round][:len(_row)] = _row % n_node
            mask = (j < max_node_num) & (_col < max_node_num)
            i, j, _col = i[mask], j[mask], _col[mask]
            # the x+17 relation will be the inverse relation of x
            i = np.concatenate((i, i + half_n_rel), 0) # add inverse relations
            j = np.concatenate((j, _col), 0)
            _col = np.concatenate((_col, j), 0)
            adj_data.append((i[:10], j[:10], _col[:10]))  # i, j, k are the coordinates of adj's non-zero entries
            n_relations.append(2*half_n_rel+1)

            f_row = i*n_node +j
            buffer.append((f_row[:50], _col[:50]))



        print('| ori_adj_len: {:.2f} | adj_len: {:.2f} |'.format(adj_lengths_ori.float().mean().item(), adj_lengths.float().mean().item()) +
            ' prune_rate: {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()))
        # print("BEFORE concept_ids.shape = ", concept_ids.shape)
        # print("BEFORE adj_lengths.shape = ", adj_lengths.shape)
        # concept_ids, adj_lengths = [x.view(-1, n_rounds, *x.size()[1:]) for x in (concept_ids, adj_lengths)]
        print("concept_ids.shape = ", concept_ids.shape)
        print("adj_lengths.shape = ", adj_lengths.shape)
        print("BEFORE buffer.shape = ", len(buffer))
        print("BEFORE buffer[0].shape = ", len(buffer[0]))
        print("BEFORE buffer[0][0].shape = ", len(buffer[0][0]))
        # print("BEFORE buffer[0][0][0].shape = ", len(buffer[0][0][0]))
        # print("BEFORE buffer[0][0][1].shape = ", len(buffer[0][0][1]))
        print("BEFORE buffer[0][1].shape = ", len(buffer[0][1]))
        
        # adj_data = torch.tensor(list(map(list, zip(*(iter(adj_data),) * n_rounds))), dtype=torch.float)
        rel = torch.tensor(n_relations)
        return concept_ids, adj_lengths, adj_data, rel, buffer
