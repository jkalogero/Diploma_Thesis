import torch
from torch import nn


class GenerativeDecoder(nn.Module):
    def __init__(self, config, vocabulary,glove, elmo):
        super().__init__()
        self.config = config

        # self.word_embed = nn.Embedding(
        #     len(vocabulary),
        #     config["word_embedding_size"],
        #     padding_idx=vocabulary.PAD_INDEX,
        # )
        self.glove_embed = nn.Embedding(
            len(vocabulary), config["glove_embedding_size"]
        )
        self.elmo_embed = nn.Embedding(
            len(vocabulary), config["elmo_embedding_size"]
        )
        self.glove_embed.weight.data = glove
        self.elmo_embed.weight.data = elmo
        self.glove_embed.weight.requires_grad = False
        self.elmo_embed.weight.requires_grad = False
        self.embed_change = nn.Linear(
            config["elmo_embedding_size"], config["word_embedding_size"]
        )
        self.answer_rnn = nn.LSTM(
            config["glove_embedding_size"] + config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )

        self.lstm_to_words = nn.Linear(
            self.config["lstm_hidden_size"], len(vocabulary)
        )

        self.dropout = nn.Dropout(p=config["dropout"])
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_output, batch):
        """Given `encoder_output`, learn to autoregressively predict
        ground-truth answer word-by-word during training.
        During evaluation, assign log-likelihood scores to all answer options.
        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """

        if self.training:
            print("GEN WHILE TRAINING")
            ans_in = batch["ans_in"]
            print("ans_in.shape = ", ans_in.shape)
            print("ans_in = ", ans_in)
            batch_size, num_rounds, max_sequence_length = ans_in.size()

            ans_in = ans_in.view(batch_size * num_rounds, max_sequence_length)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         word_embedding_size)
            # answers_embed = self.word_embed(ans_in)
            answers_embed_glove = self.glove_embed(ans_in)
            answers_embed_elmo = self.elmo_embed(ans_in)
            answers_embed_elmo = self.dropout(answers_embed_elmo)
            answers_embed_elmo = self.embed_change(answers_embed_elmo)
            answers_embed = torch.cat((answers_embed_glove,answers_embed_elmo),-1)
            print('answers_embed.shape = ', answers_embed.shape)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, batch_size * num_rounds,
            #         lstm_hidden_size)
            init_hidden = encoder_output.view(1, batch_size * num_rounds, -1)
            init_hidden = init_hidden.repeat(
                self.config["lstm_num_layers"], 1, 1
            )
            init_cell = torch.zeros_like(init_hidden)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         lstm_hidden_size)
            ans_out, (hidden, cell) = self.answer_rnn(
                answers_embed, (init_hidden, init_cell)
            )
            ans_out = self.dropout(ans_out)

            # shape: (batch_size * num_rounds, max_sequence_length,
            #         vocabulary_size)
            ans_word_scores = self.lstm_to_words(ans_out)
            return ans_word_scores

        else:
            print('check1')
            ans_in = batch["opt_in"]
            print("ans_in.shape = ", ans_in.shape)
            print("ans_in = ", ans_in)
            batch_size, num_rounds, num_options, max_sequence_length = (ans_in.size())

            ans_in = ans_in.view(
                batch_size * num_rounds * num_options, max_sequence_length
            )

            # shape: (batch_size * num_rounds * num_options, max_sequence_length
            #         word_embedding_size)
            answers_embed_glove = self.glove_embed(ans_in)
            answers_embed_elmo = self.elmo_embed(ans_in)
            answers_embed_elmo = self.dropout(answers_embed_elmo)
            answers_embed_elmo = self.embed_change(answers_embed_elmo)
            answers_embed = torch.cat((answers_embed_glove,answers_embed_elmo),-1)
            print('check2')
            print('answers_embed.shape val= ', answers_embed.shape)
            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, batch_size * num_rounds * num_options,
            #         lstm_hidden_size)
            print('encoder_output.shape = ', encoder_output.shape)
            init_hidden = encoder_output.view(batch_size, num_rounds, 1, -1)
            init_hidden = init_hidden.repeat(1, 1, num_options, 1)
            init_hidden = init_hidden.view(
                1, batch_size * num_rounds * num_options, -1
            )
            init_hidden = init_hidden.repeat(
                self.config["lstm_num_layers"], 1, 1
            )
            init_cell = torch.zeros_like(init_hidden)
            print('init_cell.shape = ', init_cell.shape)
            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, lstm_hidden_size)
            ans_out, (hidden, cell) = self.answer_rnn(
                answers_embed, (init_hidden, init_cell)
            )
            print('check3')
            print('ans_out.shape = ', ans_out.shape)
            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, vocabulary_size)
            ans_word_scores = self.logsoftmax(self.lstm_to_words(ans_out))
            print('ans_word_scores.shape = ', ans_word_scores.shape)
            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            target_ans_out = batch["opt_out"].view(
                batch_size * num_rounds * num_options, -1
            )
            print('check4')
            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            ans_word_scores = torch.gather(
                ans_word_scores, -1, target_ans_out.unsqueeze(-1)
            ).squeeze()
            ans_word_scores = (
                ans_word_scores * (target_ans_out > 0).float().cuda()
            )  # ugly
            print('check5')
            ans_scores = torch.sum(ans_word_scores, -1)
            print('check6')
            ans_scores = ans_scores.view(batch_size, num_rounds, num_options)
            print('check7')
            return ans_scores