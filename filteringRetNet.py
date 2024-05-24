import torch
import torch.nn as nn
import numpy as np
import math

class DatasetBatchIterator:


    def __init__(self, X, Y, batch_size, shuffle=True):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)

        if shuffle:
            index = np.random.permutation(X.shape[0])
            X = self.X[index]
            Y = self.Y[index]

        self.batch_size = batch_size
        self.n_batches = int(math.ceil(X.shape[0] / batch_size))
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        X_batch = torch.LongTensor(self.X[k * bs:(k + 1) * bs])
        Y_batch = torch.FloatTensor(self.Y[k * bs:(k + 1) * bs])

        return X_batch, Y_batch.view(-1, 1)



class NeuralColabFilteringRetNet(nn.Module):

    def __init__(self, user_count, movie_count, retnet_model, hidden_size, device=None):
        super().__init__()

        self.user_count = user_count
        self.movie_count = movie_count
        self.retnet_model = retnet_model
        self.linear = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, user_ids, movie_ids):
        input_ids = torch.stack((user_ids, movie_ids), dim=1)

        embedding = nn.Embedding(max(self.user_count + 1, self.movie_count + 1), self.hidden_size).to(self.device)
        inputs_embeds = embedding(input_ids).to(self.device)

        outputs = self.retnet_model(prev_output_tokens=input_ids,
                                    token_embeddings=inputs_embeds)

        last_hidden_state = outputs[0][:, -1, :]
        linear_output = self.linear(last_hidden_state)
        return linear_output

