from datasets import load_dataset
import numpy as np
import pickle
import torch


class NgramModel:
    # todo upload bigrams to HF?
    def __init__(self, bigram_path: str, d_vocab: int, batch=1, seq_len=2049, device='cuda'):
        self.device = device
        self.d_vocab = d_vocab
        self.batch = batch
        self.seq_len = seq_len

        with open(bigram_path, "rb") as f:
            bigram_counts = pickle.load(f)

        self.unigrams = (
            torch.tensor(bigram_counts.toarray().astype(np.float32)).sum(dim=1).to(device)
        )
        self.bigrams = torch.sparse_csr_tensor(
            bigram_counts.indptr.astype(np.int64),
            bigram_counts.indices.astype(np.int64),
            bigram_counts.data.astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        self.trigrams = (
            load_dataset('Confirm-Labs/pile_trigrams')
            .drop('id0_chunk')
            .sort(['id0', 'id1'])
        )


    def generate_unigrams(self) -> torch.Tensor:
        return torch.multinomial(self.unigrams, self.batch * self.seq_len).reshape(
            self.batch, self.seq_len
        )


    def sample_bigram(self, prev: torch.Tensor) -> torch.Tensor:
        """Given a batch of previous tokens, sample from a bigram model 
        using conditional distributions stored in a sparse CSR tensor."""
        starts = self.bigrams.crow_indices()[prev]
        ends = self.bigrams.crow_indices()[prev + 1]

        # 0 padding to batch rows with variable numbers of non-zero elements
        token_probs = torch.zeros((self.batch, self.d_vocab), device=self.device)
        token_col_indices = torch.zeros(
            (self.batch, self.d_vocab), dtype=torch.int32, device=self.device
        )
        for i in range(self.batch):
            token_probs[i, : ends[i] - starts[i]] = self.bigrams.values()[
                starts[i] : ends[i]
            ]
            token_col_indices[
                i, : ends[i] - starts[i]
            ] = self.bigrams.col_indices()[starts[i] : ends[i]]

        sampled_value_indices = torch.multinomial(token_probs, 1)
        return torch.gather(
            token_col_indices, 1, sampled_value_indices
        )
        

    def generate_bigrams(self) -> torch.Tensor:
        """Auto-regressively generate bigram model sequence. Initialize each 
        sequence by sampling from a unigram model."""
        result = [torch.multinomial(self.unigrams, self.batch).unsqueeze(1)]
        for _ in range(self.seq_len - 1):
            prev = result[-1]
            result.append(self.sample_bigram(prev))
        return torch.cat(result, dim=1)


    def sample_trigram(self, prev: torch.Tensor) -> torch.Tensor:
        """Given a batch of the previous two tokens, sample from a trigram
        model stored as a HF dataset of trigram counts."""
        id0_data = np.array(self.trigrams["id0"])
        start = np.searchsorted(id0_data, prev[:, 0], side='left')
        end = np.searchsorted(id0_data, prev[:, 0], side='right')
        print('id0 ranges: ', start, end)

        print(type(self.trigrams["id1"]))
        print(type(self.trigrams["id1"][start:end]))
        id1_data = np.array(self.trigrams["id1"][start:end])
        print(type(id1_data[start[0]:end[0]]))
        print(type(id1_data[start:end]))
        start = np.searchsorted(id1_data, prev[0, 1], side='left')
        end = np.searchsorted(id1_data, prev[0, 1], side='right')
        print('id1 range: ', start, end)

        indices = torch.multinomial(
            torch.tensor(self.trigrams["count"][start:end], device=self.device), 
            self.batch)
        return torch.tensor(self.trigrams["id2"][start:end])[indices]


    def generate_trigrams(self) -> torch.Tensor:
        """Auto-regressively generate trigram model sequence. Initialize each 
        sequence by sampling from a unigram model, then a bigram model."""
        result = [torch.multinomial(self.unigrams, self.batch).unsqueeze(1)]
        result.append(self.sample_bigram(result[-1]))

        for _ in self.seq_len - 2:
            prev = result[-2:]
            result.append(self.sample_trigram(prev))
        
        return torch.cat(result, dim=1)
            


def main():
    ngram_model = NgramModel(bigram_path='pythia-deduped-bigrams.pkl', 50277, 2, 2049, 'cpu')
    ngram_model.generate_trigrams()


if __name__ == "__main__":
    main()