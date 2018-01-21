import numpy as np

from copy import deepcopy

from algorithms import inside_outside_functor, inside_outside_tt, maximize_labeled_recall
from tensors import BaseTensor, TensorTrain

approximations = {
    'exact' : BaseTensor,
    'tt' : TensorTrain,
}

class PcfgParser:
    """A parser model based on probabilistic contex-free grammars
        Args:
            mu: np.ndarray (N x N x m) - array of marginal probabilities of non-terminal k to span the 
                segmend [i, j].
        Returns:
            gamma_splits - np.ndarray (N x N) - the most optimal split for the segmend [i, j] 
            gamma_indices - np.ndarray (N x N) - the most optimal nonterminal to span [i, j]
    """
    def __init__(self, rules_nonterminal, rules_terminal, root_distribution, approximation, rank):
        
        self.approximation = approximation
        self.tensor_wrapper = approximations[approximation]
        
        self.T_data = deepcopy(rules_nonterminal)
        self.T = self.tensor_wrapper(self.T_data, rank=rank)

        self.Q = deepcopy(rules_terminal)
        self.pi = deepcopy(root_distribution)

        self.T_temp = np.zeros_like(rules_nonterminal)
        self.Q_temp = np.zeros_like(rules_terminal)

    def fit(self, sequences, max_iter):

        for k in range(max_iter):
            
            for seq in sequences:
                
                alpha, beta = inside_outside_functor(seq, self.T, self.Q) if approximation != 'tt' else \
                              inside_outside_tt(seq, self.T, self.Q, self.pi)
                self.collect_statistics(seq, alpha, beta)
            
            self.recompute_parameters()


    def parse_tree(self, seq):

        alpha, beta = inside_outside_functor(seq, self.T, self.Q, self.pi) if self.approximation != 'tt' else \
                      inside_outside_tt(seq, self.T, self.Q, self.pi)
        mu = alpha * beta

        return maximize_labeled_recall(mu)


    def collect_statistics(self, seq, alpha, beta):
        
        inner_sum = np.einsum('rpd,sdq->rspq', beta[:, :, :-1], beta[:, 1:, :])

        p, q = np.meshgrid(np.arange(beta.shape[1]), np.arange(beta.shape[1]))

        numerator_zero_mask = np.where(q <= p)
        denominator_zero_mask = np.where(q < p)

        alpha_masked = deepcopy(alpha)
        alpha_masked[:, denominator_zero_mask] = 0.

        denominator = np.einsum('jpq,jpq->j', alpha_masked, beta)

        alpha_masked[:, numerator_zero_mask] = 0.

        numerator = np.einsum('jpq,jrs,rspq->jrs', alpha, self.T_data, inner_sum)

        self.T_temp += numerator / denominator[:, np.newaxis, np.newaxis]
        self.Q_temp[:, seq] += np.einsum('jhh,jhh->j', alpha, beta) / denominator

    def recompute_parameters(self):
        
        self.T_data = self.T_temp / (self.T_temp.sum(axis = [1, 2])[:, np.newaxis, np.newaxis] + self.Q_temp.sum(axis = 1)[:, np,newaxis, np.newaxis])
        self.Q = self.Q_temp / (self.Q_temp.sum(axis = 1)[:, np.newaxis] + self.T_temp.sum(axis = [1, 2])[:, np.newaxis])

        self.T = self.tensor_wrapper(self.T_data)

        self.T_temp = np.zeros_like(self.T_temp)
        self.Q_temp = np.zeros_like(self.Q_temp)