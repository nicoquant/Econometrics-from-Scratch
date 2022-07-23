from numpy.linalg.linalg import matrix_power
import numpy as np

class MarkovChains:

    def __init__(self, transition_matrix):

        self.transition_matrix = transition_matrix

    def long_run_proba(self):

        # vA = v ==> v(A - I) = 0
        A = self.transition_matrix - np.identity(len(self.transition_matrix))

        #Add a line of ones to contrains the variables to sum up as x
        A = np.vstack([np.transpose(A), np.ones(len(A))])

        # Since we compute Ax = b
        # We put b = [0, 0, 0, 1] in case we have a 3 * 3 matrix
        b = np.array([0] * (len(A) - 1) + [1])
        self.pi = np.linalg.lstsq(A, b, rcond=None)[0]

    def proba_reaching_state(self, i, j, n_steps, r_steps):

        '''
        We use the Chapman Komogorov Theorem: P[i,j](n) = ∑ P[i,k](r) * P[k,j](n - r)
        :param i: from i state
        :param j: to j state
        :param n_steps: n_steps to reach j from i
        :param r_steps: the intermediate stop
        :return: Proba of reaching state j from state i after n steps
        '''

        return matrix_power(self.transition_matrix, r_steps)[i] @ matrix_power(self.transition_matrix,n_steps - r_steps)[:,j]

class HiddenMarkov(MarkovChains):

    def add_init(self, emission):
        self.emission = emission

    def forward_algo(self, obs_var):

        '''

        :param emission: determines probabilities of observations given a hidden state
        :param obs_var: list of observed states
        :return: likelihood of a the sequence of state
        '''

        alpha = np.zeros((obs_var.shape[0], self.transition_matrix.shape[0]))
        alpha[0, :] = self.pi * self.emission[:, obs_var[0]]

        for i in range(1, obs_var.shape[0]):
            for j in range(self.transition_matrix.shape[0]):
                alpha[i, j] = alpha[i - 1] @ self.transition_matrix[:, j] * self.emission[j, obs_var[i]]

        return np.sum(alpha[-1, :])

    def find_most_likely_state(self, list_of_hidden_state):
        '''
        Brute force method: Soon be updated for the Viterbi algorithm
        '''
        self.proba_each_state = list(map(self.forward_algo,list_of_hidden_state))

        self.proba_most_likely_state = max(self.proba_each_state)

        self.most_likely_state = list_of_hidden_state[np.where(self.proba_most_likely_state == self.proba_each_state)[0][0]]

        return self.most_likely_state


if __name__ == "__main__":

    X = np.array([[0.5,0.5], [0.3,0.7]])

    mc = MarkovChains(X)
    mc.long_run_proba()
    lr = mc.pi
    pik = mc.proba_reaching_state(i = 0, j = 1, n_steps = 3, r_steps= 1)

    Emission_Matrix = np.array([[0.8,0.2], [0.4, 0.6]])

    obsSequence = np.array([0,0,1])

    hmm = HiddenMarkov(X)
    hmm.long_run_proba()
    hmm.add_init(emission=Emission_Matrix)
    proba_sequence = hmm.forward_algo(obsSequence)

    list_states = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]])
    most_likely_state = hmm.find_most_likely_state(list_states)

