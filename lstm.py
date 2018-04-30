import torch
import torch.autograd as autograd
import torch.nn as nn

class Lstm_functions(nn.Module):
    def init(self, num_hlayers):
        super(Lstm_functions, self).__init__()
        self.params = {}
        self.first_lstm = nn.LSTM(400, 400, 1)
        self.nth_lstm = nn.LSTM(400, 400, 1)
        self.h_states = []     # stores the Hdata and the cellstate as tuples for the ith
                                # time where i is the index
        for i in range(num_hlayers):
            ## the states for the lstms have to be initialised
            h_states.append([])

    def forward_lstm_1(self, xt_, ht_1_, wt_1_, t_):
        xtr = torch.mm(self.params['Wi1'], xt_)
        htr = torch.mm(self.params['W11'], ht_1_)
        wtr = torch.mm(self.params['Ww1'], wt_1_)
        input_lstm = xtr + htr + wtr + self.params['b1']
        out, states = self.first_lstm(input_lstm.view(1, 1, 400), self.h1_states[0][t_1 - 1])
        self.h_states[0].append(states)
        return out

    def forward_lstm_n(self, xt_, ht_1_, h_prev_, wt_, t_, n_):
        xtr = torch.mm(self.params['Wi' + str(n_)], xt_)
        htr = torch.mm(self.params['W' + str(n_) + str(n_)], ht_1_)
        wtr = torch.mm(self.params['Ww' + str(n_)], wt_)
        h_prevtr = torch.mm(self.params['W' + str(n_ - 1) + str(n_)], h_prev_)
        input_lstm = xtr + htr + h_prevtr + wtr + slef.params['b' + str(n_)]
        out, states = self.first_lstm(input_lstm.view(1, 1, 400), self.h_states[n_ - 1][t_ - 1])
        self.h_states[n_ - 1].append(states)
        return out

    def forward():

if __name__ == "__main__":
    Lstm_functions()
