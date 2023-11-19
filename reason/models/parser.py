import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
import os
# Set the current directory to the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options.test_options import TestOptions

from models import create_seq2seq_net, get_vocab
from executors import get_executor
import utils.utils as utils


class Seq2seqParser():
    """Model interface for seq2seq parser"""

    def __init__(self, opt):
        self.opt = opt
        self.vocab = get_vocab(opt)
        if opt.load_checkpoint_path is not None:
            self.load_checkpoint(opt.load_checkpoint_path)
        else:
            print('| creating new network')
            self.net_params = self._get_net_params(self.opt, self.vocab)
            self.seq2seq = create_seq2seq_net(**self.net_params)
        self.variable_lengths = self.net_params['variable_lengths']
        self.end_id = self.net_params['end_id']
        self.gpu_ids = opt.gpu_ids
        self.criterion = nn.NLLLoss()
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            self.seq2seq.cuda(opt.gpu_ids[0])

    def load_checkpoint(self, load_path):
        print('| loading checkpoint from %s' % load_path)
        checkpoint = torch.load(load_path)
        self.net_params = checkpoint['net_params']
        if 'fix_embedding' in vars(self.opt): # To do: change condition input to run mode
            self.net_params['fix_embedding'] = self.opt.fix_embedding
        self.seq2seq = create_seq2seq_net(**self.net_params)
        self.seq2seq.load_state_dict(checkpoint['net_state'])

    def save_checkpoint(self, save_path):
        checkpoint = {
            'net_params': self.net_params,
            'net_state': self.seq2seq.cpu().state_dict()
        }
        torch.save(checkpoint, save_path)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.seq2seq.cuda(self.gpu_ids[0])

    def set_input(self, x, y=None):
        input_lengths, idx_sorted = None, None
        if self.variable_lengths:
            x, y, input_lengths, idx_sorted = self._sort_batch(x, y)
        self.x = self._to_var(x)
        if y is not None:
            self.y = self._to_var(y)
        else:
            self.y = None
        self.input_lengths = input_lengths
        self.idx_sorted = idx_sorted

    def set_reward(self, reward):
        self.reward = reward

    def supervised_forward(self):
        assert self.y is not None, 'Must set y value'
        output_logprob = self.seq2seq(self.x, self.y, self.input_lengths)
        self.loss = self.criterion(output_logprob[:,:-1,:].contiguous().view(-1, output_logprob.size(2)), self.y[:,1:].contiguous().view(-1))
        return self._to_numpy(self.loss).sum()

    def supervised_backward(self):
        assert self.loss is not None, 'Loss not defined, must call supervised_forward first'
        self.loss.backward()

    def reinforce_forward(self):
        self.rl_seq = self.seq2seq.reinforce_forward(self.x, self.input_lengths)
        self.rl_seq = self._restore_order(self.rl_seq.data.cpu())
        self.reward = None # Need to recompute reward from environment each time a new sequence is sampled
        return self.rl_seq

    def reinforce_backward(self, entropy_factor=0.0):
        assert self.reward is not None, 'Must run forward sampling and set reward before REINFORCE'
        self.seq2seq.reinforce_backward(self.reward, entropy_factor)

    def parse(self):
        output_sequence = self.seq2seq.sample_output(self.x, self.input_lengths)
        output_sequence = output_sequence.unsqueeze(0)
        output_sequence = self._restore_order(output_sequence.data.cpu())
        return output_sequence

    def _get_net_params(self, opt, vocab):
        net_params = {
            'input_vocab_size': len(vocab['question_token_to_idx']),
            'output_vocab_size': len(vocab['program_token_to_idx']),
            'hidden_size': opt.hidden_size,
            'word_vec_dim': opt.word_vec_dim,
            'n_layers': opt.n_layers,
            'bidirectional': opt.bidirectional,
            'variable_lengths': opt.variable_lengths,
            'use_attention': opt.use_attention,
            'encoder_max_len': opt.encoder_max_len,
            'decoder_max_len': opt.decoder_max_len,
            'start_id': opt.start_id,
            'end_id': opt.end_id,
            'word2vec_path': opt.word2vec_path,
            'fix_embedding': opt.fix_embedding,
        }
        return net_params

    def _sort_batch(self, x, y):
        _, lengths = torch.eq(x, self.end_id).max(1)
        lengths += 1
        lengths_sorted, idx_sorted = lengths.sort(0, descending=True)
        x_sorted = x[idx_sorted]
        y_sorted = None
        if y is not None:
            y_sorted = y[idx_sorted]
        lengths_list = lengths_sorted.numpy()
        return x_sorted, y_sorted, lengths_list, idx_sorted

    def _restore_order(self, x):
        if self.idx_sorted is not None:
            inv_idxs = self.idx_sorted.clone()
            inv_idxs.scatter_(0, self.idx_sorted, torch.arange(x.size(0)).long())
            return x[inv_idxs]
        return x

    def _to_var(self, x):
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def _to_numpy(self, x):
        return x.data.cpu().numpy().astype(float)


############## TO COPY IN ANOTHER FOLDER ######################

def find_clevr_question_type(out_mod):
    """Find CLEVR question type according to program modules"""
    if out_mod == 'count':
        q_type = 'count'
    elif out_mod == 'exist':
        q_type = 'exist'
    elif out_mod in ['equal_integer', 'greater_than', 'less_than']:
        q_type = 'compare_num'
    elif out_mod in ['equal_size', 'equal_color', 'equal_material', 'equal_shape']:
        q_type = 'compare_attr'
    elif out_mod.startswith('query'):
        q_type = 'query'
    return q_type


def check_program(pred, gt):
    """Check if the input programs matches"""
    # ground truth programs have a start token as the first entry
    for i in range(len(pred)):
        if pred[i] != gt[i+1]:
            return False
        if pred[i] == 2:
            break
    return True


if __name__ == "__main__" :
    
    # Show content of a .h5 file
    import h5py
    import numpy as np
    import datasets.clevr_questions as clevr_questions
    from torch.utils.data import DataLoader


    f = h5py.File("../data/reason/clevr_h5/clevr_val_questions.h5", "r")

    opt = TestOptions().parse()
    parser = Seq2seqParser(opt)


    print('| running test')
    stats = {
        'count': 0,
        'count_tot': 0,
        'exist': 0,
        'exist_tot': 0,
        'compare_num': 0,
        'compare_num_tot': 0,
        'compare_attr': 0,
        'compare_attr_tot': 0,
        'query': 0,
        'query_tot': 0,
        'correct_ans': 0,
        'correct_prog': 0,
        'total': 0
    }

    dataset = clevr_questions.ClevrQuestionDataset("../data/reason/clevr_h5/clevr_val_questions.h5", 1000, "../data/reason/clevr_h5/clevr_vocab.json")
    
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=0, num_workers=1)
    executor = get_executor(opt)

    for x, y, ans, idx in loader:
        parser.set_input(x, y)
        pred_program = parser.parse()
        y_np, pg_np, idx_np, ans_np = y.numpy(), pred_program.numpy(), idx.numpy(), ans.numpy()

        for i in range(pg_np.shape[0]):
            pred_ans = executor.run(pg_np[i], idx_np[i], 'val', guess=True)
            gt_ans = executor.vocab['answer_idx_to_token'][ans_np[i]]

            q_type = find_clevr_question_type(executor.vocab['program_idx_to_token'][y_np[i][1]])
            if pred_ans == gt_ans:
                stats[q_type] += 1
                stats['correct_ans'] += 1
            if check_program(pg_np[i], y_np[i]):
                stats['correct_prog'] += 1

            stats['%s_tot' % q_type] += 1
            stats['total'] += 1
        print('| %d/%d questions processed, accuracy %f' % (stats['total'], len(loader.dataset), stats['correct_ans'] / stats['total']))

    