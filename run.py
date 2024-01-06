import sys
import os
# Set the current directory to the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reason.options.test_options import TestOptions

from reason.models import create_seq2seq_net, get_vocab
from reason.models.parser import Seq2seqParser
from reason.executors import get_executor
import reason.utils.utils as utils


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
    import reason.datasets.clevr_questions as clevr_questions
    from torch.utils.data import DataLoader


    f = h5py.File("data/reason/clevr_h5/clevr_val_questions.h5", "r")

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

    dataset = clevr_questions.ClevrQuestionDataset("data/reason/clevr_h5/clevr_val_questions.h5", 1000, "data/reason/clevr_h5/clevr_vocab.json")
    
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=0, num_workers=1)
    executor = get_executor(opt)

    for x, y, ans, idx in loader:
        parser.set_input(x, y)
        pred_program = parser.parse()
        y_np, pg_np, idx_np, ans_np = y.numpy(), pred_program.numpy(), idx.numpy(), ans.numpy()

        # Print the question
        x_list = x.numpy().tolist()
        for i in range(1, len(x_list[0])):
            if x_list[0][i] == 2:
                print("?")
                break
            print(dataset.vocab['question_idx_to_token'][x_list[0][i]], end=" ")
            #TODO Changer les probabilités de choix des objets pour prendre en compte la confiance de la detection d'objet

        for i in range(pg_np.shape[0]):
            pred_ans, scene = executor.run(pg_np[i], idx_np[i], 'val', guess=True)
            print("Scene: %s" % scene)
            _, _, desc = executor.modify_scene(scene)
            print("Modification: %s" % desc)
            gt_ans = executor.vocab['answer_idx_to_token'][ans_np[i]]

            print("Predicted answer: %s" % pred_ans)
            print("Ground truth answer: %s" % gt_ans)

            q_type = find_clevr_question_type(executor.vocab['program_idx_to_token'][y_np[i][1]])
            print("Question type: %s" % q_type)
            if pred_ans == gt_ans:
                stats[q_type] += 1
                stats['correct_ans'] += 1
            if check_program(pg_np[i], y_np[i]):
                stats['correct_prog'] += 1

            stats['%s_tot' % q_type] += 1
            stats['total'] += 1
        print('| %d/%d questions processed, accuracy %f' % (stats['total'], len(loader.dataset), stats['correct_ans'] / stats['total']))
        print("\n")

    