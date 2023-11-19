from .clevr_executor import ClevrExecutor


def get_executor(opt):
    print('| creating %s executor' % opt.dataset)
    if opt.dataset == 'clevr':
        val_scene_json = opt.clevr_val_scene_path
        test_scene_json = opt.clevr_test_scene_path
        vocab_json = opt.clevr_vocab_path
    else:
        raise ValueError('Invalid dataset')
    executor = ClevrExecutor(val_scene_json, test_scene_json, vocab_json)
    return executor