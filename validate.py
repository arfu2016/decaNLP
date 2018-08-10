import torch
from util import pad
from metrics import compute_metrics

from utils import compute_bleu_rouge
from utils import normalize

from logger_setup import define_logger
logger_temp = define_logger('validate')


def compute_validation_outputs(model, val_iter, field, optional_names=[]):
    loss, predictions, answers = [], [], []
    outputs = [[] for _ in range(len(optional_names))]
    for batch_idx, batch in enumerate(val_iter):
        l, p = model(batch)
        loss.append(l)
        predictions.append(pad(p, 150, dim=-1, val=field.vocab.stoi['<pad>']))
        a = None
        if hasattr(batch, 'wikisql_id'):
            a = batch.wikisql_id.data.cpu()
        elif hasattr(batch, 'squad_id'):
            a = batch.squad_id.data.cpu()
        elif hasattr(batch, 'woz_id'):
            a = batch.woz_id.data.cpu()
        else:
            a =  pad(batch.answer.data.cpu(), 150, dim=-1, val=field.vocab.stoi['<pad>'])
        answers.append(a)
        for opt_idx, optional_name in enumerate(optional_names):
            outputs[opt_idx].append(getattr(batch, optional_name).data.cpu()) 
    loss = torch.cat(loss, 0) if loss[0] is not None else None
    predictions = torch.cat(predictions, 0)
    answers = torch.cat(answers, 0)
    return loss, predictions, answers, [torch.cat([pad(x, 150, dim=-1, val=field.vocab.stoi['<pad>']) for x in output], 0) for output in outputs]


def get_clip(val_iter):
    return -val_iter.extra if val_iter.extra > 0 else None


def all_reverse(tensor, world_size, field, clip, dim=0):
    if world_size > 1:
        tensor = tensor.float() # tensors must be on cpu and float for all_gather
        all_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.barrier() # all_gather is experimental for gloo, found that these barriers were necessary
        torch.distributed.all_gather(all_tensors, tensor)
        torch.distributed.barrier()
        tensor = torch.cat(all_tensors, 0).long() # tensors must be long for reverse
    # for distributed training, dev sets are padded with extra examples so that the
    # tensors are all of a predictable size for all_gather. This line removes those extra examples
    return field.reverse(tensor)[:clip] 


def gather_results(model, val_iter, field, world_size, optional_names=[]):
    loss, predictions, answers, outputs = compute_validation_outputs(model, val_iter, field, optional_names=optional_names)
    clip = get_clip(val_iter)
    if not hasattr(val_iter.dataset.examples[0], 'squad_id') and not hasattr(val_iter.dataset.examples[0], 'wikisql_id') and not hasattr(val_iter.dataset.examples[0], 'woz_id'):
        answers = all_reverse(answers, world_size, field, clip)
    return loss, all_reverse(predictions, world_size, field, clip), answers, [all_reverse(x, world_size, field, clip) for x in outputs],


def print_results(keys, values, rank=None, num_print=1):
    print()
    start = rank * num_print if rank is not None else 0
    end = start + num_print
    values = [val[start:end] for val in values]
    for ex_idx in range(len(values[0])):
        for key_idx, key in enumerate(keys):
            value = values[key_idx][ex_idx]
            v = value[0] if isinstance(value, list) else value
            print(f'{key}: {repr(v)}')
        print()


def cal_bleu_rouge(keys, values, rank=None, num_print=10000):

    start = rank * num_print if rank is not None else 0
    end = start + num_print
    values = [val[start:end] for val in values]
    ref_answers, pred_answers = [], []
    for ex_idx in range(len(values[0])):
        for key_idx, key in enumerate(keys):
            value = values[key_idx][ex_idx]
            v = value[0] if isinstance(value, list) else value
            # print(f'{key}: {repr(v)}')
            if key == 'greedy':
                pred_answers.append({'answers': str(v)})
            elif key == 'answer':
                ref_answers.append({'answers': str(v), 'question_id': ex_idx})

    # compute the bleu and rouge scores if reference answers is provided
    if len(ref_answers) > 0:
        pred_dict, ref_dict = {}, {}
        for pred, ref in zip(pred_answers, ref_answers):
            question_id = ref['question_id']
            if len(ref['answers']) > 0:
                pred_dict[question_id] = normalize(pred['answers'])
                # 利用utils包，normalize strings to space joined chars
                ref_dict[question_id] = normalize(ref['answers'])
        bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        # pred_dict是预测值，ref_dict是真实值
        # 利用utils包，calculate bleu and rouge metrics
    else:
        bleu_rouge = None

    return bleu_rouge


def validate(task, val_iter, model, logger, field, world_size, rank, num_print=10, args=None):
    model.eval()
    required_names = ['greedy', 'answer']
    optional_names = ['context', 'question']
    loss, predictions, answers, results = gather_results(model, val_iter, field, world_size, optional_names=optional_names)
    predictions = [p.replace('UNK', 'OOV') for p in predictions]
    names = required_names + optional_names 
    if hasattr(val_iter.dataset.examples[0], 'wikisql_id') or hasattr(val_iter.dataset.examples[0], 'squad_id') or hasattr(val_iter.dataset.examples[0], 'woz_id'):
        answers = [val_iter.dataset.all_answers[sid] for sid in answers.tolist()]
    metrics, answers = compute_metrics(predictions, answers, bleu='iwslt' in task or 'multi30k' in task, dialogue='woz' in task,
        rouge='cnn' in task, logical_form='sql' in task, corpus_f1='zre' in task, args=args)
    results = [predictions, answers] + results
    # print('print from line 80 in validate.py')
    logger_temp.debug('Begin to validate and show examples')
    print_results(names, results, rank=rank, num_print=num_print)
    bleu_rouge = cal_bleu_rouge(names, results, rank=rank)
    logger.info('Result on dev set: {}'.format(bleu_rouge))

    return loss, metrics
