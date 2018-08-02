"""
@Project   : decaNLP
@Module    : data_load.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/2/18 2:49 PM
@Desc      : 
"""
import logging
import logging.handlers
import os
from pprint import pformat
import torch
from text import torchtext
import arguments
from util import (elapsed_time, get_splits, batch_fn, set_seed)


def initialize_logger(args, rank='main'):
    # set up file logger
    logger = logging.getLogger(f'process_{rank}')
    logger.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(args.log_dir, f'process_{rank}.log'),
        maxBytes=1024*1024*10, backupCount=1)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def prepare_data(args, field, logger):

    if field is None:
        logger.info(f'Constructing field')
        FIELD = torchtext.data.ReversibleField(
            batch_first=True, init_token='<init>', eos_token='<eos>',
            lower=args.lower, include_lengths=True)
    else:
        FIELD = field

    logger.debug(FIELD)

    train_sets, val_sets, vocab_sets = [], [], []
    # train sets, validation sets
    for task in args.train_tasks:
        logger.info(f'Loading {task}')
        # kwargs = {'test': None}
        # kwargs['subsample'] = args.subsample
        # kwargs['validation'] = None
        kwargs = {'test': None,
                  'subsample': args.subsample,
                  # 'subsample': 20000000
                  'validation': None
                  }
        logger.info(f'Adding {task} to training datasets')
        split = get_splits(args, task, FIELD, **kwargs)[0]
        # split = torchtext.datasets.generic.SQuAD.splits(fields=FIELD,
        # root=args.data, **kwargs)
        logger.info(f'{task} has {len(split)} training examples')
        train_sets.append(split)

        if args.vocab_tasks is not None and task in args.vocab_tasks:
            vocab_sets.extend(split)

    logger.debug(train_sets)

    # return FIELD, train_sets, val_sets


def main():
    args = arguments.parse()
    if args is None:
        return
    set_seed(args)
    # 给numpy and torch设定seed
    logger = initialize_logger(args)
    logger.info(f'Arguments:\n{pformat(vars(args))}')
    # 调用vars(args)的format函数，得到字符串？
    # pformat是一种format函数，从pprint中引入的

    field, save_dict = None, None
    # tuple unpacking
    if args.load is not None:
        logger.info(f'Loading field from {os.path.join(args.save, args.load)}')
        save_dict = torch.load(os.path.join(args.save, args.load))
        field = save_dict['field']
        # field is the value in the 'field' key of the data

        logger.info(field)

    # field is None
    prepare_data(args, field, logger)
    # field, train_sets, val_sets = prepare_data(args, field, logger)


if __name__ == '__main__':
# python decaNLP/data_load.py --train_tasks squad --gpus 0 --train_batch_tokens 5000 --val_batch_size 128

    main()
