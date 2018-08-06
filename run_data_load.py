"""
@Project   : decaNLP
@Module    : run_data_load.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/3/18 6:36 PM
@Desc      : 
"""
import logging
import logging.handlers
from pprint import pformat

from data_squad import arguments
from text import torchtext
from util import get_splits


def initialize_logger(args, rank='data_squad.data_load'):
    logger = logging.getLogger(f'process_{rank}')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(lineno)d - %(message)s')
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
        # 取了tuple的第一个元素，只保留train_data，不要validation data
        # split = torchtext.datasets.generic.SQuAD.splits(fields=FIELD,
        # root=args.data, **kwargs)
        logger.info(f'{task} has {len(split)} training examples')
        logger.debug(type(split))
        train_sets.append(split)

        logger.debug(args.vocab_tasks)

    logger.debug(train_sets)

    # return FIELD, train_sets, val_sets


def main():
    args = arguments.parse()
    if args is None:
        return

    # 给numpy and torch设定seed
    logger = initialize_logger(args)
    logger.info(f'Arguments:\n{pformat(vars(args))}')
    # 调用vars(args)的format函数，得到字符串？
    # pformat是一种format函数，从pprint中引入的

    field, save_dict = None, None

    # field is None
    prepare_data(args, field, logger)
    # field, train_sets, val_sets = prepare_data(args, field, logger)


if __name__ == '__main__':

    # python decaNLP/data_load.py --train_tasks squad --gpus 0
    # python run_data_load.py --train_tasks squad --gpus 0

    main()
