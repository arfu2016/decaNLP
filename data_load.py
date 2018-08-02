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
    formatter = logging.Formatter('%(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def main():
    args = arguments.parse()
    if args is None:
        return
    set_seed(args)
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

    # field, train_sets, val_sets = prepare_data(args, field, logger)


if __name__ == '__main__':
# python decaNLP/data_load.py --train_tasks squad --gpus 0 --train_batch_tokens 5000 --val_batch_size 128

    main()
