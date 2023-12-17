import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from data import (BatchType, BidirectionalOneShotIterator, DataReader,
                  ModeType, TrainDataset)
from models import HAKE,  PairRE
from torch.utils.data import DataLoader



def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='runs.py [<args>] [-h | --help]'
    )

    parser.add_argument('-testGNN', type=str, default='')

    parser.add_argument('-train', '--do_train', action='store_true')
    parser.add_argument('-valid', '--do_valid', action='store_true')
    parser.add_argument('-test', '--do_test', action='store_true')

    parser.add_argument('--data_path', type=str, default='DATA')
    parser.add_argument('-data', type=str, default='wiki')
    parser.add_argument('--model', default='HAKE', type=str)

    parser.add_argument('-n', '--negative_sample_size', default=64, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=35.0, type=float)
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-mw', '--modulus_weight', default=3.5, type=float)
    parser.add_argument('-pw', '--phase_weight', default=1.0, type=float)
    parser.add_argument('--db', default=False, type=bool)
    
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default="", type=str)
    
    parser.add_argument('--save', default='EXPS', type=str)
    parser.add_argument('--bs', default='normal', type=str)
    parser.add_argument('--max_steps', default=640000, type=int)
    parser.add_argument('--best_evaluate', type=str, default='F1')
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--init_step', default=0, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    
    parser.add_argument('--log_steps', default=500, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    parser.add_argument('--no_decay', action='store_true', help='Learning rate do not decay')
    parser.add_argument('--rank', type=int, default=100)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-perfix', type=str, default='0.8_', help='perfix of train and test')
    parser.add_argument('-thr', type=float, default=-1, help='score >= thr is recognized as accepting the triple')

    return parser.parse_args(args)


def override_config(args):
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as f:
        args_dict = json.load(f)

    args.save_path = args_dict['save_path']
    args.model = args_dict['model']
    args.hidden_dim = args_dict['hidden_dim']


def save_model(model, optimizer, save_variable_list, args, desc):
    args_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, f'checkpoint.{desc}')
    )


def set_logger(args):
    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    log = logging.debug if mode == "Training average" else logging.info
    for metric in metrics:
        log('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    TTime = time.localtime()
    TTime = '{}{:02d}{:02d}_{:02d}:{:02d}'.format(*TTime)
    args.save_path = os.path.join(args.save, args.data, f'{args.model}-{args.perfix}{os.uname().nodename}-{TTime}')
    if args.init_checkpoint:
        override_config(args)
    elif args.data is None:
        raise ValueError('one of init_checkpoint/data must be choosed.')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args)
    
    args.dataset = os.path.join(args.data_path, args.data, "data")
    data_reader = DataReader(args.dataset, args.perfix)
    num_entity = len(data_reader.entity_dict)
    num_relation = len(data_reader.relation_dict)
    
    logging.info('Model: {}'.format(args.model))
    logging.info('Data Path: {}'.format(args.dataset))
    logging.info('Num Entity: {}'.format(num_entity))
    logging.info('Num Relation: {}'.format(num_relation))

    logging.info('Num Train: {}'.format(len(data_reader.train_data)))
    logging.info('Num Valid: {}'.format(len(data_reader.valid_data)))
    logging.info('Num Test: {}'.format(len(data_reader.test_data)))
    logging.info(f"thr: {args.thr}")

    
    if args.model == 'HAKE':
        kge_model = HAKE(num_entity, num_relation, args.hidden_dim, args.gamma, args.modulus_weight, args.phase_weight)
    elif args.model == 'PairRE':
        kge_model = PairRE(num_entity, num_relation, args.hidden_dim, args.gamma)

    logging.debug('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.debug('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    kge_model = kge_model.cuda()

    if args.do_train:
        train_dataloader_head = DataLoader(
            TrainDataset(data_reader, args.negative_sample_size, BatchType.HEAD_BATCH),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(data_reader, args.negative_sample_size, BatchType.TAIL_BATCH),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )

        warm_up_steps = args.max_steps // 4

    bestF1 = -1.
    bestMRR = -1000
    if args.init_checkpoint:
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, f'checkpoint.{"continue" if args.do_train else "best"}'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train and args.best_evaluate == 'F1':
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            bestF1 = checkpoint['bestF1']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif args.do_train and args.best_evaluate == 'MRR':
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            bestMRR = checkpoint['bestMRR']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Randomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    if args.testGNN:
        gnn_ht = torch.load(args.testGNN)
        ht_scos = [torch.tensor(x).cuda() for x in zip(*gnn_ht.items())]
        if not args.do_train:
            kge_model.GNNtest(kge_model, data_reader, ht_scos, args, step)
            exit()


    kge_model.gamma[0] = args.gamma
    logging.debug('Start Training...')
    logging.debug('init_step = %d' % init_step)
    logging.debug('batch_size = %d' % args.batch_size)
    if not args.do_test:
        logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % kge_model.gamma.item())
    logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    if args.do_train:
        start_time = time.time()
        training_logs = []

        for step in range(init_step, args.max_steps+1):

            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            training_logs.append(log)

            if step >= warm_up_steps:
                if not args.no_decay:
                    current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                if args.best_evaluate == 'F1':
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps,
                        'bestF1': bestF1
                    }
                elif args.best_evaluate == 'MRR':
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps,
                        'bestMRR': bestMRR
                    }
                save_model(kge_model, optimizer, save_variable_list, args, 'continue')

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                trained = step % args.valid_steps + 1e-5
                remain = args.valid_steps - trained
                print(f'\rNext valid:{remain}, eta:{remain * (time.time() - start_time) / trained:.1f}s', end='    ')

            if args.do_test and step % args.valid_steps == 0 and step > init_step + args.valid_steps:
            #if args.do_test and step % args.valid_steps == 0 and step >= args.init_step + args.valid_steps/2:
                if args.db:
                    args.valid_steps *= 2
                print('\r', end='')
                logging.info('Evaluating on Test Dataset...')
                if args.testGNN:
                    metrics = kge_model.GNNtest(kge_model, data_reader, ht_scos, args, step)
                else:
                    metrics = kge_model.test_step(kge_model, data_reader, ModeType.TEST, args, step)
                start_time = time.time()
                log_metrics('Test', step, metrics)
                if args.best_evaluate == 'F1':
                    if bestF1 < metrics['F1']:
                        bestF1 = metrics['F1']
                        save_variable_list = {
                            'step': step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps,
                            'bestF1': bestF1
                        }
                        save_model(kge_model, optimizer, save_variable_list, args, 'best')
                elif args.best_evaluate == 'MRR':
                    if bestMRR < metrics['MRR']:
                        bestF1 = metrics['MRR']
                        save_variable_list = {
                            'step': step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps,
                            'bestMRR': bestMRR
                        }
                        save_model(kge_model, optimizer, save_variable_list, args, 'best')
        if args.best_evaluate == 'F1':
            save_variable_list = {
                'step': step,
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps,
                'bestF1': bestF1
            }
        elif args.best_evaluate == 'MRR':
            save_variable_list = {
                'step': step,
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps,
                'bestMRR': bestMRR
            }
        save_model(kge_model, optimizer, save_variable_list, args, "continue")

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, data_reader, ModeType.VALID, args, step)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, data_reader, ModeType.TEST, args, step)
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    if args.do_train:
        os.system(f"grep INFO {args.save_path}/train.log > {args.save_path}/train.info")
