# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import copy
import os
import sys
import dnnlib
import re

from dnnlib import util
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults


#----------------------------------------------------------------------------

def train_encoder(network_pkl, resume_pkl, dataset, data_dir, result_dir, num_gpus, total_kimg, gamma, mirror_augment, metrics):
    train     = EasyDict(run_func_name='encoder.training_loop')                # Options for training loop.
    E         = EasyDict(func_name='encoder.E_main')                           # Options for encoding loop.
    E_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for encoder optimizer.
    E_loss    = EasyDict(func_name='encoder.E_loss')                           # Options for encoder loss.
    sched     = EasyDict()                                                     # Options for TrainingSchedule.
    grid      = EasyDict(size='8k', layout='random')                           # Options for setup_snapshot_image_grid().
    sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}                                   # Options for tflib.init_tf().

    if not os.path.exists(data_dir):
        print ('Error: dataset root directory does not exist.')
        sys.exit(1)

    for metric in metrics:
        if metric not in metric_defaults:
            print ('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    train.data_dir = data_dir
    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.image_snapshot_ticks = train.network_snapshot_ticks = 10
    sched.E_lrate_base = 0.002
    sched.minibatch_size_base = 32
    sched.minibatch_gpu_base = 4
    metrics = [metric_defaults[x] for x in metrics]
    desc = 'usad'

    desc += '-' + dataset
    dataset_args = EasyDict(tfrecord_dir=dataset)

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(E_args=E, E_opt_args=E_opt, E_loss_args=E_loss)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.update(resume_pkl=resume_pkl,network_pkl=network_pkl)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = result_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

def encode_real_images(network_pkl, dataset_name, data_dir, num_images, num_snapshots):
    
    """sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command')"""
    
    return

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

_examples = '''examples:
    
  # Train the encoder network
  python %(prog)s train-encoder --network=http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl --data-dir=datasets --dataset=ffhq

  # Encode real images
  python %(prog)s encode-real-images --network=http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl --dataset=car --data-dir=~/datasets

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''Usad encoder.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')
    
    train_encoder_parser = subparsers.add_parser('train-encoder', help='Train the encoder network')
    train_encoder_parser.add_argument('--network', help='Generator pickle filename', dest='network_pkl', required=True)
    train_encoder_parser.add_argument('--resume', help='Encoder pickle filename', dest='resume_pkl', default=None)
    train_encoder_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    train_encoder_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    train_encoder_parser.add_argument('--dataset', help='Training dataset', required=True)
    train_encoder_parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
    train_encoder_parser.add_argument('--total-kimg', help='Training length in thousands of images (default: %(default)s)', metavar='KIMG', default=25000, type=int)
    train_encoder_parser.add_argument('--gamma', help='R1 regularization weight (default is config dependent)', default=None, type=float)
    train_encoder_parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)', default=False, metavar='BOOL', type=_str_to_bool)
    train_encoder_parser.add_argument('--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)', default='fid50k', type=_parse_comma_sep)

    encode_real_images_parser = subparsers.add_parser('project-real-images', help='Project real images')
    encode_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    encode_real_images_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    encode_real_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    encode_real_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    encode_real_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3)
    encode_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    kwargs.pop("command")

    func_name_map = {
        'train-encoder': 'run_encoder.train_encoder',
        'encode-real-images': 'run_encoder.encode_real_images'
    }
    
    run_func_obj = util.get_obj_by_name(func_name_map[subcmd])
    assert callable(run_func_obj)    
    run_func_obj(**vars(args))
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
