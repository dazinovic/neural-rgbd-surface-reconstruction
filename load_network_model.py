import os
from load_scannet import get_num_training_frames
import optimize


def load_network_model_from_disk(expname, iter, basedir='./logs'):

    config = os.path.join(basedir, expname, 'config.txt')
    print('Args:')
    print(open(config, 'r').read())

    parser = optimize.config_parser()
    ft_str = ''
    if iter is not None:
        ft_str = '--ft_path {}'.format(os.path.join(basedir, expname, f'model_{iter:06}.npy'))
    args = parser.parse_args('--config {} '.format(config) + ft_str)

    args.num_training_frames = get_num_training_frames(args.datadir, trainskip=args.trainskip)
    print(args.num_training_frames)

    # Create nerf model
    _, render_kwargs_test, _, _, models = optimize.create_nerf(args)

    query_fn = render_kwargs_test['network_query_fn']

    network_fn = render_kwargs_test['network_fn']
    if args.N_importance > 0 and not args.share_coarse_fine:
        network_fn = render_kwargs_test['network_fine']

    feature_array = None
    if 'feature_array' in models:
        feature_array = models['feature_array']

    return args, render_kwargs_test, query_fn, feature_array, network_fn
