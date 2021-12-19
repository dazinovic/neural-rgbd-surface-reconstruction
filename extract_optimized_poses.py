import os
import numpy as np
import optimize
from dataloader_util import load_poses
from pose_array import PoseArray


def get_pose_array(expname, iter, basedir='./logs'):

    config = os.path.join(basedir, expname, 'config.txt')
    print('Args:')
    print(open(config, 'r').read())

    parser = optimize.config_parser()
    args = parser.parse_args('--config {} '.format(config))

    # Load poses
    tmp, valid = load_poses(os.path.join(args.datadir, 'trainval_poses.txt'))
    poses = []
    for i in range(len(tmp)):
        if valid[i]:
            poses.append(tmp[i])

    poses = np.array(poses).astype(np.float32)
    poses = poses[::args.trainskip]
    poses[:, :3, 3] += args.translation
    poses[:, :3, 3] *= args.sc_factor
    args.num_training_frames = len(poses)

    # Create pose array
    pose_array = PoseArray(args.num_training_frames)
    pose_array_path = os.path.join(basedir, expname, f'pose_array_{iter:06}.npy')
    print('Reloading pose array from', pose_array_path)
    pose_array.set_weights(np.load(pose_array_path, allow_pickle=True))

    return poses, pose_array, args


def extract_poses(poses, pose_array, args):

    os.makedirs(os.path.join(basedir, expname, 'poses'), exist_ok=True)

    original_poses = poses.copy()
    original_poses[:, :3, 3] /= args.sc_factor
    original_poses[:, :3, 3] -= args.translation
    original_poses = np.reshape(original_poses, [-1, 4])
    np.savetxt(os.path.join(basedir, expname, 'poses', 'poses.txt'), original_poses, fmt="%.6f")

    # Apply pose transformation
    pose_delta = []
    optimized_poses = []
    for idx in range(poses.shape[0]):
        R = pose_array.get_rotation_matrices(np.array([idx, 1])).numpy()[0, :, :]
        t = pose_array.get_translations(np.array([idx, 1])).numpy()[0, :, np.newaxis]

        T = np.concatenate([R, t], -1)
        T = np.concatenate([T, np.array([[0, 0, 0, 1]])], 0)
        pose_delta.append(T)

        poses[idx] = T @ poses[idx]
        optimized_poses.append(poses[idx])

    pose_delta = np.array(pose_delta).astype(np.float32)
    pose_delta = np.reshape(pose_delta, [-1, 4])
    np.savetxt(os.path.join(basedir, expname, 'poses', 'pose_delta.txt'), pose_delta, fmt="%.6f")

    optimized_poses = np.array(optimized_poses).astype(np.float32)
    optimized_poses[:, :3, 3] /= args.sc_factor
    optimized_poses[:, :3, 3] -= args.translation
    optimized_poses = np.reshape(optimized_poses, [-1, 4])
    np.savetxt(os.path.join(basedir, expname, 'poses', 'optimized_poses.txt'), optimized_poses, fmt="%.6f")


if __name__ == '__main__':
    # Checkpoint path information
    experiments = [
        {
            'basedir': './logs',
            'expname': 'whiteroom'
        },
    ]

    iter = 400000

    for e in experiments:
        basedir, expname = e.values()
        print(basedir, expname)

        poses, pose_array, args = get_pose_array(expname, iter, basedir)
        extract_poses(poses, pose_array, args)
