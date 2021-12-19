import os
import imageio
from dataloader_util import *


def get_training_poses(basedir, translation=0.0, sc_factor=1.0, trainskip=1):
    all_poses, valid = load_poses(os.path.join(basedir, 'trainval_poses.txt'))

    train_frames = []
    for idx in range(0, len(all_poses), trainskip):
        if valid[idx]:
            train_frames.append(idx)

    all_poses = np.array(all_poses).astype(np.float32)
    training_poses = all_poses[train_frames]

    training_poses[:, :3, 3] += translation
    training_poses[:, :3, 3] *= sc_factor

    return training_poses


def get_num_training_frames(basedir, trainskip):
    poses = get_training_poses(basedir, trainskip=trainskip)

    return poses.shape[0]


def get_intrinsics(basedir, crop):
    depth = imageio.imread(os.path.join(basedir, 'depth_filtered', 'depth0.png'))
    H, W = depth.shape[:2]
    H = H - crop / 2
    W = W - crop / 2
    focal = load_focal_length(os.path.join(basedir, 'focal.txt'))

    return H, W, focal


def load_scannet_data(basedir, trainskip, downsample_factor=1, translation=0.0, sc_factor=1., crop=0):

    # Get image filenames, poses and intrinsics
    img_files = [f for f in sorted(os.listdir(os.path.join(basedir, 'images')), key=alphanum_key) if f.endswith('png')]
    depth_files = [f for f in sorted(os.listdir(os.path.join(basedir, 'depth_filtered')), key=alphanum_key) if f.endswith('png')]
    all_poses, valid_poses = load_poses(os.path.join(basedir, 'trainval_poses.txt'))

    # Train, val and test split
    num_frames = len(img_files)
    train_frame_ids = list(range(0, num_frames, trainskip))

    # Lists for the data to load into
    images = []
    depth_maps = []
    poses = []
    frame_indices = []

    # Read images and depth maps for which valid poses exist
    for i in train_frame_ids:
        if valid_poses[i]:
            img = imageio.imread(os.path.join(basedir, 'images', img_files[i]))
            depth = imageio.imread(os.path.join(basedir, 'depth_filtered', depth_files[i]))

            images.append(img)
            depth_maps.append(depth)
            poses.append(all_poses[i])
            frame_indices.append(i)

    # Map images to [0, 1] range
    images = (np.array(images) / 255.).astype(np.float32)

    # Convert depth to meters, then to "network units"
    depth_shift = 1000.0
    depth_maps = (np.array(depth_maps) / depth_shift).astype(np.float32)
    depth_maps *= sc_factor
    depth_maps = depth_maps[..., np.newaxis]

    poses = np.array(poses).astype(np.float32)
    poses[:, :3, 3] += translation
    poses[:, :3, 3] *= sc_factor

    # Intrinsics
    H, W = depth_maps[0].shape[:2]
    focal = load_focal_length(os.path.join(basedir, 'focal.txt'))

    # Resize color frames to match depth
    images = resize_images(images, H, W)

    # Crop the undistortion artifacts
    if crop > 0:
        images = images[:, crop:-crop, crop:-crop, :]
        depth_maps = depth_maps[:, crop:-crop, crop:-crop, :]
        H, W = depth_maps[0].shape[:2]

    if downsample_factor > 1:
        H = H//downsample_factor
        W = W//downsample_factor
        focal = focal/downsample_factor
        images = resize_images(images, H, W)
        depth_maps = resize_images(depth_maps, H, W, interpolation=cv2.INTER_NEAREST)

    return images, depth_maps, poses, [H, W, focal], frame_indices
