import tensorflow as tf
from nerf_helpers import img2mae
from nerf_helpers import img2mse


def compute_loss(prediction, target, loss_type='l2'):
    if loss_type == 'l2':
        return img2mse(prediction, target)
    elif loss_type == 'l1':
        return img2mae(prediction, target)

    raise Exception('Unsupported loss type')


def get_masks(z_vals, target_d, truncation):

    front_mask = tf.where(z_vals < (target_d - truncation), tf.ones_like(z_vals), tf.zeros_like(z_vals))
    back_mask = tf.where(z_vals > (target_d + truncation), tf.ones_like(z_vals), tf.zeros_like(z_vals))
    depth_mask = tf.where(target_d > 0.0, tf.ones_like(target_d), tf.zeros_like(target_d))
    sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

    num_fs_samples = tf.math.count_nonzero(front_mask, dtype=tf.float32)
    num_sdf_samples = tf.math.count_nonzero(sdf_mask, dtype=tf.float32)
    num_samples = num_sdf_samples + num_fs_samples
    fs_weight = 1.0 - num_fs_samples / num_samples
    sdf_weight = 1.0 - num_sdf_samples / num_samples

    return front_mask, sdf_mask, fs_weight, sdf_weight


def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation, loss_type):

    front_mask, sdf_mask, fs_weight, sdf_weight = get_masks(z_vals, target_d, truncation)

    fs_loss = compute_loss(predicted_sdf * front_mask, tf.ones_like(predicted_sdf) * front_mask, loss_type) * fs_weight
    sdf_loss = compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask, target_d * sdf_mask, loss_type) * sdf_weight

    return fs_loss, sdf_loss


def get_depth_loss(predicted_depth, target_d, loss_type='l2'):
    depth_mask = tf.where(target_d > 0, tf.ones_like(target_d), tf.zeros_like(target_d))
    eps = 1e-4
    num_pixel = tf.size(depth_mask, out_type=tf.float32)
    num_valid = tf.math.count_nonzero(depth_mask, dtype=tf.float32) + eps
    depth_valid_weight = num_pixel / num_valid

    return compute_loss(predicted_depth[..., tf.newaxis] * depth_mask, target_d * depth_mask, loss_type) * depth_valid_weight

