import load_network_model
import os
import scene_bounds

import tensorflow as tf
import numpy as np

import marching_cubes as mcubes
import trimesh


def get_batch_query_fn(query_fn, feature_array, network_fn):

    fn = lambda f, i0, i1: query_fn(f[i0:i1, None, :], viewdirs=tf.zeros_like(f[i0:i1]),
                                    feature_array=feature_array,
                                    pose_array=None,
                                    frame_ids=tf.zeros_like(f[i0:i1, 0], dtype=tf.int32),
                                    deformation_field=None,
                                    c2w_array=None,
                                    network_fn=network_fn)

    return fn


def extract_mesh(query_fn, feature_array, network_fn, args, voxel_size=0.01, isolevel=0.0, scene_name='', mesh_savepath=''):

    # Query network on dense 3d grid of points
    voxel_size *= args.sc_factor  # in "network space"

    tx, ty, tz = scene_bounds.get_scene_bounds(scene_name, voxel_size, True)

    query_pts = np.stack(np.meshgrid(tx, ty, tz, indexing='ij'), -1).astype(np.float32)
    print(query_pts.shape)
    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3])

    fn = get_batch_query_fn(query_fn, feature_array, network_fn)

    chunk = 1024 * 64
    raw = np.concatenate([fn(flat, i, i + chunk)[0].numpy() for i in range(0, flat.shape[0], chunk)], 0)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])
    sigma = raw[..., -1]

    print('Running Marching Cubes')
    vertices, triangles = mcubes.marching_cubes(sigma, isolevel, truncation=3.0)
    print('done', vertices.shape, triangles.shape)

    # normalize vertex positions
    vertices[:, :3] /= np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])

    # Rescale and translate
    scale = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]])
    offset = np.array([tx[0], ty[0], tz[0]])
    vertices[:, :3] = scale[np.newaxis, :] * vertices[:, :3] + offset

    # Transform to metric units
    vertices[:, :3] = vertices[:, :3] / args.sc_factor - args.translation

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    # Transform the mesh to Scannet's coordinate system
    gl_to_scannet = np.array([[1, 0, 0, 0],
                              [0, 0, -1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]]).astype(np.float32).reshape([4, 4])

    mesh.apply_transform(gl_to_scannet)

    if mesh_savepath == '':
        mesh_savepath = os.path.join(args.basedir, args.expname, f"mesh_vs{voxel_size / args.sc_factor.ply}")
    mesh.export(mesh_savepath)

    print('Mesh saved')


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

        # Create nerf model
        args, render_kwargs_test, query_fn, feature_array, network_fn = load_network_model.load_network_model_from_disk(expname, iter, basedir)

        args.basedir = basedir
        args.expname = expname
        mesh_savepath = os.path.join(basedir, expname, f"mesh_color_vs0.01_{iter:06}.ply")

        extract_mesh(query_fn, feature_array, network_fn, args, voxel_size=0.01, scene_name='whiteroom', mesh_savepath=mesh_savepath)
