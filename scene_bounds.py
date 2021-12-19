import numpy as np


def get_scene_bounds(scene_name, voxel_size, interp=False):

    if scene_name == 'scene0000':
        x_min, x_max = -1.2, 1.2
        y_min, y_max = -0.1, 1.1
        z_min, z_max = -1.2, 1.2

    elif scene_name == 'scene0002':
        x_min, x_max = -1.2, 1.2
        y_min, y_max = -0.1, 1.5
        z_min, z_max = -1.2, 1.2

    elif scene_name == 'scene0005':
        x_min, x_max = -1.38, 1.42
        y_min, y_max = -0.1, 0.9
        z_min, z_max = -1.22, 1.58

    elif scene_name == 'scene0012':
        x_min, x_max = -1.2, 1.2
        y_min, y_max = -0.1, 1.1
        z_min, z_max = -1.2, 1.2

    elif scene_name == 'scene0050':
        x_min, x_max = -1.0, 1.0
        y_min, y_max = 0.0, 1.6
        z_min, z_max = -1.0, 1.0

    elif scene_name == 'scene0054':
        x_min, x_max = -1.4, 1.4
        y_min, y_max = -0.3, 1.4
        z_min, z_max = -1.4, 1.4

    elif scene_name == 'whiteroom':
        x_min, x_max = -1.2, 1.0
        y_min, y_max = -1.3, 0.9
        z_min, z_max = -0.8, 0.8

    elif scene_name == 'kitchen':
        x_min, x_max = -1.0, 1.4
        y_min, y_max = -1.4, 1.0
        z_min, z_max = -0.8, 1.0

    elif scene_name == 'breakfast':
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -0.9, 0.9
        z_min, z_max = -1.0, 1.1

    elif scene_name == 'staircase':
        x_min, x_max = -1.2, 1.1
        y_min, y_max = -1.1, 1.2
        z_min, z_max = -0.8, 1.2

    elif scene_name == 'icl_living_room':
        x_min, x_max = -1.1, 1.1
        y_min, y_max = -1.1, 1.1
        z_min, z_max = -0.6, 0.5

    elif scene_name == 'complete_kitchen':
        x_min, x_max = -1.2, 1.2
        y_min, y_max = -0.9, 0.9
        z_min, z_max = -0.6, 0.6

    elif scene_name == 'green_room':
        x_min, x_max = -0.85, 0.65
        y_min, y_max = -1.1, 1.1
        z_min, z_max = -0.8, 0.6

    elif scene_name == 'grey_white_room':
        x_min, x_max = -0.62, 0.62
        y_min, y_max = -0.83, 0.83
        z_min, z_max = -0.56, 0.6

    elif scene_name == 'morning_apartment':
        x_min, x_max = -0.86, 0.86
        y_min, y_max = -1.0, 0.94
        z_min, z_max = -0.75, 0.75

    elif scene_name == 'thin_objects':
        x_min, x_max = -0.25, 1.45
        y_min, y_max = 0.1, 1.7
        z_min, z_max = -1.25, 0.0

    else:
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        z_min, z_max = -1.0, 1.0

    if interp:
        x_min = x_min - 0.5 * voxel_size
        y_min = y_min - 0.5 * voxel_size
        z_min = z_min - 0.5 * voxel_size

        x_max = x_max + 0.5 * voxel_size
        y_max = y_max + 0.5 * voxel_size
        z_max = z_max + 0.5 * voxel_size

    Nx = round((x_max - x_min) / voxel_size + 0.0005)
    Ny = round((y_max - y_min) / voxel_size + 0.0005)
    Nz = round((z_max - z_min) / voxel_size + 0.0005)

    tx = np.linspace(x_min, x_max, Nx + 1)
    ty = np.linspace(y_min, y_max, Ny + 1)
    tz = np.linspace(z_min, z_max, Nz + 1)

    return tx, ty, tz
