import os
import sys
import time
import subprocess
import itertools as it
import tempfile

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
import open3d as o3d
import trimesh

import bpy
import bmesh
from mathutils import Matrix
from .rigutils import ArmatureGenerator

import onnxruntime

import ailia
from model_utils import check_and_download_models  # noqa: E402
import binvox_rw
from rignet_utils import inside_check, sample_on_bone
from rignet_utils import meanshift_cluster, nms_meanshift, flip
from rignet_utils import increase_cost_for_outside_bone
from rignet_utils import RigInfo, TreeNode, primMST_symmetry, loadSkel_recur
from rignet_utils import get_bones, calc_geodesic_matrix, post_filter, assemble_skel_skin
from vis_utils import draw_shifted_pts, show_obj_skel

WEIGHT_JOINTNET_PATH = 'models/gcn_meanshift.onnx'
MODEL_JOINTNET_PATH = 'models/gcn_meanshift.onnx.prototxt'
WEIGHT_ROOTNET_SE_PATH = 'models/rootnet_shape_enc.onnx'
MODEL_ROOTNET_SE_PATH = 'models/rootnet_shape_enc.onnx.prototxt'
WEIGHT_ROOTNET_SA1_PATH = 'models/rootnet_sa1_conv.onnx'
MODEL_ROOTNET_SA1_PATH = 'models/rootnet_sa1_conv.onnx.prototxt'
WEIGHT_ROOTNET_SA2_PATH = 'models/rootnet_sa2_conv.onnx'
MODEL_ROOTNET_SA2_PATH = 'models/rootnet_sa2_conv.onnx.prototxt'
WEIGHT_ROOTNET_SA3_PATH = 'models/rootnet_sa3.onnx'
MODEL_ROOTNET_SA3_PATH = 'models/rootnet_sa3.onnx.prototxt'
WEIGHT_ROOTNET_FP3_PATH = 'models/rootnet_fp3_nn.onnx'
MODEL_ROOTNET_FP3_PATH = 'models/rootnet_fp3_nn.onnx.prototxt'
WEIGHT_ROOTNET_FP2_PATH = 'models/rootnet_fp2_nn.onnx'
MODEL_ROOTNET_FP2_PATH = 'models/rootnet_fp2_nn.onnx.prototxt'
WEIGHT_ROOTNET_FP1_PATH = 'models/rootnet_fp1_nn.onnx'
MODEL_ROOTNET_FP1_PATH = 'models/rootnet_fp1_nn.onnx.prototxt'
WEIGHT_ROOTNET_BL_PATH = 'models/rootnet_back_layers.onnx'
MODEL_ROOTNET_BL_PATH = 'models/rootnet_back_layers.onnx.prototxt'
WEIGHT_BONENET_SA1_PATH = 'models/bonenet_sa1_conv.onnx'
MODEL_BONENET_SA1_PATH = 'models/bonenet_sa1_conv.onnx.prototxt'
WEIGHT_BONENET_SA2_PATH = 'models/bonenet_sa2_conv.onnx'
MODEL_BONENET_SA2_PATH = 'models/bonenet_sa2_conv.onnx.prototxt'
WEIGHT_BONENET_SA3_PATH = 'models/bonenet_sa3.onnx'
MODEL_BONENET_SA3_PATH = 'models/bonenet_sa3.onnx.prototxt'
WEIGHT_BONENET_SE_PATH = 'models/bonenet_shape_enc.onnx'
MODEL_BONENET_SE_PATH = 'models/bonenet_shape_enc.onnx.prototxt'
WEIGHT_BONENET_EF_PATH = 'models/bonenet_expand_joint_feature.onnx'
MODEL_BONENET_EF_PATH = 'models/bonenet_expand_joint_feature.onnx.prototxt'
WEIGHT_BONENET_MT_PATH = 'models/bonenet_mix_transform.onnx'
MODEL_BONENET_MT_PATH = 'models/bonenet_mix_transform.onnx.prototxt'
WEIGHT_SKINNET_PATH = 'models/skinnet.onnx'
MODEL_SKINNET_PATH = 'models/skinnet.onnx.prototxt'

REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/rignet/'

MESH_NORMALIZED = None


def normalize_obj(mesh_v):
    dims = [max(mesh_v[:, 0]) - min(mesh_v[:, 0]),
            max(mesh_v[:, 1]) - min(mesh_v[:, 1]),
            max(mesh_v[:, 2]) - min(mesh_v[:, 2])]
    scale = 1.0 / max(dims)
    pivot = np.array([(min(mesh_v[:, 0]) + max(mesh_v[:, 0])) / 2, min(mesh_v[:, 1]),
                      (min(mesh_v[:, 2]) + max(mesh_v[:, 2])) / 2])
    mesh_v[:, 0] -= pivot[0]
    mesh_v[:, 1] -= pivot[1]
    mesh_v[:, 2] -= pivot[2]
    mesh_v *= scale
    return mesh_v, pivot, scale


def get_tpl_edges(remesh_obj_v, remesh_obj_f):
    edge_index = []
    for v in range(len(remesh_obj_v)):
        face_ids = np.argwhere(remesh_obj_f == v)[:, 0]
        neighbor_ids = set()
        for face_id in face_ids:
            for v_id in range(3):
                if remesh_obj_f[face_id, v_id] != v:
                    neighbor_ids.add(remesh_obj_f[face_id, v_id])
        neighbor_ids = list(neighbor_ids)
        neighbor_ids = [np.array([v, n])[np.newaxis, :] for n in neighbor_ids]
        neighbor_ids = np.concatenate(neighbor_ids, axis=0)
        edge_index.append(neighbor_ids)
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def get_geo_edges(surface_geodesic, remesh_obj_v):
    edge_index = []
    surface_geodesic += 1.0 * np.eye(len(surface_geodesic))  # remove self-loop edge here
    for i in range(len(remesh_obj_v)):
        geodesic_ball_samples = np.argwhere(surface_geodesic[i, :] <= 0.06).squeeze(1)
        if len(geodesic_ball_samples) > 10:
            geodesic_ball_samples = np.random.choice(geodesic_ball_samples, 10, replace=False)
        edge_index.append(np.concatenate((np.repeat(i, len(geodesic_ball_samples))[:, np.newaxis],
                                          geodesic_ball_samples[:, np.newaxis]), axis=1))
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def add_self_loops(
        edge_index, num_nodes=None):
    N = np.max(edge_index) + 1 if num_nodes is None else num_nodes
    loop_index = np.arange(N)
    loop_index = np.repeat(np.expand_dims(loop_index, 0), 2, axis=0)
    edge_index = np.concatenate([edge_index, loop_index], axis=1)
    return edge_index


def calc_surface_geodesic(mesh):
    # We denselu sample 4000 points to be more accuracy.
    samples = mesh.sample_points_poisson_disk(number_of_points=4000)
    pts = np.asarray(samples.points)
    pts_normal = np.asarray(samples.normals)

    time1 = time.time()
    N = len(pts)
    verts_dist = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    verts_nn = np.argsort(verts_dist, axis=1)
    conn_matrix = lil_matrix((N, N), dtype=np.float32)

    for p in range(N):
        nn_p = verts_nn[p, 1:6]
        norm_nn_p = np.linalg.norm(pts_normal[nn_p], axis=1)
        norm_p = np.linalg.norm(pts_normal[p])
        cos_similar = np.dot(pts_normal[nn_p], pts_normal[p]) / (norm_nn_p * norm_p + 1e-10)
        nn_p = nn_p[cos_similar > -0.5]
        conn_matrix[p, nn_p] = verts_dist[p, nn_p]
    [dist, _] = dijkstra(conn_matrix, directed=False, indices=range(N),
                         return_predecessors=True, unweighted=False)

    # replace inf distance with euclidean distance + 8
    # 6.12 is the maximal geodesic distance without considering inf, I add 8 to be safer.
    inf_pos = np.argwhere(np.isinf(dist))
    if len(inf_pos) > 0:
        euc_distance = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
        dist[inf_pos[:, 0], inf_pos[:, 1]] = 8.0 + euc_distance[inf_pos[:, 0], inf_pos[:, 1]]

    verts = np.array(mesh.vertices)
    vert_pts_distance = np.sqrt(np.sum((verts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    vert_pts_nn = np.argmin(vert_pts_distance, axis=0)
    surface_geodesic = dist[vert_pts_nn, :][:, vert_pts_nn]
    time2 = time.time()

    print('surface geodesic calculation: {} seconds'.format((time2 - time1)))
    return surface_geodesic


def create_single_data(mesh_obj):
    """
    create input data for the network. The data is wrapped by Data structure in pytorch-geometric library
    :param mesh_filaname: name of the input mesh
    :return: wrapped data, voxelized mesh, and geodesic distance matrix of all vertices
    """

    # triangulate first
    bm = bmesh.new()
    bm.from_object(mesh_obj, bpy.context.evaluated_depsgraph_get())
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')

    # apply modifiers
    mesh_obj.data.clear_geometry()
    for mod in reversed(mesh_obj.modifiers):
        mesh_obj.modifiers.remove(mod)

    bm.to_mesh(mesh_obj.data)
    bpy.context.evaluated_depsgraph_get()

    # rotate -90 deg on X axis
    mat = Matrix(((1.0, 0.0, 0.0, 0.0),
                  (0.0, 0.0, 1.0, 0.0),
                  (0.0, -1.0, 0, 0.0),
                  (0.0, 0.0, 0.0, 1.0)))

    bmesh.ops.transform(bm, matrix=mat, verts=bm.verts[:])
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    mesh_v = np.asarray([list(v.co) for v in bm.verts])
    mesh_f = np.asarray([[v.index for v in f.verts] for f in bm.faces])

    bm.free()

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_v), o3d.open3d.utility.Vector3iVector(mesh_f))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # renew mesh component list with o3d mesh, for consistency
    mesh_v = np.asarray(mesh.vertices)
    mesh_vn = np.asarray(mesh.vertex_normals)
    mesh_f = np.asarray(mesh.triangles)

    mesh_v, translation_normalize, scale_normalize = normalize_obj(mesh_v)
    mesh_normalized = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh_v),
        triangles=o3d.utility.Vector3iVector(mesh_f))
    global MESH_NORMALIZED
    MESH_NORMALIZED = mesh_normalized

    # vertices
    v = np.concatenate((mesh_v, mesh_vn), axis=1)
    v = v.astype(np.float32)

    # topology edges
    print("     gathering topological edges.")
    tpl_e = get_tpl_edges(mesh_v, mesh_f).T
    tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.shape[0])
    tpl_e = tpl_e.astype(np.int64)

    # surface geodesic distance matrix
    print("     calculating surface geodesic matrix.")
    surface_geodesic = calc_surface_geodesic(mesh)

    # geodesic edges
    print("     gathering geodesic edges.")
    geo_e = get_geo_edges(surface_geodesic, mesh_v).T
    geo_e, _ = add_self_loops(geo_e, num_nodes=v.shape[0])
    geo_e = geo_e.astype(np.int64)

    # batch
    batch = np.zeros(len(v), dtype=np.int64)

    # voxel
    fo_normalized = tempfile.NamedTemporaryFile(suffix='_normalized.obj')
    fo_normalized.close()

    o3d.io.write_triangle_mesh(fo_normalized.name, mesh_normalized)

    # TODO: we might cache the .binvox file somewhere, as in the RigNet quickstart example
    rignet_path = bpy.context.preferences.addons[__package__].preferences.rignet_path
    binvox_exe = os.path.join(rignet_path, "binvox")

    if sys.platform.startswith("win"):
        binvox_exe += ".exe"

    if not os.path.isfile(binvox_exe):
        os.unlink(fo_normalized.name)
        raise FileNotFoundError("binvox executable not found in {0}, please check RigNet path in the addon preferences")

    subprocess.call([binvox_exe, "-d", "88", fo_normalized.name])
    with open(os.path.splitext(fo_normalized.name)[0] + '.binvox', 'rb') as fvox:
        vox = binvox_rw.read_as_3d_array(fvox)

    os.unlink(fo_normalized.name)

    data = dict(
        batch=batch, pos=v[:, 0:3],
        tpl_edge_index=tpl_e, geo_edge_index=geo_e,
    )
    return data, vox, surface_geodesic, translation_normalize, scale_normalize


def predict_joints(input_data, vox, joint_pred_net, threshold, bandwidth=None, mesh_filename=None):
    """
    Predict joints
    :param input_data: wrapped input data
    :param vox: voxelized mesh
    :param joint_pred_net: network for predicting joints
    :param threshold: density threshold to filter out shifted points
    :param bandwidth: bandwidth for meanshift clustering
    :param mesh_filename: mesh filename for visualization
    :return: wrapped data with predicted joints, pair-wise bone representation added.
    """
    data_displacement, _, attn_pred, bandwidth_pred = joint_pred_net(input_data)
    y_pred = data_displacement + input_data.pos
    y_pred_np = y_pred.data.cpu().numpy()
    attn_pred_np = attn_pred.data.cpu().numpy()
    y_pred_np, index_inside = inside_check(y_pred_np, vox)
    attn_pred_np = attn_pred_np[index_inside, :]
    y_pred_np = y_pred_np[attn_pred_np.squeeze() > 1e-3]
    attn_pred_np = attn_pred_np[attn_pred_np.squeeze() > 1e-3]

    # symmetrize points by reflecting
    y_pred_np_reflect = y_pred_np * np.array([[-1, 1, 1]])
    y_pred_np = np.concatenate((y_pred_np, y_pred_np_reflect), axis=0)
    attn_pred_np = np.tile(attn_pred_np, (2, 1))

    # img = draw_shifted_pts(mesh_filename, y_pred_np, weights=attn_pred_np)
    if not bandwidth:
        bandwidth = bandwidth_pred.item()
    y_pred_np = meanshift_cluster(y_pred_np, bandwidth, attn_pred_np, max_iter=40)
    # img = draw_shifted_pts(mesh_filename, y_pred_np, weights=attn_pred_np)

    Y_dist = np.sum(((y_pred_np[np.newaxis, ...] - y_pred_np[:, np.newaxis, :]) ** 2), axis=2)
    density = np.maximum(bandwidth ** 2 - Y_dist, np.zeros(Y_dist.shape))
    density = np.sum(density, axis=0)
    density_sum = np.sum(density)
    y_pred_np = y_pred_np[density / density_sum > threshold]
    attn_pred_np = attn_pred_np[density / density_sum > threshold][:, 0]
    density = density[density / density_sum > threshold]

    # img = draw_shifted_pts(mesh_filename, y_pred_np, weights=attn_pred_np)
    pred_joints = nms_meanshift(y_pred_np, density, bandwidth)
    pred_joints, _ = flip(pred_joints)
    # img = draw_shifted_pts(mesh_filename, pred_joints)

    # prepare and add new data members
    pairs = list(it.combinations(range(pred_joints.shape[0]), 2))
    pair_attr = []
    for pr in pairs:
        dist = np.linalg.norm(pred_joints[pr[0]] - pred_joints[pr[1]])
        bone_samples = sample_on_bone(pred_joints[pr[0]], pred_joints[pr[1]])
        bone_samples_inside, _ = inside_check(bone_samples, vox)
        outside_proportion = len(bone_samples_inside) / (len(bone_samples) + 1e-10)
        attr = np.array([dist, outside_proportion, 1])
        pair_attr.append(attr)
    pairs = np.array(pairs)
    pair_attr = np.array(pair_attr)
    pairs = torch.from_numpy(pairs).float()
    pair_attr = torch.from_numpy(pair_attr).float()
    pred_joints = torch.from_numpy(pred_joints).float()
    joints_batch = torch.zeros(len(pred_joints), dtype=torch.long)
    pairs_batch = torch.zeros(len(pairs), dtype=torch.long)

    input_data.joints = pred_joints
    input_data.pairs = pairs
    input_data.pair_attr = pair_attr
    input_data.joints_batch = joints_batch
    input_data.pairs_batch = pairs_batch
    return input_data


def predict_skeleton(input_data, vox, root_pred_net, bone_pred_net, mesh_filename=None):
    """
    Predict skeleton structure based on joints
    :param input_data: wrapped data
    :param vox: voxelized mesh
    :param root_pred_net: network to predict root
    :param bone_pred_net: network to predict pairwise connectivity cost
    :param mesh_filename: meshfilename for debugging
    :return: predicted skeleton structure
    """
    root_id = getInitId(input_data, root_pred_net)
    pred_joints = input_data.joints.data.cpu().numpy()

    with torch.no_grad():
        connect_prob, _ = bone_pred_net(input_data, permute_joints=False)
        connect_prob = torch.sigmoid(connect_prob)
    pair_idx = input_data.pairs.long().data.cpu().numpy()
    prob_matrix = np.zeros((len(input_data.joints), len(input_data.joints)))
    prob_matrix[pair_idx[:, 0], pair_idx[:, 1]] = connect_prob.data.cpu().numpy().squeeze()
    prob_matrix = prob_matrix + prob_matrix.transpose()
    cost_matrix = -np.log(prob_matrix + 1e-10)
    cost_matrix = increase_cost_for_outside_bone(cost_matrix, pred_joints, vox)

    pred_skel = Info()
    parent, key = primMST_symmetry(cost_matrix, root_id, pred_joints)
    for i in range(len(parent)):
        if parent[i] == -1:
            pred_skel.root = TreeNode('root', tuple(pred_joints[i]))
            break
    loadSkel_recur(pred_skel.root, i, None, pred_joints, parent)
    pred_skel.joint_pos = pred_skel.get_joint_dict()

    return pred_skel


def calc_geodesic_matrix(bones, mesh_v, surface_geodesic, mesh_filename, subsampling=False):
    """
    calculate volumetric geodesic distance from vertices to each bones
    :param bones: B*6 numpy array where each row stores the starting and ending joint position of a bone
    :param mesh_v: V*3 mesh vertices
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: an approaximate volumetric geodesic distance matrix V*B, were (v,b) is the distance from vertex v to bone b
    """

    if subsampling:
        mesh0 = o3d.io.read_triangle_mesh(mesh_filename)
        mesh0 = mesh0.simplify_quadric_decimation(3000)
        o3d.io.write_triangle_mesh(mesh_filename.replace(".obj", "_simplified.obj"), mesh0)
        mesh_trimesh = trimesh.load(mesh_filename.replace(".obj", "_simplified.obj"))
        subsamples_ids = np.random.choice(len(mesh_v), np.min((len(mesh_v), 1500)), replace=False)
        subsamples = mesh_v[subsamples_ids, :]
        surface_geodesic = surface_geodesic[subsamples_ids, :][:, subsamples_ids]
    else:
        mesh_trimesh = trimesh.load(mesh_filename)
        subsamples = mesh_v
    origins, ends, pts_bone_dist = pts2line(subsamples, bones)
    pts_bone_visibility = calc_pts2bone_visible_mat(mesh_trimesh, origins, ends)
    pts_bone_visibility = pts_bone_visibility.reshape(len(bones), len(subsamples)).transpose()
    pts_bone_dist = pts_bone_dist.reshape(len(bones), len(subsamples)).transpose()
    # remove visible points which are too far
    for b in range(pts_bone_visibility.shape[1]):
        visible_pts = np.argwhere(pts_bone_visibility[:, b] == 1).squeeze(1)
        if len(visible_pts) == 0:
            continue
        threshold_b = np.percentile(pts_bone_dist[visible_pts, b], 15)
        pts_bone_visibility[pts_bone_dist[:, b] > 1.3 * threshold_b, b] = False

    visible_matrix = np.zeros(pts_bone_visibility.shape)
    visible_matrix[np.where(pts_bone_visibility == 1)] = pts_bone_dist[np.where(pts_bone_visibility == 1)]
    for c in range(visible_matrix.shape[1]):
        unvisible_pts = np.argwhere(pts_bone_visibility[:, c] == 0).squeeze(1)
        visible_pts = np.argwhere(pts_bone_visibility[:, c] == 1).squeeze(1)
        if len(visible_pts) == 0:
            visible_matrix[:, c] = pts_bone_dist[:, c]
            continue
        for r in unvisible_pts:
            dist1 = np.min(surface_geodesic[r, visible_pts])
            nn_visible = visible_pts[np.argmin(surface_geodesic[r, visible_pts])]
            if np.isinf(dist1):
                visible_matrix[r, c] = 8.0 + pts_bone_dist[r, c]
            else:
                visible_matrix[r, c] = dist1 + visible_matrix[nn_visible, c]
    if subsampling:
        nn_dist = np.sum((mesh_v[:, np.newaxis, :] - subsamples[np.newaxis, ...]) ** 2, axis=2)
        nn_ind = np.argmin(nn_dist, axis=1)
        visible_matrix = visible_matrix[nn_ind, :]
        os.remove(mesh_filename.replace(".obj", "_simplified.obj"))
    return visible_matrix


def calc_geodesic_matrix_2(bones, mesh_v, surface_geodesic, use_sampling=False, decimation=3000, sampling=1500):
    """
    calculate volumetric geodesic distance from vertices to each bones
    :param bones: B*6 numpy array where each row stores the starting and ending joint position of a bone
    :param mesh_v: V*3 mesh vertices
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: an approaximate volumetric geodesic distance matrix V*B, were (v,b) is the distance from vertex v to bone b
    """

    if use_sampling:
        mesh0 = MESH_NORMALIZED
        mesh0 = mesh0.simplify_quadric_decimation(decimation)

        fo_simplified = tempfile.NamedTemporaryFile(suffix='_simplified.obj')
        fo_simplified.close()
        o3d.io.write_triangle_mesh(fo_simplified.name, mesh0)
        mesh_trimesh = trimesh.load(fo_simplified.name)
        os.unlink(fo_simplified.name)

        subsamples_ids = np.random.choice(len(mesh_v), np.min((len(mesh_v), sampling)), replace=False)
        subsamples = mesh_v[subsamples_ids, :]
        surface_geodesic = surface_geodesic[subsamples_ids, :][:, subsamples_ids]
    else:
        fo = tempfile.NamedTemporaryFile(suffix='.obj')
        fo.close()
        o3d.io.write_triangle_mesh(fo.name, MESH_NORMALIZED)
        mesh_trimesh = trimesh.load(fo.name)
        os.unlink(fo.name)
        subsamples = mesh_v

    origins, ends, pts_bone_dist = pts2line(subsamples, bones)
    pts_bone_visibility = calc_pts2bone_visible_mat(mesh_trimesh, origins, ends)
    pts_bone_visibility = pts_bone_visibility.reshape(len(bones), len(subsamples)).transpose()
    pts_bone_dist = pts_bone_dist.reshape(len(bones), len(subsamples)).transpose()
    # remove visible points which are too far
    for b in range(pts_bone_visibility.shape[1]):
        visible_pts = np.argwhere(pts_bone_visibility[:, b] == 1).squeeze(1)
        if len(visible_pts) == 0:
            continue
        threshold_b = np.percentile(pts_bone_dist[visible_pts, b], 15)
        pts_bone_visibility[pts_bone_dist[:, b] > 1.3 * threshold_b, b] = False

    visible_matrix = np.zeros(pts_bone_visibility.shape)
    visible_matrix[np.where(pts_bone_visibility == 1)] = pts_bone_dist[np.where(pts_bone_visibility == 1)]
    for c in range(visible_matrix.shape[1]):
        unvisible_pts = np.argwhere(pts_bone_visibility[:, c] == 0).squeeze(1)
        visible_pts = np.argwhere(pts_bone_visibility[:, c] == 1).squeeze(1)
        if len(visible_pts) == 0:
            visible_matrix[:, c] = pts_bone_dist[:, c]
            continue
        for r in unvisible_pts:
            dist1 = np.min(surface_geodesic[r, visible_pts])
            nn_visible = visible_pts[np.argmin(surface_geodesic[r, visible_pts])]
            if np.isinf(dist1):
                visible_matrix[r, c] = 8.0 + pts_bone_dist[r, c]
            else:
                visible_matrix[r, c] = dist1 + visible_matrix[nn_visible, c]
    if use_sampling:
        nn_dist = np.sum((mesh_v[:, np.newaxis, :] - subsamples[np.newaxis, ...]) ** 2, axis=2)
        nn_ind = np.argmin(nn_dist, axis=1)
        visible_matrix = visible_matrix[nn_ind, :]
    return visible_matrix


def predict_skinning(
        input_data, pred_skel, skin_pred_net, surface_geodesic,
        subsampling=False, decimation=3000, sampling=1500):
    """
    predict skinning
    :param input_data: wrapped input data
    :param pred_skel: predicted skeleton
    :param skin_pred_net: network to predict skinning weights
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: predicted rig with skinning weights information
    """
    global output_folder
    num_nearest_bone = 5
    bones, bone_names, bone_isleaf = get_bones(pred_skel)
    mesh_v = input_data.pos.data.cpu().numpy()
    print("     calculating volumetric geodesic distance from vertices to bone. This step takes some time...")

    geo_dist = calc_geodesic_matrix_2(
        bones, mesh_v, surface_geodesic,
        use_sampling=subsampling, decimation=decimation, sampling=sampling)
    input_samples = []  # joint_pos (x, y, z), (bone_id, 1/D)*5
    loss_mask = []
    skin_nn = []
    for v_id in range(len(mesh_v)):
        geo_dist_v = geo_dist[v_id]
        bone_id_near_to_far = np.argsort(geo_dist_v)
        this_sample = []
        this_nn = []
        this_mask = []
        for i in range(num_nearest_bone):
            if i >= len(bones):
                this_sample += bones[bone_id_near_to_far[0]].tolist()
                this_sample.append(1.0 / (geo_dist_v[bone_id_near_to_far[0]] + 1e-10))
                this_sample.append(bone_isleaf[bone_id_near_to_far[0]])
                this_nn.append(0)
                this_mask.append(0)
            else:
                skel_bone_id = bone_id_near_to_far[i]
                this_sample += bones[skel_bone_id].tolist()
                this_sample.append(1.0 / (geo_dist_v[skel_bone_id] + 1e-10))
                this_sample.append(bone_isleaf[skel_bone_id])
                this_nn.append(skel_bone_id)
                this_mask.append(1)
        input_samples.append(np.array(this_sample)[np.newaxis, :])
        skin_nn.append(np.array(this_nn)[np.newaxis, :])
        loss_mask.append(np.array(this_mask)[np.newaxis, :])

    skin_input = np.concatenate(input_samples, axis=0)
    loss_mask = np.concatenate(loss_mask, axis=0)
    skin_nn = np.concatenate(skin_nn, axis=0)
    skin_input = torch.from_numpy(skin_input).float()
    input_data.skin_input = skin_input

    skin_pred = skin_pred_net(input_data)
    skin_pred = torch.softmax(skin_pred, dim=1)
    skin_pred = skin_pred.data.cpu().numpy()
    skin_pred = skin_pred * loss_mask

    skin_nn = skin_nn[:, 0:num_nearest_bone]
    skin_pred_full = np.zeros((len(skin_pred), len(bone_names)))
    for v in range(len(skin_pred)):
        for nn_id in range(len(skin_nn[v, :])):
            skin_pred_full[v, skin_nn[v, nn_id]] = skin_pred[v, nn_id]
    print("     filtering skinning prediction")
    tpl_e = input_data.tpl_edge_index.data.cpu().numpy()
    skin_pred_full = post_filter(skin_pred_full, tpl_e, num_ring=1)
    skin_pred_full[skin_pred_full < np.max(skin_pred_full, axis=1, keepdims=True) * 0.35] = 0.0
    skin_pred_full = skin_pred_full / (skin_pred_full.sum(axis=1, keepdims=True) + 1e-10)
    skel_res = assemble_skel_skin(pred_skel, skin_pred_full)
    return skel_res


def predict_rig(mesh_obj, bandwidth, threshold, downsample_skinning=True, decimation=3000, sampling=1500):
    print("predicting rig")
    # downsample_skinning is used to speed up the calculation of volumetric geodesic distance
    # and to save cpu memory in skinning calculation.
    # Change to False to be more accurate but less efficient.

    # load all weights
    print("loading all networks...")

    rignet_path = bpy.context.preferences.addons[__package__].preferences.rignet_path
    print("rignet_path---->", rignet_path)

    print('=== JOINETNET ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_JOINTNET_PATH), os.path.join(rignet_path, MODEL_JOINTNET_PATH), REMOTE_PATH)
    print('=== ROOTNET (1/8) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_ROOTNET_SE_PATH), os.path.join(rignet_path, MODEL_ROOTNET_SE_PATH),
        REMOTE_PATH)
    print('=== ROOTNET (2/8) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_ROOTNET_SA1_PATH), os.path.join(rignet_path, MODEL_ROOTNET_SA1_PATH),
        REMOTE_PATH)
    print('=== ROOTNET (3/8) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_ROOTNET_SA2_PATH), os.path.join(rignet_path, MODEL_ROOTNET_SA2_PATH),
        REMOTE_PATH)
    print('=== ROOTNET (4/8) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_ROOTNET_SA3_PATH), os.path.join(rignet_path, MODEL_ROOTNET_SA3_PATH),
        REMOTE_PATH)
    print('=== ROOTNET (5/8) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_ROOTNET_FP3_PATH), os.path.join(rignet_path, MODEL_ROOTNET_FP3_PATH),
        REMOTE_PATH)
    print('=== ROOTNET (6/8) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_ROOTNET_FP2_PATH), os.path.join(rignet_path, MODEL_ROOTNET_FP2_PATH),
        REMOTE_PATH)
    print('=== ROOTNET (7/8) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_ROOTNET_FP1_PATH), os.path.join(rignet_path, MODEL_ROOTNET_FP1_PATH),
        REMOTE_PATH)
    print('=== ROOTNET (8/8) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_ROOTNET_BL_PATH), os.path.join(rignet_path, MODEL_ROOTNET_BL_PATH),
        REMOTE_PATH)
    print('=== BONENET (1/6) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_BONENET_SA1_PATH), os.path.join(rignet_path, MODEL_BONENET_SA1_PATH),
        REMOTE_PATH)
    print('=== BONENET (2/6) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_BONENET_SA2_PATH), os.path.join(rignet_path, MODEL_BONENET_SA2_PATH),
        REMOTE_PATH)
    print('=== BONENET (3/6) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_BONENET_SA3_PATH), os.path.join(rignet_path, MODEL_BONENET_SA3_PATH),
        REMOTE_PATH)
    print('=== BONENET (4/6) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_BONENET_SE_PATH), os.path.join(rignet_path, MODEL_BONENET_SE_PATH),
        REMOTE_PATH)
    print('=== BONENET (5/6) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_BONENET_EF_PATH), os.path.join(rignet_path, MODEL_BONENET_EF_PATH),
        REMOTE_PATH)
    print('=== BONENET (6/6) ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_BONENET_MT_PATH), os.path.join(rignet_path, MODEL_BONENET_MT_PATH),
        REMOTE_PATH)
    print('=== SKINNET ===')
    check_and_download_models(
        os.path.join(rignet_path, WEIGHT_SKINNET_PATH), os.path.join(rignet_path, MODEL_SKINNET_PATH), REMOTE_PATH)

    net_info = {}
    net_info['jointNet'] = onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_JOINTNET_PATH))
    net_info['rootNet'] = {
        'shape_encoder': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_ROOTNET_SE_PATH)),
        'sa1_module': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_ROOTNET_SA1_PATH)),
        'sa2_module': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_ROOTNET_SA2_PATH)),
        'sa3_module': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_ROOTNET_SA3_PATH)),
        'fp3_module': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_ROOTNET_FP3_PATH)),
        'fp2_module': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_ROOTNET_FP2_PATH)),
        'fp1_module': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_ROOTNET_FP1_PATH)),
        'back_layers': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_ROOTNET_BL_PATH)),
    }
    net_info['boneNet'] = {
        'sa1_module': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_BONENET_SA1_PATH)),
        'sa2_module': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_BONENET_SA2_PATH)),
        'sa3_module': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_BONENET_SA3_PATH)),
        'shape_encoder': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_BONENET_SE_PATH)),
        'expand_joint_feature': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_BONENET_EF_PATH)),
        'mix_transform': onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_BONENET_MT_PATH)),
    }
    net_info['skinNet'] = onnxruntime.InferenceSession(os.path.join(rignet_path, WEIGHT_SKINNET_PATH))

    data, vox, surface_geodesic, translation_normalize, scale_normalize = create_single_data(mesh_obj)

    print("predicting joints")
    data = predict_joints(data, vox, jointNet, threshold, bandwidth=bandwidth)

    print("predicting connectivity")
    pred_skeleton = predict_skeleton(data, vox, rootNet, boneNet)
    # pred_skeleton.normalize(scale_normalize, -translation_normalize)

    print("predicting skinning")
    pred_rig = predict_skinning(
        data, pred_skeleton, skinNet, surface_geodesic,
        subsampling=downsample_skinning, decimation=decimation, sampling=sampling)

    # here we reverse the normalization to the original scale and position
    pred_rig.normalize(scale_normalize, -translation_normalize)

    mesh_obj.vertex_groups.clear()

    for obj in bpy.data.objects:
        obj.select_set(False)

    mat = Matrix(((1.0, 0.0, 0.0, 0.0),
                  (0.0, 0, -1.0, 0.0),
                  (0.0, 1, 0, 0.0),
                  (0.0, 0.0, 0.0, 1.0)))
    ArmatureGenerator(pred_rig, mesh_obj).generate(matrix=mat)
