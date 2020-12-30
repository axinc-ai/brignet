import os
import sys
import time
import subprocess
import itertools as it
import tempfile

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.special import softmax
import open3d as o3d
import trimesh

import bpy
import bmesh
from mathutils import Matrix
from .rigutils import ArmatureGenerator

import torch
from torch_geometric.nn import fps
from torch_geometric.nn import radius
from torch_geometric.nn import knn_interpolate
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

ONNX_RUNTIME = True

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
    tpl_e = add_self_loops(tpl_e, num_nodes=v.shape[0])
    tpl_e = tpl_e.astype(np.int64)

    # surface geodesic distance matrix
    print("     calculating surface geodesic matrix.")
    surface_geodesic = calc_surface_geodesic(mesh)

    # geodesic edges
    print("     gathering geodesic edges.")
    geo_e = get_geo_edges(surface_geodesic, mesh_v).T
    geo_e = add_self_loops(geo_e, num_nodes=v.shape[0])
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


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def geometric_fps(src, batch=None, ratio=None):
    src = torch.from_numpy(src)
    batch = torch.from_numpy(batch)
    res = fps(src, batch=batch, ratio=ratio)
    return res.numpy()


def geometric_radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    batch_x = torch.from_numpy(batch_x) if batch_x is not None else None
    batch_y = torch.from_numpy(batch_y) if batch_y is not None else None
    row, col = radius(x, y, r, batch_x, batch_y, max_num_neighbors=max_num_neighbors)
    return row.numpy(), col.numpy()


def geometric_knn_interpolate(x, pos_x, pos_y, batch_x=None, batch_y=None, k=3):
    x = torch.from_numpy(x)
    pos_x = torch.from_numpy(pos_x)
    pos_y = torch.from_numpy(pos_y)
    batch_x = torch.from_numpy(batch_x)
    batch_y = torch.from_numpy(batch_y)
    res = knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k)
    return res.numpy()


def predict_joints(
        data, vox, joint_net, threshold, bandwidth=0.04, mesh_filename=None):
    """
    Predict joints
    :param data: wrapped input data
    :param vox: voxelized mesh
    :param joint_net: network for predicting joints
    :param threshold: density threshold to filter out shifted points
    :param bandwidth: bandwidth for meanshift clustering
    :return: wrapped data with predicted joints, pair-wise bone representation added.
    """

    batch, pos, geo_edge_index, tpl_edge_index = \
        data['batch'], data['pos'], data['geo_edge_index'], data['tpl_edge_index']

    if not ONNX_RUNTIME:
        output = joint_net.predict({
            'batch': batch, 'pos': pos,
            'geo_edge_index': geo_edge_index, 'tpl_edge_index': tpl_edge_index
        })
    else:
        in_batch = joint_net.get_inputs()[0].name
        in_pos = joint_net.get_inputs()[1].name
        in_geo_e = joint_net.get_inputs()[2].name
        in_tpl_e = joint_net.get_inputs()[3].name
        out_displacement = joint_net.get_outputs()[0].name
        out_attn_pred0 = joint_net.get_outputs()[1].name
        out_attn_pred = joint_net.get_outputs()[2].name
        output = joint_net.run(
            [out_displacement, out_attn_pred0, out_attn_pred],
            {
                in_batch: batch, in_pos: pos,
                in_geo_e: geo_edge_index, in_tpl_e: tpl_edge_index
            })
    data_displacement, _, attn_pred = output

    y_pred = data_displacement + data['pos']
    y_pred, index_inside = inside_check(y_pred, vox)
    attn_pred = attn_pred[index_inside, :]
    y_pred = y_pred[attn_pred.squeeze() > 1e-3]
    attn_pred = attn_pred[attn_pred.squeeze() > 1e-3]

    # symmetrize points by reflecting
    y_pred_reflect = y_pred * np.array([[-1, 1, 1]])
    y_pred = np.concatenate((y_pred, y_pred_reflect), axis=0)
    attn_pred = np.tile(attn_pred, (2, 1))

    # img = draw_shifted_pts(mesh_filename, y_pred, weights=attn_pred)
    y_pred = meanshift_cluster(y_pred, bandwidth, attn_pred, max_iter=40)
    # img = draw_shifted_pts(mesh_filename, y_pred, weights=attn_pred)

    Y_dist = np.sum(((y_pred[np.newaxis, ...] - y_pred[:, np.newaxis, :]) ** 2), axis=2)
    density = np.maximum(bandwidth ** 2 - Y_dist, np.zeros(Y_dist.shape))
    density = np.sum(density, axis=0)
    density_sum = np.sum(density)
    y_pred = y_pred[density / density_sum > threshold]
    attn_pred = attn_pred[density / density_sum > threshold][:, 0]
    density = density[density / density_sum > threshold]

    # img = draw_shifted_pts(mesh_filename, y_pred, weights=attn_pred)
    pred_joints = nms_meanshift(y_pred, density, bandwidth)
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
    joints_batch = np.zeros(len(pred_joints), dtype=np.int64)
    pairs_batch = np.zeros(len(pairs), dtype=np.int64)

    data['joints'] = pred_joints.astype(np.float32)
    data['pairs'] = pairs.astype(np.float32)
    data['pair_attr'] = pair_attr.astype(np.float32)
    data['joints_batch'] = joints_batch
    data['pairs_batch'] = pairs_batch
    return data


def getInitId(data, root_net):
    """
    predict root joint ID via rootnet
    :param data:
    :param model:
    :return:
    """
    batch, pos, geo_edge_index, tpl_edge_index = \
        data['batch'], data['pos'], data['geo_edge_index'], data['tpl_edge_index']
    joints, joints_batch = data['joints'], data['joints_batch']
    idx = np.random.randn(joints.shape[0]).argsort()
    joints_shuffle = joints[idx]

    # shape_encoder
    shape_encoder = root_net['shape_encoder']
    if not ONNX_RUNTIME:
        x_glb_shape = shape_encoder.predict({
            'batch': batch, 'pos': pos,
            'geo_edge_index': geo_edge_index, 'tpl_edge_index': tpl_edge_index
        })[0]
    else:
        in_batch = shape_encoder.get_inputs()[0].name
        in_pos = shape_encoder.get_inputs()[1].name
        in_geo_e = shape_encoder.get_inputs()[2].name
        in_tpl_e = shape_encoder.get_inputs()[3].name
        out_x_glb_shape = shape_encoder.get_outputs()[0].name
        x_glb_shape = shape_encoder.run(
            [out_x_glb_shape],
            {
                in_batch: batch, in_pos: pos,
                in_geo_e: geo_edge_index, in_tpl_e: tpl_edge_index
            })[0]
    shape_feature = np.repeat(x_glb_shape, len(joints_batch[joints_batch == 0]), axis=0)

    x = np.abs(joints_shuffle[:, 0:1])
    pos = joints_shuffle
    batch = joints_batch
    sa0_joint = (x, pos, batch)

    # sa1_joint
    ratio, r = 0.999, 0.4
    idx = geometric_fps(pos, batch, ratio=ratio)
    row, col = geometric_radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=64)
    edge_index = np.stack([col, row], axis=0)
    sa1_module = root_net['sa1_module']
    if not ONNX_RUNTIME:
        x = sa1_module.predict({
            'batch': batch, 'pos': pos, 'in_pos_idx': pos[idx], 'edge_index': edge_index
        })
    else:
        in_x = sa1_module.get_inputs()[0].name
        in_pos = sa1_module.get_inputs()[1].name
        in_pos_idx = sa1_module.get_inputs()[2].name
        in_edge_index = sa1_module.get_inputs()[3].name
        out = sa1_module.get_outputs()[0].name
        x = sa1_module.run(
            [out],
            {
                in_x: x, in_pos: pos, in_pos_idx: pos[idx], in_edge_index: edge_index
            })[0]
    pos, batch = pos[idx], batch[idx]
    sa1_joint = (x, pos, batch)

    # sa2_joint
    ratio, r = 0.33, 0.6
    idx = geometric_fps(pos, batch, ratio=ratio)
    row, col = geometric_radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=64)
    edge_index = np.stack([col, row], axis=0)
    sa2_module = root_net['sa2_module']
    if not ONNX_RUNTIME:
        x = sa2_module.predict({
            'batch': batch, 'pos': pos, 'in_pos_idx': pos[idx], 'edge_index': edge_index
        })
    else:
        in_x = sa2_module.get_inputs()[0].name
        in_pos = sa2_module.get_inputs()[1].name
        in_pos_idx = sa2_module.get_inputs()[2].name
        in_edge_index = sa2_module.get_inputs()[3].name
        out = sa2_module.get_outputs()[0].name
        x = sa2_module.run(
            [out],
            {
                in_x: x, in_pos: pos, in_pos_idx: pos[idx], in_edge_index: edge_index
            })[0]
    pos, batch = pos[idx], batch[idx]
    sa2_joint = (x, pos, batch)

    # sa3_joint
    sa3_module = root_net['sa3_module']
    if not ONNX_RUNTIME:
        output = sa3_module.predict({
            'batch': batch, 'pos': pos, 'batch': batch
        })
    else:
        in_x = sa3_module.get_inputs()[0].name
        in_pos = sa3_module.get_inputs()[1].name
        in_batch = sa3_module.get_inputs()[2].name
        out_x = sa3_module.get_outputs()[0].name
        out_pos = sa3_module.get_outputs()[1].name
        out_batch = sa3_module.get_outputs()[2].name
        output = sa3_module.run(
            [out_x, out_pos, out_batch],
            {
                in_x: x, in_pos: pos, in_batch: batch
            })
    sa3_joint = output

    # fp3_joint
    x, pos, batch = sa3_joint
    x_skip, pos_skip, batch_skip = sa2_joint
    x = geometric_knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=1)
    x = np.concatenate([x, x_skip], axis=1)
    fp3_module = root_net['fp3_module']
    if not ONNX_RUNTIME:
        x = fp3_module.predict({'x': x})[0]
    else:
        in_x = fp3_module.get_inputs()[0].name
        out_x = fp3_module.get_outputs()[0].name
        x = fp3_module.run([out_x], {in_x: x})[0]
    pos, batch = pos_skip, batch_skip

    # fp2_joint
    x_skip, pos_skip, batch_skip = sa1_joint
    x = geometric_knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=3)
    x = np.concatenate([x, x_skip], axis=1)
    fp2_module = root_net['fp2_module']
    if not ONNX_RUNTIME:
        x = fp2_module.predict({'x': x})[0]
    else:
        in_x = fp2_module.get_inputs()[0].name
        out_x = fp2_module.get_outputs()[0].name
        x = fp2_module.run([out_x], {in_x: x})[0]
    pos, batch = pos_skip, batch_skip

    # fp1_joint
    x_skip, pos_skip, batch_skip = sa0_joint
    x = geometric_knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=3)
    x = np.concatenate([x, x_skip], axis=1)
    fp1_module = root_net['fp1_module']
    if not ONNX_RUNTIME:
        joint_feature = fp1_module.predict({'x': x})[0]
    else:
        in_x = fp1_module.get_inputs()[0].name
        out_x = fp1_module.get_outputs()[0].name
        joint_feature = fp1_module.run([out_x], {in_x: x})[0]

    x_joint = np.concatenate([shape_feature, joint_feature], axis=1)

    back_layers = root_net['back_layers']
    if not ONNX_RUNTIME:
        x_joint = back_layers.predict({'x_joint': x_joint})[0]
    else:
        in_x = back_layers.get_inputs()[0].name
        out_x = back_layers.get_outputs()[0].name
        x_joint = back_layers.run([out_x], {in_x: x_joint})[0]

    root_prob = x_joint
    root_prob = sigmoid(root_prob)
    root_id = np.argmax(root_prob)

    return root_id


def predict_skeleton(
        data, vox, root_net, bone_net):
    """
    Predict skeleton structure based on joints
    :param data: wrapped data
    :param vox: voxelized mesh
    :param root_net: network to predict root, pairwise connectivity cost
    :param mesh_filename: meshfilename for debugging
    :return: predicted skeleton structure
    """
    root_id = getInitId(data, root_net)

    joints, pairs, pair_attr, joints_batch, pairs_batch = \
        data['joints'], data['pairs'], data['pair_attr'], data['joints_batch'], data['pairs_batch']
    sa0_joints = (None, joints, joints_batch)
    _, pos, batch = sa0_joints

    # sa1_joint
    ratio, r = 0.999, 0.4
    idx = geometric_fps(pos, batch, ratio=ratio)
    row, col = geometric_radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=64)
    edge_index = np.stack([col, row], axis=0)
    sa1_module = bone_net['sa1_module']
    if not ONNX_RUNTIME:
        x = sa1_module.predict({
            'pos': pos, 'in_pos_idx': pos[idx], 'edge_index': edge_index
        })
    else:
        in_pos = sa1_module.get_inputs()[0].name
        in_pos_idx = sa1_module.get_inputs()[1].name
        in_edge_index = sa1_module.get_inputs()[2].name
        out = sa1_module.get_outputs()[0].name
        x = sa1_module.run(
            [out],
            {
                in_pos: pos, in_pos_idx: pos[idx], in_edge_index: edge_index
            })[0]
    pos, batch = pos[idx], batch[idx]

    # sa2_joint
    ratio, r = 0.33, 0.6
    idx = geometric_fps(pos, batch, ratio=ratio)
    row, col = geometric_radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=64)
    edge_index = np.stack([col, row], axis=0)
    sa2_module = bone_net['sa2_module']
    if not ONNX_RUNTIME:
        x = sa2_module.predict({
            'batch': batch, 'pos': pos, 'in_pos_idx': pos[idx], 'edge_index': edge_index
        })
    else:
        in_x = sa2_module.get_inputs()[0].name
        in_pos = sa2_module.get_inputs()[1].name
        in_pos_idx = sa2_module.get_inputs()[2].name
        in_edge_index = sa2_module.get_inputs()[3].name
        out = sa2_module.get_outputs()[0].name
        x = sa2_module.run(
            [out],
            {
                in_x: x, in_pos: pos, in_pos_idx: pos[idx], in_edge_index: edge_index
            })[0]
    pos, batch = pos[idx], batch[idx]

    # sa3_joint
    sa3_module = bone_net['sa3_module']
    if not ONNX_RUNTIME:
        output = sa3_module.predict({
            'batch': batch, 'pos': pos, 'batch': batch
        })
    else:
        in_x = sa3_module.get_inputs()[0].name
        in_pos = sa3_module.get_inputs()[1].name
        in_batch = sa3_module.get_inputs()[2].name
        out_x = sa3_module.get_outputs()[0].name
        out_pos = sa3_module.get_outputs()[1].name
        out_batch = sa3_module.get_outputs()[2].name
        output = sa3_module.run(
            [out_x, out_pos, out_batch],
            {
                in_x: x, in_pos: pos, in_batch: batch
            })
    joint_feature, _, _ = output
    joint_feature = np.repeat(joint_feature, len(pairs_batch[pairs_batch == 0]), axis=0)

    batch, pos, geo_edge_index, tpl_edge_index = \
        data['batch'], data['pos'], data['geo_edge_index'], data['tpl_edge_index']

    # shape_encoder
    shape_encoder = bone_net['shape_encoder']
    if not ONNX_RUNTIME:
        shape_feature = shape_encoder.predict({
            'batch': batch, 'pos': pos,
            'geo_edge_index': geo_edge_index, 'tpl_edge_index': tpl_edge_index
        })[0]
    else:
        in_batch = shape_encoder.get_inputs()[0].name
        in_pos = shape_encoder.get_inputs()[1].name
        in_geo_e = shape_encoder.get_inputs()[2].name
        in_tpl_e = shape_encoder.get_inputs()[3].name
        out_shape_feature = shape_encoder.get_outputs()[0].name
        shape_feature = shape_encoder.run(
            [out_shape_feature],
            {
                in_batch: batch, in_pos: pos,
                in_geo_e: geo_edge_index, in_tpl_e: tpl_edge_index
            })[0]
    shape_feature = np.repeat(shape_feature, len(pairs_batch[pairs_batch == 0]), axis=0)

    pairs = pairs.astype(np.int64)
    joints_pair = np.concatenate(
        (joints[pairs[:, 0]], joints[pairs[:, 1]], pair_attr[:, :-1]), axis=1)

    # expand_joint_feature
    expand_joint_feature = bone_net['expand_joint_feature']
    if not ONNX_RUNTIME:
        pair_feature = expand_joint_feature.predict({
            'joints_pair': joints_pair,
        })[0]
    else:
        in_x = expand_joint_feature.get_inputs()[0].name
        out_x = expand_joint_feature.get_outputs()[0].name
        pair_feature = expand_joint_feature.run(
            [out_x], {in_x: joints_pair})[0]

    pair_feature = np.concatenate((shape_feature, joint_feature, pair_feature), axis=1)

    # mix_transform
    mix_transform = bone_net['mix_transform']
    if not ONNX_RUNTIME:
        pre_label = mix_transform.predict({
            'pair_feature': pair_feature,
        })[0]
    else:
        in_x = mix_transform.get_inputs()[0].name
        out_x = mix_transform.get_outputs()[0].name
        pre_label = mix_transform.run(
            [out_x], {in_x: pair_feature})[0]

    connect_prob = pre_label
    connect_prob = sigmoid(connect_prob)

    pair_idx = pairs
    prob_matrix = np.zeros((len(joints), len(joints)))
    prob_matrix[pair_idx[:, 0], pair_idx[:, 1]] = connect_prob.squeeze()
    prob_matrix = prob_matrix + prob_matrix.transpose()
    cost_matrix = -np.log(prob_matrix + 1e-10)
    cost_matrix = increase_cost_for_outside_bone(cost_matrix, joints, vox)

    pred_skel = RigInfo()
    parent, key = primMST_symmetry(cost_matrix, root_id, joints)
    for i in range(len(parent)):
        if parent[i] == -1:
            pred_skel.root = TreeNode('root', tuple(joints[i]))
            break
    loadSkel_recur(pred_skel.root, i, None, joints, parent)
    pred_skel.joint_pos = pred_skel.get_joint_dict()

    return pred_skel


def predict_skinning(
        data, pred_skel, skin_net, surface_geodesic,
        subsampling=False, decimation=3000, sampling=1500):
    """
    predict skinning
    :param data: wrapped input data
    :param pred_skel: predicted skeleton
    :param skin_net: network to predict skinning weights
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: predicted rig with skinning weights information
    """
    num_nearest_bone = 5
    bones, bone_names, bone_isleaf = get_bones(pred_skel)
    mesh_v = data['pos']

    print("     calculating volumetric geodesic distance from vertices to bone. This step takes some time...")
    geo_dist = calc_geodesic_matrix(
        bones, mesh_v, surface_geodesic,
        use_sampling=subsampling, decimation=decimation, sampling=sampling,
        mesh_normalized=MESH_NORMALIZED)
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
    data['skin_input'] = skin_input

    pos, tpl_edge_index, geo_edge_index, batch = \
        data['pos'], data['tpl_edge_index'], data['geo_edge_index'], data['batch']
    if not ONNX_RUNTIME:
        skin_pred = skin_net.predict({
            'batch': batch, 'pos': pos, 'geo_edge_index': geo_edge_index, 'tpl_edge_index': tpl_edge_index,
            'sample': skin_input.astype(np.float32)
        })[0]
    else:
        in_batch = skin_net.get_inputs()[0].name
        in_pos = skin_net.get_inputs()[1].name
        in_geo_e = skin_net.get_inputs()[2].name
        in_tpl_e = skin_net.get_inputs()[3].name
        in_sample = skin_net.get_inputs()[4].name
        out = skin_net.get_outputs()[0].name
        skin_pred = skin_net.run(
            [out],
            {
                in_batch: batch, in_pos: pos, in_geo_e: geo_edge_index, in_tpl_e: tpl_edge_index,
                in_sample: skin_input.astype(np.float32)
            })[0]

    skin_pred = softmax(skin_pred, axis=1)
    skin_pred = skin_pred * loss_mask

    skin_nn = skin_nn[:, 0:num_nearest_bone]
    skin_pred_full = np.zeros((len(skin_pred), len(bone_names)))
    for v in range(len(skin_pred)):
        for nn_id in range(len(skin_nn[v, :])):
            skin_pred_full[v, skin_nn[v, nn_id]] = skin_pred[v, nn_id]

    print("     filtering skinning prediction")
    tpl_e = tpl_edge_index
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
    data = predict_joints(data, vox, net_info['jointNet'], threshold, bandwidth=bandwidth)

    print("predicting connectivity")
    pred_skeleton = predict_skeleton(data, vox, net_info['rootNet'], net_info['boneNet'])
    # pred_skeleton.normalize(scale_normalize, -translation_normalize)

    print("predicting skinning")
    pred_rig = predict_skinning(
        data, pred_skeleton, net_info['skinNet'], surface_geodesic,
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
