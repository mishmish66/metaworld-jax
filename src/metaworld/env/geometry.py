"""
Rotation conventions of Meta-World's observations, and contact tests between geoms of a simulated world.
"""
import itertools
from collections.abc import Sequence

import numpy as np

import jax.numpy as jnp
import mujoco
from mujoco.mjx._src import collision_convex
from mujoco.mjx._src import mesh as mjx_mesh
from mujoco.mjx._src.collision_types import ConvexInfo


def mat2quat(R):
    """Quaternion (x, y, z, w) of rotation matrix `R`, as scipy's `Rotation.from_matrix(R).as_quat()`."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    choice = jnp.argmax(jnp.array([R[0, 0], R[1, 1], R[2, 2], trace]))
    candidates = jnp.stack([
        jnp.array([1. - trace + 2. * R[0, 0], R[1, 0] + R[0, 1], R[2, 0] + R[0, 2], R[2, 1] - R[1, 2]]),
        jnp.array([R[0, 1] + R[1, 0], 1. - trace + 2. * R[1, 1], R[2, 1] + R[1, 2], R[0, 2] - R[2, 0]]),
        jnp.array([R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], 1. - trace + 2. * R[2, 2], R[1, 0] - R[0, 1]]),
        jnp.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], 1. + trace]),
    ])
    q = candidates[choice]
    return q / jnp.linalg.norm(q)


# number of sides of the convex prism that stands in for a cylinder
_CYLINDER_PRISM_SIDES = 16


def _convex_arrays(vert, face):
    """MJX convex arrays (vertices, face vertices, face normals, edges, edge face normals) of a convex polytope."""
    face_normal = mjx_mesh._get_face_norm(vert, face)
    edge, edge_face_normal = mjx_mesh._get_edge_normals(face, face_normal)
    return vert, vert[face], face_normal, edge, edge_face_normal


def _unit_prism(sides):
    """Convex prism approximating the cylinder of radius 1 and half-height 1 along z, as MJX convex arrays."""
    angles = np.arange(sides) * 2. * np.pi / sides
    radius = 2. / (1. + np.cos(np.pi / sides))
    ring = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    vert = np.concatenate([np.concatenate([ring, np.full((sides, 1), z)], axis=1) for z in (-1., 1.)])
    k = np.arange(sides)
    walls = np.stack([k, (k + 1) % sides, sides + (k + 1) % sides, sides + k], axis=1)
    walls = np.pad(walls, ((0, 0), (0, sides - 4)), mode="edge")
    face = np.concatenate([k[::-1][None], (sides + k)[None], walls])
    return _convex_arrays(vert, face)


def _unit_box():
    """The box with half-extents 1, as MJX convex arrays."""
    vert = np.array(list(itertools.product((-1., 1.), (-1., 1.), (-1., 1.))))
    face = np.array([0, 4, 5, 1, 0, 2, 6, 4, 6, 7, 5, 4, 2, 3, 7, 6, 1, 5, 7, 3, 0, 1, 3, 2]).reshape(-1, 4)
    return _convex_arrays(vert, face)


_UNIT_PRISM = _unit_prism(_CYLINDER_PRISM_SIDES)
_UNIT_BOX = _unit_box()


def _scaled(unit, scale):
    """MJX convex arrays of the unit polytope `unit` scaled by `scale`."""
    vert, face, face_normal, edge, edge_face_normal = unit
    return vert.astype(np.float32) * scale, face.astype(np.float32) * scale, face_normal, edge, edge_face_normal


def _convex_shape(m, g, inflate=0.):
    """`ConvexInfo` of geom `g`, without a pose, a box's half-extents grown by `inflate`."""
    size = m.geom_size[g].astype(np.float32)
    match mujoco.mjtGeom(m.geom_type[g]):
        case mujoco.mjtGeom.mjGEOM_MESH:
            mesh = mjx_mesh.convex(m, m.geom_dataid[g])
            arrays = mesh.vert, mesh.face, mesh.face_normal, mesh.edge, mesh.edge_face_normal
        case mujoco.mjtGeom.mjGEOM_BOX:
            size = size + np.float32(inflate)
            arrays = _scaled(_UNIT_BOX, size)
        case mujoco.mjtGeom.mjGEOM_CYLINDER:
            arrays = _scaled(_UNIT_PRISM, size[[0, 0, 1]])
        case t:
            raise NotImplementedError(f"contact tests for geom type {t.name}")
    return ConvexInfo(None, None, jnp.asarray(size), *(jnp.asarray(x) for x in arrays))


def _sphere_box_distance(m, sphere, box, d):
    """Signed distance between sphere geom `sphere` and box geom `box` in world data `d`."""
    half = jnp.asarray(m.geom_size[box], jnp.float32)
    p = jnp.matmul(d.geom_xmat[box].reshape(3, 3).T, d.geom_xpos[sphere] - d.geom_xpos[box], precision="highest")
    outside = jnp.linalg.norm(p - jnp.clip(p, -half, half))
    inside = -jnp.min(half - jnp.abs(p))
    return jnp.where(outside > 0., outside, inside) - m.geom_size[sphere][0]


def _touching(m, a, b, d, margin):
    """Whether geoms `a` and `b`, one of them a box, are closer than `margin` in world data `d`."""
    box, other = (a, b) if m.geom_type[a] == mujoco.mjtGeom.mjGEOM_BOX else (b, a)
    if m.geom_type[box] != mujoco.mjtGeom.mjGEOM_BOX:
        raise NotImplementedError("contact tests between geoms that are not boxes")
    if m.geom_type[other] == mujoco.mjtGeom.mjGEOM_SPHERE:
        return _sphere_box_distance(m, other, box, d) < margin
    # the box grown by the margin penetrates the other geom
    grown = _convex_shape(m, box, margin).replace(pos=d.geom_xpos[box], mat=d.geom_xmat[box].reshape(3, 3))
    shape = _convex_shape(m, other).replace(pos=d.geom_xpos[other], mat=d.geom_xmat[other].reshape(3, 3))
    boxes = m.geom_type[other] == mujoco.mjtGeom.mjGEOM_BOX
    dist, _, _ = (collision_convex._box_box if boxes else collision_convex._convex_convex)(grown, shape)
    return jnp.min(dist) < 0.


class ContactTest:
    """
    Whether every geom of one group touches some geom of another in a simulated world. Pairs that MuJoCo
    never collides (disjoint contype / conaffinity bits, or geoms of the same body) are skipped.
    """
    def __init__(self, m, geoms_a: Sequence[int], geoms_b: Sequence[int]):
        self._m = m
        self._pairs = [
            [
                b for b in geoms_b if m.geom_bodyid[a] != m.geom_bodyid[b]
                and (m.geom_contype[a] & m.geom_conaffinity[b] or m.geom_contype[b] & m.geom_conaffinity[a])
            ]
            for a in geoms_a
        ]
        self._geoms_a = list(geoms_a)
        self._margin = np.asarray(m.geom_margin)

    def __call__(self, d):
        """Whether each geom of the first group touches a geom of the second in world data `d`."""
        touching = []
        for a, bs in zip(self._geoms_a, self._pairs):
            hits = [_touching(self._m, a, b, d, max(self._margin[a], self._margin[b])) for b in bs]
            touching.append(jnp.any(jnp.stack(hits)) if hits else jnp.asarray(False))
        return jnp.all(jnp.stack(touching))
