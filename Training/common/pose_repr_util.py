import numpy as np
from scipy.spatial.transform import Rotation as R


# def compute_relative_pose(pos, rot, base_pos, base_rot_mat,
#                           rot_transformer_to_mat,
#                           rot_transformer_to_target,
#                           backward=False,
#                           delta=False):
#     if not backward:
#         # forward pass
#         if not delta:
#             output_pos = pos if base_pos is None else pos - base_pos
#             output_rot = rot_transformer_to_target.forward(
#                 rot_transformer_to_mat.forward(rot) @ np.linalg.inv(base_rot_mat))
#             return output_pos, output_rot
#         else:
#             all_pos = np.concatenate([base_pos[None,...], pos], axis=0)
#             output_pos = np.diff(all_pos, axis=0)
            
#             rot_mat = rot_transformer_to_mat.forward(rot)
#             all_rot_mat = np.concatenate([base_rot_mat[None,...], rot_mat], axis=0)
#             prev_rot = np.linalg.inv(all_rot_mat[:-1])
#             curr_rot = all_rot_mat[1:]
#             rot = np.matmul(curr_rot, prev_rot)
#             output_rot = rot_transformer_to_target.forward(rot)
#             return output_pos, output_rot
            
#     else:
#         # backward pass
#         if not delta:
#             output_pos = pos if base_pos is None else pos + base_pos
#             output_rot = rot_transformer_to_mat.inverse(
#                 rot_transformer_to_target.inverse(rot) @ base_rot_mat)
#             return output_pos, output_rot
#         else:
#             output_pos = np.cumsum(pos, axis=0) + base_pos
            
#             rot_mat = rot_transformer_to_target.inverse(rot)
#             output_rot_mat = np.zeros_like(rot_mat)
#             curr_rot = base_rot_mat
#             for i in range(len(rot_mat)):
#                 curr_rot = rot_mat[i] @ curr_rot
#                 output_rot_mat[i] = curr_rot
#             output_rot = rot_transformer_to_mat.inverse(rot)
#             return output_pos, output_rot

def pose_to_mat(pose_6d):
    """Convert [pos(3), rot_axis_angle(3)] → 4x4 matrix."""
    pos = pose_6d[..., :3]
    rotvec = pose_6d[..., 3:6]
    rotmat = R.from_rotvec(rotvec).as_matrix()
    mat = np.zeros(pose_6d.shape[:-1] + (4, 4), dtype=pose_6d.dtype)
    mat[..., :3, :3] = rotmat
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1.0
    return mat


def mat_to_pose(mat):
    """Convert 4x4 matrix → [pos(3), rot_axis_angle(3)]."""
    pos = mat[..., :3, 3]
    rot = R.from_matrix(mat[..., :3, :3]).as_rotvec()
    pose = np.concatenate([pos, rot], axis=-1)
    return pose

def absolute_to_relative_action(abs_pose_6d, env_obs):
    """
    Convert an absolute 6D pose into a relative action (for a single robot).
    This is the inverse of relative_to_absolute_action().
    """
    # 1️⃣ Build current (base) pose matrix from env_obs
    base_pose_mat = pose_to_mat(env_obs)

    # 2️⃣ Convert target absolute pose (6D) → matrix
    abs_pose_mat = pose_to_mat(abs_pose_6d)

    # 3️⃣ Compute relative transform (base^-1 @ absolute)
    rel_pose_mat = np.linalg.inv(base_pose_mat) @ abs_pose_mat

    # 4️⃣ Convert back to [pos, rot_axis_angle]
    rel_pose = mat_to_pose(rel_pose_mat)
    return rel_pose

def relative_to_absolute_action(action_rel_6d, env_obs):
    """
    Convert a predicted relative 6D action into absolute pose (for a single robot).
    """
    # 1️⃣ Build current (base) pose matrix from env_obs
    base_pose_mat = pose_to_mat(env_obs)

    # 2️⃣ Convert predicted relative action (6D) → matrix
    action_pose_mat = pose_to_mat(action_rel_6d)

    # 3️⃣ Transform relative → absolute
    abs_pose_mat = base_pose_mat @ action_pose_mat

    # 4️⃣ Convert back to [pos, rot_axis_angle]
    abs_pose = mat_to_pose(abs_pose_mat)
    return abs_pose


def convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep='abs', backward=False):
    if not backward:
        # training transform
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'rel':
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[...,:3,3] - base_pose_mat[:3,3]
            rot = pose_mat[...,:3,:3] @ np.linalg.inv(base_pose_mat[:3,:3])
            out = np.copy(pose_mat)
            out[...,:3,:3] = rot
            out[...,:3,3] = pos
            return out
        elif pose_rep == 'relative':
            out = np.linalg.inv(base_pose_mat) @ pose_mat
            return out
        elif pose_rep == 'delta':
            all_pos = np.concatenate([base_pose_mat[None,:3,3], pose_mat[...,:3,3]], axis=0)
            out_pos = np.diff(all_pos, axis=0)
            
            all_rot_mat = np.concatenate([base_pose_mat[None,:3,:3], pose_mat[...,:3,:3]], axis=0)
            prev_rot = np.linalg.inv(all_rot_mat[:-1])
            curr_rot = all_rot_mat[1:]
            out_rot = np.matmul(curr_rot, prev_rot)
            
            out = np.copy(pose_mat)
            out[...,:3,:3] = out_rot
            out[...,:3,3] = out_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

    else:
        # eval transform
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'rel':
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[...,:3,3] + base_pose_mat[:3,3]
            rot = pose_mat[...,:3,:3] @ base_pose_mat[:3,:3]
            out = np.copy(pose_mat)
            out[...,:3,:3] = rot
            out[...,:3,3] = pos
            return out
        elif pose_rep == 'relative':
            out = base_pose_mat @ pose_mat
            return out
        elif pose_rep == 'delta':
            output_pos = np.cumsum(pose_mat[...,:3,3], axis=0) + base_pose_mat[:3,3]
            
            output_rot_mat = np.zeros_like(pose_mat[...,:3,:3])
            curr_rot = base_pose_mat[:3,:3]
            for i in range(len(pose_mat)):
                curr_rot = pose_mat[i,:3,:3] @ curr_rot
                output_rot_mat[i] = curr_rot
            
            out = np.copy(pose_mat)
            out[...,:3,:3] = output_rot_mat
            out[...,:3,3] = output_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")
            


# # --- Test ---
# np.random.seed(42)

# Random current pose
env_obs = np.array([-0.1173643,   0.50973348,  0.28119145,  3.11168808,  0.09795643, -0.02324483])
rel_action = np.array([-0.02157085,  0.00311111,  0.00464755,  1.2188743,   0.95760465,  0.1327787])

absolute = relative_to_absolute_action(rel_action, env_obs)

print(f"Absolute Action = {absolute}")

abs_action = np.array([-0.13875897,  0.50514442,  0.27697378, -1.71391118,  0.08380121, -0.96925235])

relative = absolute_to_relative_action(abs_action, env_obs)

print(f"Relative Action: {relative}")