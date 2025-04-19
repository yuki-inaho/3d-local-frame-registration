"""
ロバスト部分集合マッチング：ローカル座標フレーム版 (非線形最適化導入)
k-クリーク列挙とフレームマッチングによるポーズ推定

機能:
- ローカル座標フレーム (点+姿勢) の表現と剛体変換。
- フレーム間のペアワイズ幾何学的不変量に基づく関連グラフ構築。
- Bitsetを用いたBron-Kerboschアルゴリズムによるk-クリーク列挙。
- フレームマッチング (SVD初期化 + 非線形最適化) による6DoFポーズ推定。
- k-クリーク候補の評価 (サイズ、位置/姿勢残差) と最良対応サブセット・ポーズ選択。
- 結果の3D可視化。
"""
from __future__ import annotations
import logging
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Sequence, cast

import numpy as np
from numpy.linalg import svd, norm, det
from scipy.spatial.transform import Rotation as scipy_R
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# === 1. データ構造 ===


@dataclass(frozen=True, slots=True)
class SE3:
    """SE(3) 剛体変換"""

    R: np.ndarray = field(repr=False)  # 3x3 rotation matrix
    t: np.ndarray = field(repr=False)  # 3x1 translation vector

    def __post_init__(self):
        if self.R.shape != (3, 3) or self.t.shape != (3,):
            raise ValueError("R must be 3x3, t must be 3D vector")
        if not np.allclose(self.R.T @ self.R, np.identity(3), atol=1e-6):
            logging.debug("Input R might not be orthogonal.")
        if not np.isclose(det(self.R), 1.0):
            # Correct R to be true rotation if determinant is -1 (reflection)
            if np.isclose(det(self.R), -1.0):
                logging.warning("Input R is a reflection matrix. Attempting correction.")
                U, _, Vt = svd(self.R)
                R_corrected = U @ np.diag([1, 1, -1]) @ Vt
                object.__setattr__(self, "R", R_corrected)
                if not np.isclose(det(self.R), 1.0):
                    raise ValueError("Could not correct reflection matrix to rotation.")
            else:
                logging.warning(f"Determinant of R is {det(self.R):.4f}, expected 1.0.")

    @classmethod
    def identity(cls):
        return cls(np.identity(3), np.zeros(3))

    def inverse(self) -> SE3:
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return SE3(R_inv, t_inv)

    def transform_point(self, p: np.ndarray) -> np.ndarray:
        return self.R @ p + self.t

    def transform_rotation(self, R_in: np.ndarray) -> np.ndarray:
        return self.R @ R_in

    def __matmul__(self, other: SE3) -> SE3:
        R_new = self.R @ other.R
        t_new = self.R @ other.t + self.t
        return SE3(R_new, t_new)


@dataclass(frozen=True, slots=True)
class LocalFrame:
    """ローカル座標フレーム (点 + 姿勢)"""

    p: np.ndarray = field(repr=False)  # 3D position vector
    R: np.ndarray = field(repr=False)  # 3x3 rotation matrix
    unique_id: int

    def __post_init__(self):
        if self.p.shape != (3,) or self.R.shape != (3, 3):
            raise ValueError("p must be 3D vector, R must be 3x3 matrix.")
        if not np.allclose(self.R.T @ self.R, np.identity(3), atol=1e-6):
            logging.warning(f"Frame {self.unique_id} R might not be orthogonal.")
        if not np.isclose(det(self.R), 1.0):
            logging.warning(f"Frame {self.unique_id} R determinant is {det(self.R):.4f}, expected 1.0.")

    def transform(self, pose: SE3) -> LocalFrame:
        """フレームを pose (C->W) によって変換する"""
        p_W = pose.transform_point(self.p)
        R_W = pose.transform_rotation(self.R)
        return LocalFrame(p=p_W, R=R_W, unique_id=self.unique_id)


# === 2. フレーム生成ヘルパー ===


def create_frame_from_point_normal(p: np.ndarray, n: np.ndarray, unique_id: int) -> LocalFrame:
    """点と主方向 (法線) からローカル座標フレームを生成する"""
    p_arr = np.asarray(p)
    n_arr = np.asarray(n)
    if p_arr.shape != (3,) or n_arr.shape != (3,):
        raise ValueError("p and n must be 3D vectors.")

    # 1. Z軸 (v3) を法線 n とする (正規化)
    v3 = n_arr / norm(n_arr)
    if np.isnan(v3).any():
        raise ValueError(f"Cannot normalize zero vector n for ID {unique_id}")

    # 2. X軸 (v1) を求める
    #   ワールド Z 軸 [0, 0, 1] との外積を試みる
    world_z = np.array([0.0, 0.0, 1.0])
    v1_unnorm = np.cross(v3, world_z)
    norm_v1 = norm(v1_unnorm)

    # もし v3 がワールド Z 軸と平行なら、ワールド Y 軸 [0, 1, 0] を使う
    if norm_v1 < 1e-6:
        world_y = np.array([0.0, 1.0, 0.0])
        v1_unnorm = np.cross(v3, world_y)
        norm_v1 = norm(v1_unnorm)
        if norm_v1 < 1e-6:  # これも平行なら X 軸を使う
            world_x = np.array([1.0, 0.0, 0.0])
            v1_unnorm = np.cross(v3, world_x)
            norm_v1 = norm(v1_unnorm)
            if norm_v1 < 1e-6:
                raise ValueError(f"Could not find orthogonal vector to n for ID {unique_id}")

    v1 = v1_unnorm / norm_v1

    # 3. Y軸 (v2) を外積で求める (右手系)
    v2 = np.cross(v3, v1)  # v1, v3 は単位ベクトルなので v2 も単位ベクトル

    # 回転行列 R = [v1, v2, v3] (列ベクトル)
    R = np.stack([v1, v2, v3], axis=1)

    return LocalFrame(p=p_arr, R=R, unique_id=unique_id)


# === 3. フレーム間の幾何計算 ===


def frame_distance(frame1: LocalFrame, frame2: LocalFrame) -> float:
    """フレーム原点間のユークリッド距離"""
    return norm(frame1.p - frame2.p)


def relative_rotation_angle_rad(frame1: LocalFrame, frame2: LocalFrame) -> float:
    """フレーム間の相対回転角度 (ラジアン)"""
    R_rel = frame1.R.T @ frame2.R  # frame1 から見た frame2 の回転
    try:
        # scipy.Rotation.from_matrix が det=-1 を扱えない場合があるためチェック
        if not np.isclose(det(R_rel), 1.0):
            logging.debug(f"Relative rotation determinant is {det(R_rel):.4f}")
            # 必要なら補正するか、エラーとする
            if np.isclose(det(R_rel), -1.0):
                # 例: 反射を補正 (SVDを使うなど、より頑健な方法も検討)
                U, _, Vt = svd(R_rel)
                R_rel = U @ np.diag([1, 1, -1]) @ Vt
            else:
                return np.pi  # 不正な回転として最大の角度を返すなど

        # magnitude() は [0, pi] の回転角度を返す
        angle_rad = scipy_R.from_matrix(R_rel).magnitude()
    except ValueError as e:
        logging.warning(f"Scipy Rotation error computing relative angle: {e}, R_rel=\n{R_rel}")
        # 誤差が大きい場合などに magnitude が計算できないことがある
        # 例えば、単位行列に近い場合は 0 を返す
        if np.allclose(R_rel, np.identity(3)):
            angle_rad = 0.0
        else:
            angle_rad = np.pi  # 計算失敗時は最大角度を返すなどの対処
    return angle_rad


# === 4. 関連グラフ構築 ===


def precompute_pairwise_frame_invariants(
    frames: Dict[int, LocalFrame],
) -> Tuple[Dict[int, int], np.ndarray, np.ndarray]:
    """指定されたフレーム集合について、ペアワイズ距離と相対角度を事前計算"""
    ids = list(frames.keys())
    num_frames = len(ids)
    id_to_idx = {frame_id: i for i, frame_id in enumerate(ids)}

    dist_matrix = np.full((num_frames, num_frames), np.inf)
    rel_angle_matrix = np.full((num_frames, num_frames), np.inf)  # rad

    for i1_idx, id1 in enumerate(ids):
        dist_matrix[i1_idx, i1_idx] = 0.0
        rel_angle_matrix[i1_idx, i1_idx] = 0.0
        for i2_idx in range(i1_idx + 1, num_frames):
            id2 = ids[i2_idx]
            frame1 = frames[id1]
            frame2 = frames[id2]
            dist = frame_distance(frame1, frame2)
            angle_rel = relative_rotation_angle_rad(frame1, frame2)
            dist_matrix[i1_idx, i2_idx] = dist_matrix[i2_idx, i1_idx] = dist
            rel_angle_matrix[i1_idx, i2_idx] = rel_angle_matrix[i2_idx, i1_idx] = angle_rel

    return id_to_idx, dist_matrix, rel_angle_matrix


def build_association_graph_frames(
    src_frames: Dict[int, LocalFrame],
    tmpl_frames: Dict[int, LocalFrame],
    dist_thresh: float,
    rel_angle_thresh_rad: float,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """フレーム間のペアワイズ整合性に基づいて関連グラフを構築"""
    src_ids = list(src_frames.keys())
    tmpl_ids = list(tmpl_frames.keys())
    N = len(src_ids)
    K = len(tmpl_ids)
    vertices = [(i, j) for i in src_ids for j in tmpl_ids]
    V = len(vertices)
    logging.info(f"Building graph with {V} potential nodes ({N} src x {K} tmpl).")

    adjacency_bitsets = [0] * V

    # --- ペアワイズ不変量の事前計算 ---
    logging.debug("Precomputing pairwise invariants for source frames...")
    src_id_to_idx, src_dist_mat, src_rel_angle_mat = precompute_pairwise_frame_invariants(src_frames)
    logging.debug("Precomputing pairwise invariants for template frames...")
    tmpl_id_to_idx, tmpl_dist_mat, tmpl_rel_angle_mat = precompute_pairwise_frame_invariants(tmpl_frames)
    logging.debug("Precomputation done.")

    # --- 辺の追加 ---
    num_edges = 0
    for u_idx, (i_u, j_u) in enumerate(vertices):
        for v_idx in range(u_idx + 1, V):
            i_v, j_v = vertices[v_idx]
            if i_u == i_v or j_u == j_v:
                continue  # 1対1制約

            # --- ペアワイズ整合性チェック ---
            i_u_m = src_id_to_idx[i_u]
            i_v_m = src_id_to_idx[i_v]
            j_u_m = tmpl_id_to_idx[j_u]
            j_v_m = tmpl_id_to_idx[j_v]

            # 1. 点間距離の一致チェック
            dist_src = src_dist_mat[i_u_m, i_v_m]
            dist_tmpl = tmpl_dist_mat[j_u_m, j_v_m]
            compatible_dist = False
            if not np.isinf(dist_src) and not np.isinf(dist_tmpl):
                compatible_dist = np.abs(dist_src - dist_tmpl) <= dist_thresh

            # 2. 相対回転角度の一致チェック
            rel_angle_src = src_rel_angle_mat[i_u_m, i_v_m]
            rel_angle_tmpl = tmpl_rel_angle_mat[j_u_m, j_v_m]
            compatible_angle = False
            if not np.isinf(rel_angle_src) and not np.isinf(rel_angle_tmpl):
                # 角度の差は [0, pi] の範囲で比較
                angle_diff = np.abs(rel_angle_src - rel_angle_tmpl)
                # 角度差が pi を超える場合は 2pi - diff で計算も可能だが、magnitude()が[0, pi]を返すので不要
                compatible_angle = angle_diff <= rel_angle_thresh_rad

            # デバッグ出力
            logging.debug(
                f"Pair (S{i_u}-S{i_v}, T{j_u}-T{j_v}): "
                f"dist |Δd|={abs(dist_src-dist_tmpl):.4f} <= {dist_thresh:.4f}? {compatible_dist}; "
                f"rel_angle |Δθ|={np.rad2deg(angle_diff):.2f} <= {np.rad2deg(rel_angle_thresh_rad):.2f}? {compatible_angle}"
            )

            if compatible_dist and compatible_angle:
                adjacency_bitsets[u_idx] |= 1 << v_idx
                adjacency_bitsets[v_idx] |= 1 << u_idx
                num_edges += 1

    if num_edges == 0 and V > 0:
        logging.warning("Association graph has no edges! Check thresholds or data.")
    logging.info(f"Frame Graph: {num_edges} edges added.")
    return vertices, adjacency_bitsets


# === 5. k-クリーク列挙 ===
def bitcount(mask: int) -> int:
    return bin(mask).count("1")


def lsb_index(mask: int) -> int:
    return (mask & -mask).bit_length() - 1 if mask else -1


def select_pivot_tomita(P: int, X: int, adj: List[int]) -> int:
    pivot_candidates = P | X
    if pivot_candidates == 0:
        return -1
    best_pivot, max_neighbors_in_P = -1, -1
    active_candidates = pivot_candidates
    while active_candidates != 0:
        u_idx = lsb_index(active_candidates)
        active_candidates &= ~(1 << u_idx)
        neighbors_in_P_count = bitcount(adj[u_idx] & P)
        if neighbors_in_P_count > max_neighbors_in_P:
            max_neighbors_in_P = neighbors_in_P_count
            best_pivot = u_idx
    return best_pivot if best_pivot != -1 else lsb_index(P | X)


def bron_kerbosch_k_pivot(
    adj: List[int], k: int, *, R: int = 0, P: int, X: int = 0, found_k_cliques: List[int], max_size_found: List[int]
) -> None:
    if P == 0 and X == 0:
        current_size = bitcount(R)
        if current_size >= k:
            found_k_cliques.append(R)
            if current_size > max_size_found[0]:
                max_size_found[0] = current_size
        return
    potential_max_size = bitcount(R) + bitcount(P)
    if potential_max_size < k or potential_max_size <= max_size_found[0]:
        return

    pivot_u_idx = select_pivot_tomita(P, X, adj)
    if pivot_u_idx == -1:
        return

    P_without_neighbors_of_pivot = P & ~adj[pivot_u_idx]
    while P_without_neighbors_of_pivot != 0:
        v_idx = lsb_index(P_without_neighbors_of_pivot)
        v_mask = 1 << v_idx
        bron_kerbosch_k_pivot(
            adj=adj,
            k=k,
            R=(R | v_mask),
            P=(P & adj[v_idx]),
            X=(X & adj[v_idx]),
            found_k_cliques=found_k_cliques,
            max_size_found=max_size_found,
        )
        P &= ~v_mask
        X |= v_mask
        P_without_neighbors_of_pivot &= ~v_mask


# === 6. ポーズ推定 (フレームマッチング) ===
def estimate_pose_frames_svd(
    correspondences: Sequence[Tuple[int, int]],
    src_frames: Dict[int, LocalFrame],
    tmpl_frames: Dict[int, LocalFrame],
    estimate_rotation_from_frames: bool = True,  # 姿勢情報から回転を推定するか
) -> Optional[SE3]:
    """
    フレーム対応集合からSVDベースで剛体変換(SE3)を推定。
    クリークサイズ 1 以上で計算可能。
    """
    num_corr = len(correspondences)
    if num_corr < 1:
        logging.warning("EstimatePoseFrames: Need >= 1 correspondence.")
        return None

    # --- 回転 R の推定 ---
    R_opt = np.identity(3)
    if estimate_rotation_from_frames and num_corr >= 1:
        H_R = np.zeros((3, 3))
        for i_src, j_tmpl in correspondences:
            try:
                R_C = src_frames[i_src].R
                R_W = tmpl_frames[j_tmpl].R
                H_R += R_C @ R_W.T  # Note: H = sum(R_C * R_W^T)
            except KeyError:
                logging.error(f"Frame data missing for correspondence S{i_src}-T{j_tmpl} during rotation estimation.")
                return None
        try:
            U, _, Vt = svd(H_R)
            V = Vt.T
            # Ensure proper rotation (det=1)
            S_reflect = np.diag([1, 1, det(V @ U.T)])
            R_opt = V @ S_reflect @ U.T
            if not np.allclose(det(R_opt), 1.0):
                logging.warning(f"Frame Rotation Estimation: R det={det(R_opt):.4f}")
                # Attempt to correct if reflection (should be handled by S_reflect)
                if np.isclose(det(R_opt), -1.0):
                    R_opt = V @ np.diag([1, 1, -1]) @ U.T  # Force correction
        except np.linalg.LinAlgError as e:
            logging.error(f"Frame Rotation SVD failed: {e}")
            # Fallback: If SVD fails (e.g., only 1 correspondence?), use point-based rotation if possible
            if num_corr < 3:
                logging.warning("Cannot estimate rotation reliably from < 3 correspondences or failed SVD.")
                # For size 1, rotation is R_W @ R_C.T
                if num_corr == 1:
                    i_src, j_tmpl = correspondences[0]
                    R_opt = tmpl_frames[j_tmpl].R @ src_frames[i_src].R.T
                else:  # Size 2 case is ambiguous
                    return None  # Or try alternative
            else:  # SVD failed for >= 3, try point-based
                pass  # Proceed to point-based SVD below
        except ValueError as e:  # Handle potential scipy Rotation errors if used
            logging.error(f"Rotation matrix issue during frame pose estimation: {e}")
            return None

    # --- 点ベース SVD (オプション or フォールバック) ---
    # estimate_rotation_from_frames=False の場合、または回転推定失敗時に num_corr >= 3 ならこちらを使う
    if (not estimate_rotation_from_frames or R_opt is None) and num_corr >= 3:
        logging.debug("Estimating rotation using point-based SVD.")
        P_src = np.array([src_frames[i].p for i, j in correspondences])
        P_tmpl = np.array([tmpl_frames[j].p for i, j in correspondences])
        center_src = np.mean(P_src, axis=0)
        center_tmpl = np.mean(P_tmpl, axis=0)
        P_src_centered = P_src - center_src
        P_tmpl_centered = P_tmpl - center_tmpl
        H_p = P_src_centered.T @ P_tmpl_centered  # 3xN @ Nx3 -> 3x3

        try:
            Up, _, Vpt = svd(H_p)
            Vp = Vpt.T
            S_reflect_p = np.diag([1, 1, det(Vp @ Up.T)])
            R_opt = Vp @ S_reflect_p @ Up.T
            if not np.allclose(det(R_opt), 1.0):
                logging.warning(f"Point SVD Rotation: R det={det(R_opt):.4f}")
                if np.isclose(det(R_opt), -1.0):
                    R_opt = Vp @ np.diag([1, 1, -1]) @ Up.T  # Force correction

        except np.linalg.LinAlgError as e:
            logging.error(f"Point SVD failed: {e}")
            return None

    # 回転推定が結局できなかった場合
    if R_opt is None:
        logging.error("Rotation estimation failed.")
        return None

    # --- 並進 t の推定 ---
    # (回転 R_opt が確定した前提で、点の重心を合わせる)
    P_src_list = []
    P_tmpl_list = []
    for i_src, j_tmpl in correspondences:
        try:
            P_src_list.append(src_frames[i_src].p)
            P_tmpl_list.append(tmpl_frames[j_tmpl].p)
        except KeyError:
            logging.error(f"Frame data missing for correspondence S{i_src}-T{j_tmpl} during translation estimation.")
            return None

    P_src_arr = np.array(P_src_list)
    P_tmpl_arr = np.array(P_tmpl_list)
    center_src = np.mean(P_src_arr, axis=0)
    center_tmpl = np.mean(P_tmpl_arr, axis=0)

    t_opt = center_tmpl - R_opt @ center_src

    return SE3(R_opt, t_opt)


# === 7. 評価と選択 ===
def calculate_frame_residuals(
    correspondences: Sequence[Tuple[int, int]],
    src_frames: Dict[int, LocalFrame],
    tmpl_frames: Dict[int, LocalFrame],
    pose_estimate: SE3,
    w_p: float = 1.0,  # 位置残差の重み
    w_theta: float = 1.0,  # 姿勢 (角度) 残差の重み (例: (rad)^2 あたり)
) -> Tuple[float, int]:
    """推定ポーズにおけるフレーム間の平均二乗残差を計算"""
    total_residual_sq = 0.0
    num_valid_residuals = 0

    for i_src, j_tmpl in correspondences:
        try:
            frame_C = src_frames[i_src]
            frame_W_tmpl = tmpl_frames[j_tmpl]

            # ソースフレームを推定ポーズで変換
            frame_W_est = frame_C.transform(pose_estimate)

            # 1. 位置残差
            pos_err = norm(frame_W_est.p - frame_W_tmpl.p)

            # 2. 姿勢残差 (回転角度)
            R_err_rel = frame_W_tmpl.R.T @ frame_W_est.R  # テンプレートから見た推定姿勢のずれ
            angle_err_rad = relative_rotation_angle_rad(frame_W_tmpl, frame_W_est)

            if np.isnan(pos_err) or np.isnan(angle_err_rad):
                logging.debug(f"Skipping residual for S{i_src}-T{j_tmpl} due to NaN values.")
                continue

            residual_sq = w_p * (pos_err**2) + w_theta * (angle_err_rad**2)
            total_residual_sq += residual_sq
            num_valid_residuals += 1

        except KeyError:
            logging.error(f"KeyError during residual calculation for S{i_src} or T{j_tmpl}")
            continue
        except Exception as e:
            logging.error(f"Residual calc error for S{i_src}-T{j_tmpl}: {e}")
            continue

    if num_valid_residuals == 0:
        return np.inf, 0

    avg_residual_sq = total_residual_sq / num_valid_residuals
    return avg_residual_sq, num_valid_residuals


def evaluate_and_select_clique_frames(
    cliques_bitsets: List[int],
    vertices: Dict[int, Tuple[int, int]],
    src_frames: Dict[int, LocalFrame],
    tmpl_frames: Dict[int, LocalFrame],
    k_min_eval: int = 1,  # 評価する最小クリークサイズ (ポーズ推定は1から可能)
    w_p: float = 1.0,
    w_theta: float = 1.0,
) -> Optional[Tuple[List[Tuple[int, int]], SE3]]:
    """最良クリークを選択 (サイズ優先、次にフレーム残差優先)"""
    if not cliques_bitsets:
        logging.warning("No cliques passed for evaluation.")
        return None

    evaluated_results = []
    logging.info(f"Evaluating {len(cliques_bitsets)} clique candidates (size >= {k_min_eval})...")

    for clique_bs in cliques_bitsets:
        clique_size = bitcount(clique_bs)
        if clique_size < k_min_eval:
            continue

        # 対応リストを取得
        correspondences = [vertices[idx] for idx in vertices if (clique_bs >> idx) & 1]
        if not correspondences:
            continue  # Should not happen if clique_size >= 1

        # フレームマッチングでポーズ推定
        # サイズ1,2でも推定を試みる (信頼性は低いかもしれない)
        pose_estimate = estimate_pose_frames_svd(correspondences, src_frames, tmpl_frames)

        if pose_estimate is None:
            logging.debug(f"Pose estimation failed for clique of size {clique_size}.")
            continue

        # 残差計算
        avg_residual_sq, num_valid = calculate_frame_residuals(
            correspondences, src_frames, tmpl_frames, pose_estimate, w_p, w_theta
        )

        if num_valid > 0:
            evaluated_results.append(
                {
                    "size": clique_size,
                    "avg_residual_sq": avg_residual_sq,
                    "pose": pose_estimate,
                    "correspondences": correspondences,
                }
            )
        else:
            logging.debug(f"No valid residuals calculated for clique of size {clique_size}.")

    if not evaluated_results:
        logging.warning("No valid evaluated cliques found.")
        return None

    # サイズ優先、次に残差でソート
    evaluated_results.sort(key=lambda x: (-x["size"], x["avg_residual_sq"]))

    best = evaluated_results[0]
    logging.info(f"Selected best clique: Size={best['size']}, AvgRes^2(pos+rot)={best['avg_residual_sq']:.4f}")
    return best["correspondences"], best["pose"]


# === 8. 可視化 ===
def plot_frame(ax, p, R, color_x="r", color_y="g", color_z="b", alpha=1.0, length=0.1, label_prefix="", unique_id=""):
    """ローカル座標フレームを描画するヘルパー関数"""
    # 最初のフレームのみラベル表示 (簡略化のため)
    show_label = hasattr(ax, "_plot_frame_labels_shown") and unique_id in ax._plot_frame_labels_shown.get(
        label_prefix, set()
    )
    if not show_label:
        lbl_x = f"{label_prefix} X"
        lbl_y = f"{label_prefix} Y"
        lbl_z = f"{label_prefix} Z"
        if not hasattr(ax, "_plot_frame_labels_shown"):
            ax._plot_frame_labels_shown = {}
        if label_prefix not in ax._plot_frame_labels_shown:
            ax._plot_frame_labels_shown[label_prefix] = set()
        ax._plot_frame_labels_shown[label_prefix].add(unique_id)
    else:
        lbl_x, lbl_y, lbl_z = None, None, None

    ax.quiver(p[0], p[1], p[2], R[0, 0], R[1, 0], R[2, 0], length=length, color=color_x, alpha=alpha, label=lbl_x)
    ax.quiver(p[0], p[1], p[2], R[0, 1], R[1, 1], R[2, 1], length=length, color=color_y, alpha=alpha, label=lbl_y)
    ax.quiver(p[0], p[1], p[2], R[0, 2], R[1, 2], R[2, 2], length=length, color=color_z, alpha=alpha, label=lbl_z)
    ax.text(p[0], p[1], p[2], f" {label_prefix}{unique_id}", color=color_x, alpha=alpha, fontsize=7)


def plot_results_frames(
    src_frames: Dict[int, LocalFrame],
    tmpl_frames: Dict[int, LocalFrame],
    final_pose: Optional[SE3],
    correspondences: Optional[List[Tuple[int, int]]],
    true_pose: Optional[SE3] = None,  # オプション: 真のポーズ (C->W)
):
    """フレームベースの結果を3Dで可視化する"""
    try:
        # TkAgg バックエンドを試みる (環境依存)
        try:
            matplotlib.use("TkAgg")
        except ImportError:
            logging.warning("TkAgg backend not found, using default.")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax = cast(Axes3D, ax)
        arrow_length = 0.1
        ax._plot_frame_labels_shown = {}  # ラベル表示管理用

        # --- 描画 ---
        # Template frames (Light RGB, alpha=0.6)
        for j, frame_W in tmpl_frames.items():
            plot_frame(
                ax,
                frame_W.p,
                frame_W.R,
                color_x="salmon",
                color_y="lightgreen",
                color_z="lightblue",
                alpha=0.6,
                length=arrow_length,
                label_prefix="T",
                unique_id=j,
            )

        title = "Result: Frames"
        # Source frames
        if final_pose:
            title = "Registered Source Frames (RGB=Inlier/Gray=Outlier) vs Template (Light RGB)"
            inlier_src_indices = {i for i, j in correspondences} if correspondences else set()

            for i, frame_C in src_frames.items():
                # 推定ポーズでワールド座標系へ変換
                frame_W_est = frame_C.transform(final_pose)
                if i in inlier_src_indices:
                    # Inlier (Dark RGB, alpha=1.0)
                    plot_frame(
                        ax,
                        frame_W_est.p,
                        frame_W_est.R,
                        color_x="red",
                        color_y="green",
                        color_z="blue",
                        alpha=1.0,
                        length=arrow_length,
                        label_prefix="S",
                        unique_id=i,
                    )
                else:
                    # Outlier (Gray, alpha=0.8)
                    plot_frame(
                        ax,
                        frame_W_est.p,
                        frame_W_est.R,
                        color_x="dimgray",
                        color_y="dimgray",
                        color_z="dimgray",
                        alpha=0.7,
                        length=arrow_length,
                        label_prefix="S",
                        unique_id=i,
                    )
        else:
            title = "Pose Estimation Failed - Showing Original Source Frames (Magenta) vs Template (Light RGB)"
            # ポーズ推定失敗時は元のカメラ座標系のままマゼンタで表示
            for i, frame_C in src_frames.items():
                plot_frame(
                    ax,
                    frame_C.p,
                    frame_C.R,
                    color_x="magenta",
                    color_y="magenta",
                    color_z="magenta",
                    alpha=0.7,
                    length=arrow_length,
                    label_prefix="S(orig)",
                    unique_id=i,
                )

        # --- 軸・凡例など ---
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # ax.legend()
        ax.set_title(title)

        # 軸スケール調整 (点の位置に基づいて)
        try:
            all_pts = [f.p for f in tmpl_frames.values()]
            if final_pose:
                all_pts.extend([f.transform(final_pose).p for f in src_frames.values()])
            else:
                all_pts.extend([f.p for f in src_frames.values()])

            valid_pts = [pt for pt in all_pts if pt is not None and np.all(np.isfinite(pt))]
            if not valid_pts:
                raise ValueError("No valid points for axis scaling.")

            valid_pts_arr = np.array(valid_pts)
            center = np.mean(valid_pts_arr, axis=0)
            ranges = np.ptp(valid_pts_arr, axis=0) if valid_pts_arr.shape[0] > 1 else np.ones(3) * arrow_length * 5
            max_range = np.max(ranges) if ranges.size > 0 and np.max(ranges) > 1e-6 else arrow_length * 10
            half_range = max(max_range * 0.6, arrow_length * 3)  # 矢印が見える程度のマージン

            ax.set_xlim(center[0] - half_range, center[0] + half_range)
            ax.set_ylim(center[1] - half_range, center[1] + half_range)
            ax.set_zlim(center[2] - half_range, center[2] + half_range)
            ax.set_aspect("auto")  # 'equal' は mplot3d では問題が多い

        except Exception as e:
            logging.warning(f"Axes scaling failed: {e}. Using 'auto' scaling.")
            ax.axis("auto")

        plt.tight_layout()
        plt.show()

    except ImportError:
        logging.warning("Matplotlib or TkAgg backend not available. Skipping plot.")
    except Exception as e:
        logging.error(f"An error occurred during plotting: {e}", exc_info=True)


# === 9. メイン実行部 ===
def run_pipeline(args):
    """データ生成から評価・可視化までのメインパイプライン"""
    # --- データ生成 ---
    np.random.seed(args.seed)
    pos_noise_std = args.pos_noise
    rot_noise_deg = args.rot_noise
    num_random_outliers = args.num_outliers
    allow_missing_inlier = args.allow_missing
    k_min_clique = args.k_min  # クリーク探索の最小サイズ

    if k_min_clique < 1:
        logging.warning("k_min must be >= 1. Setting k_min to 1.")
        k_min_clique = 1

    # -- テンプレートフレーム生成 --
    tmpl_side = 0.5
    template_points_dict = {
        0: {"p": [tmpl_side, 0, 0], "n": [0, 1, 0]},
        1: {"p": [-tmpl_side, 0, 0], "n": [0, 1, 0]},
        2: {"p": [0, tmpl_side, 0], "n": [0, 1, 0]},
        3: {"p": [0, -tmpl_side, 0], "n": [0, 1, 0]},
    }
    tmpl_frames: Dict[int, LocalFrame] = {}
    for j, tp in template_points_dict.items():
        try:
            tmpl_frames[j] = create_frame_from_point_normal(tp["p"], tp["n"], unique_id=j)
        except ValueError as e:
            logging.error(f"Failed to create template frame T{j}: {e}")
            return
    K = len(tmpl_frames)

    # -- 真のポーズ生成 --
    R_true_mat = scipy_R.random(random_state=args.seed).as_matrix()  # シード固定
    t_true_vec = np.random.uniform(-1.0, 1.0, 3)
    T_true_W_C = SE3(R_true_mat, t_true_vec)  # World -> Cam
    T_true_C_W = T_true_W_C.inverse()  # Cam -> World (推定対象)
    logging.info(
        "True Pose (C->W):\nR=%s\nt=%s",
        np.array_str(T_true_C_W.R, precision=3),
        np.array_str(T_true_C_W.t, precision=3),
    )

    # -- ソースフレーム生成 --
    src_frames: Dict[int, LocalFrame] = {}
    true_correspondences: Dict[int, int] = {}  # key: src_id, value: tmpl_id
    available_tmpl_indices = list(tmpl_frames.keys())
    src_idx_counter = 0

    # 欠損シミュレーション
    if allow_missing_inlier and K > 0:
        missing_idx = np.random.choice(available_tmpl_indices)
        available_tmpl_indices.remove(missing_idx)
        logging.info(f"Simulating missing template frame T{missing_idx}")

    # インライア生成
    for j_tmpl in available_tmpl_indices:
        frame_j_W = tmpl_frames[j_tmpl]
        # 理想的なソースフレーム (カメラ座標系 C)
        frame_i_C_ideal = frame_j_W.transform(T_true_W_C)  # W -> C は T_true_W_C

        # ノイズ付加
        p_i_C_noisy = frame_i_C_ideal.p + np.random.normal(0, pos_noise_std, 3)

        # 回転ノイズ (軸角度で生成し、元の姿勢に左から掛ける)
        axis = np.random.randn(3)
        axis = axis / norm(axis) if norm(axis) > 1e-9 else np.array([1.0, 0.0, 0.0])
        angle = np.random.normal(0, np.deg2rad(rot_noise_deg))
        delta_R_mat = scipy_R.from_rotvec(angle * axis).as_matrix()
        R_i_C_noisy = delta_R_mat @ frame_i_C_ideal.R  # ノイズ回転を左から適用

        try:
            # noisy な p と R から直接フレームを作る
            src_frames[src_idx_counter] = LocalFrame(p=p_i_C_noisy, R=R_i_C_noisy, unique_id=src_idx_counter)
            true_correspondences[src_idx_counter] = j_tmpl
            src_idx_counter += 1
        except ValueError as e:
            logging.warning(f"Could not create noisy source frame for template T{j_tmpl}: {e}")

    # アウトライア生成 (ランダムな位置と姿勢)
    outlier_pos_center = np.zeros(3)
    outlier_pos_range = tmpl_side * 2.0
    for _ in range(num_random_outliers):
        p_rand_C = outlier_pos_center + np.random.uniform(-outlier_pos_range, outlier_pos_range, 3)
        # ランダムな回転行列を生成
        R_rand_C = scipy_R.random().as_matrix()
        try:
            src_frames[src_idx_counter] = LocalFrame(p=p_rand_C, R=R_rand_C, unique_id=src_idx_counter)
            src_idx_counter += 1
        except ValueError as e:
            logging.warning(f"Could not create outlier frame: {e}")

    N_actual = len(src_frames)
    num_true_inliers = len(true_correspondences)
    logging.info(f"Generated {K} template frames and {N_actual} source frames ({num_true_inliers} true inliers).")
    if N_actual == 0 or K == 0:
        logging.error("Need at least one source and one template frame.")
        return

    # --- 閾値設定 ---
    dist_thresh = args.dist_thresh
    rel_angle_thresh_rad = np.deg2rad(args.rel_angle_thresh)
    logging.info(f"Using Thresholds: dist <= {dist_thresh:.3f} m, rel_angle <= {args.rel_angle_thresh:.1f} deg")
    logging.info(f"Minimum clique size k = {k_min_clique}")

    # --- パイプライン実行 ---
    try:
        # 1. 関連グラフ構築
        start_graph = time.time()
        vertices_list, adjacency_bitsets = build_association_graph_frames(
            src_frames, tmpl_frames, dist_thresh, rel_angle_thresh_rad
        )
        vertices_dict = {idx: pair for idx, pair in enumerate(vertices_list)}
        end_graph = time.time()
        logging.info(f"Graph build time: {end_graph - start_graph:.3f} s")

        V = len(vertices_dict)
        num_edges = sum(bitcount(adj) for adj in adjacency_bitsets) // 2
        if V == 0 or num_edges == 0:
            logging.error("Association graph has no vertices or no edges. Stopping.")
            if not args.no_plot:
                plot_results_frames(src_frames, tmpl_frames, None, None)
            return

        # 2. k-クリーク列挙
        start_clique = time.time()
        initial_P = (1 << V) - 1
        found_cliques_list = []
        max_size_tracker = [0]
        if V > 64:
            logging.warning(f"Vertices ({V}) > 64, bitset ops maybe slow/unreliable.")

        if k_min_clique > V:
            logging.warning(f"k_min ({k_min_clique}) > num_vertices ({V}). No cliques will be found.")
        elif V > 0:
            bron_kerbosch_k_pivot(
                adj=adjacency_bitsets,
                k=k_min_clique,
                P=initial_P,
                found_k_cliques=found_cliques_list,
                max_size_found=max_size_tracker,
            )
        end_clique = time.time()
        logging.info(f"Clique finding time: {end_clique - start_clique:.4f} s.")
        logging.info(
            f"Found {len(found_cliques_list)} potential cliques (size >= {k_min_clique}). Max size: {max_size_tracker[0]}"
        )

        # 3. 極大クリークフィルタリング
        maximal_cliques_bitsets = []
        found_cliques_list.sort(key=bitcount, reverse=True)
        processed_cliques = set()
        for i, c1 in enumerate(found_cliques_list):
            if c1 in processed_cliques:
                continue
            is_maximal = True
            for j in range(i):
                c2 = found_cliques_list[j]
                if (c1 & c2) == c1:
                    is_maximal = False
                    processed_cliques.add(c1)
                    break
            if is_maximal:
                maximal_cliques_bitsets.append(c1)
        logging.info(f"Found {len(maximal_cliques_bitsets)} maximal cliques (size >= {k_min_clique}).")

        # 4. 評価と最良クリーク選択
        start_eval = time.time()
        result = evaluate_and_select_clique_frames(
            maximal_cliques_bitsets,
            vertices_dict,
            src_frames,
            tmpl_frames,
            k_min_eval=1,  # サイズ1から評価
            w_p=args.w_p,
            w_theta=args.w_theta,
        )
        end_eval = time.time()
        logging.info(f"Clique evaluation time: {end_eval - start_eval:.4f} s.")

        # 5. 結果表示と比較
        if result:
            best_correspondences, final_pose = result
            logging.info(f"Final correspondences: {len(best_correspondences)}")
            logging.info(
                f"Final Pose (C->W):\nR=\n{np.array_str(final_pose.R, precision=3)}\nt={np.array_str(final_pose.t, precision=3)}"
            )

            # 真のポーズとの比較
            R_err_mat = final_pose.R.T @ T_true_C_W.R
            try:
                R_err_deg = np.rad2deg(scipy_R.from_matrix(R_err_mat).magnitude())
            except ValueError as e:
                logging.warning(f"Could not compute rotation error magnitude: {e}")
                R_err_deg = np.nan
            t_err_m = norm(final_pose.t - T_true_C_W.t)
            logging.info("--- Performance ---")
            logging.info(f"Rotation Err: {R_err_deg:.3f} deg")
            logging.info(f"Translation Err: {t_err_m:.4f} m")

            # 対応の正しさ評価
            correct_matches = 0
            inlier_count = len(best_correspondences)
            logging.info("Correspondences (SrcID -> TmplID):")
            for i_src, j_tmpl in sorted(best_correspondences):
                is_correct = i_src in true_correspondences and true_correspondences[i_src] == j_tmpl
                correct_matches += is_correct
                true_tmpl_id_str = (
                    f"T{true_correspondences.get(i_src, 'N/A')}" if i_src in true_correspondences else "(Outlier)"
                )
                logging.info(
                    f"  S{i_src} -> T{j_tmpl} {'(Correct)' if is_correct else '(Incorrect, True: '+true_tmpl_id_str+')'}"
                )

            precision = correct_matches / inlier_count if inlier_count > 0 else 0
            recall = correct_matches / num_true_inliers if num_true_inliers > 0 else 0
            logging.info(f"Precision: {precision:.2f} ({correct_matches}/{inlier_count})")
            logging.info(f"Recall: {recall:.2f} ({correct_matches}/{num_true_inliers})")

            # 可視化
            if not args.no_plot:
                plot_results_frames(src_frames, tmpl_frames, final_pose, best_correspondences, true_pose=T_true_C_W)

        else:
            logging.error("Pipeline failed: No best clique selected.")
            if not args.no_plot:
                plot_results_frames(src_frames, tmpl_frames, None, None)

    except Exception as e:
        logging.error(f"An error occurred in the main pipeline: {e}", exc_info=True)
        if not args.no_plot:
            try:
                plot_results_frames(src_frames, tmpl_frames, None, None)
            except Exception as plot_e:
                logging.error(f"Failed to plot after error: {plot_e}")


def main():
    parser = argparse.ArgumentParser(description="Robust subset matching of 3D frames using k-clique enumeration.")
    parser.add_argument("--no-plot", action="store_true", help="Disable visualization plot.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--pos-noise", type=float, default=0.03, help="Position noise std dev.")
    parser.add_argument("--rot-noise", type=float, default=15.0, help="Rotation noise std dev (degrees).")
    parser.add_argument("--num-outliers", type=int, default=100, help="Number of outlier frames.")
    parser.add_argument("--allow-missing", action="store_true", help="Simulate one missing inlier.")
    parser.add_argument("--k-min", type=int, default=3, help="Minimum clique size for search (>=1).")
    parser.add_argument("--dist-thresh", type=float, default=0.8, help="Pairwise distance threshold (meters).")
    parser.add_argument(
        "--rel-angle-thresh", type=float, default=10.0, help="Pairwise relative angle threshold (degrees)."
    )
    parser.add_argument("--w_p", type=float, default=1.0, help="Weight for position residual squared.")
    parser.add_argument("--w_theta", type=float, default=1.0, help="Weight for angular residual squared (rad^2).")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
