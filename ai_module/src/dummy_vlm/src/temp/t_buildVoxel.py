import numpy as np
import rospy
import math
import pickle
import hashlib
import os
# Open3D is optional (used only for visualization); handle absence gracefully
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    o3d = None
    _HAS_O3D = False

import torch
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
from voxel_map.voxel import VoxelizedPointcloud
from visualization_msgs.msg import MarkerArray
import threading

# SigLIP2 config: choose latest available model with local-cache-first then online fallback
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_SIGLIP2_ENV = os.getenv("SIGLIP2_MODEL_ID", "").strip() or None
_SIGLIP2_CANDIDATES = ([_SIGLIP2_ENV] if _SIGLIP2_ENV else []) + [
    "google/siglip2-base-patch16-512",
    "google/siglip-so400m-patch14-384",
    "google/siglip-base-patch16-256-multilingual"
]

class SemanticVoxelMap:
    def __init__(self):
        self.voxel_map = VoxelizedPointcloud()
        self.markers = {}
        self.lock = threading.Lock()
        self.voxel_map_path = 'voxel_map.pkl'
        # Model-related members
        self.clip_model = None
        self.preprocessor = None
        self.embedding_dim = 768  # default; will try to infer from model config
        self._offline_semantics = False
        self._init_model()

    def _init_model(self):
        """
        Initialize SigLIP2 text model lazily with robust offline fallback.
        Tries local cache for preferred IDs; if unavailable, attempts online; otherwise enables offline semantics.
        """
        # 1) Try local cache only for candidate model IDs
        for mid in _SIGLIP2_CANDIDATES:
            try:
                self.preprocessor = AutoProcessor.from_pretrained(mid, local_files_only=True)
                self.clip_model = AutoModel.from_pretrained(mid, local_files_only=True).to(_DEVICE)
                # Infer embedding dimension
                try:
                    if hasattr(self.clip_model, "config") and hasattr(self.clip_model.config, "projection_dim"):
                        self.embedding_dim = int(self.clip_model.config.projection_dim)
                    elif hasattr(self.clip_model, "text_model") and hasattr(self.clip_model.text_model, "config") and hasattr(self.clip_model.text_model.config, "hidden_size"):
                        self.embedding_dim = int(self.clip_model.text_model.config.hidden_size)
                except Exception:
                    pass
                try:
                    rospy.loginfo(f"SigLIP2 text model loaded from local cache: {mid}")
                except Exception:
                    pass
                self._offline_semantics = False
                return
            except Exception:
                continue

        # 2) Try online if cache not available
        last_err = None
        for mid in _SIGLIP2_CANDIDATES:
            try:
                self.preprocessor = AutoProcessor.from_pretrained(mid)
                self.clip_model = AutoModel.from_pretrained(mid).to(_DEVICE)
                # Infer embedding dimension
                try:
                    if hasattr(self.clip_model, "config") and hasattr(self.clip_model.config, "projection_dim"):
                        self.embedding_dim = int(self.clip_model.config.projection_dim)
                    elif hasattr(self.clip_model, "text_model") and hasattr(self.clip_model.text_model, "config") and hasattr(self.clip_model.text_model.config, "hidden_size"):
                        self.embedding_dim = int(self.clip_model.text_model.config.hidden_size)
                except Exception:
                    pass
                try:
                    rospy.loginfo(f"SigLIP2 text model loaded online: {mid}")
                except Exception:
                    pass
                self._offline_semantics = False
                return
            except Exception as e_online:
                last_err = e_online
                continue

        # 3) Offline fallback (deterministic embeddings)
        self.clip_model = None
        self.preprocessor = None
        self._offline_semantics = True
        try:
            msg = str(last_err) if last_err is not None else "no candidates available"
            rospy.logwarn(f"SigLIP2 unavailable; enabling offline semantics. Reason: {msg}")
        except Exception:
            pass

    def _embed_offline(self, class_names):
        """
        Produce deterministic embeddings without external models.
        Uses a SHA-256 hash of the class name to seed a RandomState.
        """
        vecs = []
        for name in class_names:
            try:
                seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16) % (2**32)
            except Exception:
                seed = 0
            rs = np.random.RandomState(seed)
            v = rs.randn(self.embedding_dim).astype(np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vecs.append(v)
        return np.stack(vecs, axis=0) if len(vecs) > 0 else np.zeros((0, self.embedding_dim), dtype=np.float32)

    def marker_callback(self, msg):
        with self.lock:
            for marker in msg.markers:
                marker_id = marker.id
                if marker_id not in self.markers:  # Only process new markers
                    self.markers[marker_id] = {
                        'position': [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
                        'scale': [marker.scale.x, marker.scale.y, marker.scale.z],
                        'orientation': [marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w],
                        'heading': marker.pose.orientation.w,  # Assuming heading is the w component of orientation
                        'class_name': marker.ns  # Assuming 'ns' is the namespace containing class names
                    }

                    # Process and add the new marker to the voxel map
                    self.process_marker(marker_id)

    def process_marker(self, marker_id):
        marker_data = self.markers[marker_id]
        point = np.array([marker_data['position']])
        scale = np.array([marker_data['scale']])
        orientation = np.array([marker_data['orientation']])
        heading = np.array([marker_data['heading']])
        class_name = np.array([marker_data['class_name']])

        # Convert to tensors
        point_tensor = torch.from_numpy(point.astype(np.float32))
        scale_tensor = torch.from_numpy(scale.astype(np.float32))
        orientation_tensor = torch.from_numpy(orientation.astype(np.float32))
        heading_tensor = torch.from_numpy(heading.astype(np.float32))

        clip_embeddings = self.compute_clip_embeddings(class_name)
        clip_embeddings_tensor = torch.from_numpy(clip_embeddings).float()

        # Add new points to the existing voxel map
        weights = torch.ones_like(point_tensor[:, 0])  # Uniform weight of 1 for the new point
        self.voxel_map.add(points=point_tensor, features=clip_embeddings_tensor, rgb=None, weights=None, scale=scale_tensor)

        # Optionally save the updated voxel map periodically
        # self.save_voxel_map(self.voxel_map)

    def compute_clip_embeddings(self, class_names):
        if isinstance(class_names, (np.ndarray, np.generic)):
            class_names = class_names.tolist()
        elif isinstance(class_names, str):
            class_names = [class_names]
        elif not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
            raise TypeError("Input text should be a string, a list of strings or a nested list of strings")

        if self._offline_semantics or self.clip_model is None or self.preprocessor is None:
            return self._embed_offline(class_names)

        with torch.no_grad():
            inputs = self.preprocessor(text=class_names, return_tensors="pt", padding=True, truncation=True)
            # Move tensors to the appropriate device
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(_DEVICE)

            tensor_inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            all_clip_tokens = None

            # Prefer a direct text-feature method if available
            if hasattr(self.clip_model, "get_text_features"):
                try:
                    all_clip_tokens = self.clip_model.get_text_features(**tensor_inputs)
                except Exception:
                    all_clip_tokens = None

            # Fallbacks using model outputs
            if all_clip_tokens is None:
                outputs = self.clip_model(**tensor_inputs)
                if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                    all_clip_tokens = outputs.text_embeds
                elif hasattr(outputs, "last_hidden_state"):
                    tokens = outputs.last_hidden_state[:, 0, :]  # CLS pooling
                    # Apply projection if present
                    if hasattr(self.clip_model, "text_projection") and hasattr(self.clip_model.text_projection, "weight"):
                        tokens = tokens @ self.clip_model.text_projection.weight.T
                    all_clip_tokens = tokens
                else:
                    raise RuntimeError("Unable to compute text embeddings from SigLIP2 model outputs")

            all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)

        return all_clip_tokens.detach().cpu().numpy()

    def save_voxel_map(self, voxel_map):
        with open(self.voxel_map_path, 'wb') as file:
            pickle.dump(voxel_map, file)
        print(f"Voxel map saved to {self.voxel_map_path}")

    def visualize_voxel_map(self, voxel_map):
        # Visualization is optional; skip if Open3D is not available
        if not _HAS_O3D:
            try:
                import rospy
                rospy.logwarn("Open3D not available; skipping voxel map visualization")
            except Exception:
                pass
            return

        pcd = o3d.geometry.PointCloud()
        points, features, _, _, _ = voxel_map.get_pointcloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.random.rand(len(points), 3)  # Random colors
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

    def load_voxel_map(self, file_path):
        with open(file_path, 'rb') as file:
            voxel_map = pickle.load(file)
        return voxel_map