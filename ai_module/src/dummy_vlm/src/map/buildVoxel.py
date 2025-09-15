import numpy as np
import rospy
import math
import pickle
import hashlib
# Open3D is optional (used only for visualization); handle absence gracefully
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    o3d = None
    _HAS_O3D = False

import torch
from transformers import AutoProcessor, OwlViTModel
import torch.nn.functional as F
from voxel_map.voxel import VoxelizedPointcloud
from visualization_msgs.msg import MarkerArray
import threading

# Defer OwlViT loading to runtime and allow offline fallback
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_ID = 'google/owlvit-base-patch32'







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
        Initialize OwlViT lazily with robust offline fallback.
        Tries local cache first; if unavailable, attempts online; otherwise enables offline semantics.
        """
        # Try local cache only
        try:
            self.preprocessor = AutoProcessor.from_pretrained(_MODEL_ID, local_files_only=True)
            self.clip_model = OwlViTModel.from_pretrained(_MODEL_ID, local_files_only=True).to(_DEVICE)
            # Try to infer embedding dimension if possible
            try:
                if hasattr(self.clip_model, "text_model") and hasattr(self.clip_model.text_model, "config") and hasattr(self.clip_model.text_model.config, "hidden_size"):
                    self.embedding_dim = int(self.clip_model.text_model.config.hidden_size)
                elif hasattr(self.clip_model, "config") and hasattr(self.clip_model.config, "text_config") and hasattr(self.clip_model.config.text_config, "hidden_size"):
                    self.embedding_dim = int(self.clip_model.config.text_config.hidden_size)
            except Exception:
                pass
            try:
                rospy.loginfo("OwlViT loaded from local cache")
            except Exception:
                pass
            self._offline_semantics = False
            return
        except Exception as e_local:
            # Fall through to online attempt

            pass

        # Try online if cache not available
        try:
            self.preprocessor = AutoProcessor.from_pretrained(_MODEL_ID)
            self.clip_model = OwlViTModel.from_pretrained(_MODEL_ID).to(_DEVICE)
            # Infer embedding dimension
            try:
                if hasattr(self.clip_model, "text_model") and hasattr(self.clip_model.text_model, "config") and hasattr(self.clip_model.text_model.config, "hidden_size"):
                    self.embedding_dim = int(self.clip_model.text_model.config.hidden_size)
                elif hasattr(self.clip_model, "config") and hasattr(self.clip_model.config, "text_config") and hasattr(self.clip_model.config.text_config, "hidden_size"):
                    self.embedding_dim = int(self.clip_model.config.text_config.hidden_size)
            except Exception:
                pass
            try:
                rospy.loginfo("OwlViT loaded online")
            except Exception:
                pass
            self._offline_semantics = False
        except Exception as e_online:
            self.clip_model = None
            self.preprocessor = None
            self._offline_semantics = True
            try:
                rospy.logwarn(f"OwlViT unavailable; enabling offline semantics. Reason: {e_online}")
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
            inputs = self.preprocessor(text=class_names, return_tensors="pt")
            # Move inputs to the appropriate device
            for k in list(inputs.keys()):
                inputs[k] = inputs[k].to(_DEVICE)
            all_clip_tokens = self.clip_model.get_text_features(**inputs)
            all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        
        return all_clip_tokens.cpu().numpy()

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