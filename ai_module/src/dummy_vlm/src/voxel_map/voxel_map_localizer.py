import numpy as np
import torch
import torch.nn.functional as F
import os
import pickle
import hashlib
from voxel_map.voxel import VoxelizedPointcloud
import rospy
from visualization_msgs.msg import Marker
from transformers import AutoProcessor, OwlViTModel



class VoxelMapLocalizer():
    def __init__(self, voxel_map, device=None):





        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection

            # This will download the model weights and processor files to your cache
            processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

            print("Model and processor for google/owlvit-base-patch32 have been downloaded and cached.")
        except:
            print("Model and processor for google/owlvit-base-patch32 have not loaded")   

        self.model_name = 'google/owlvit-base-patch32'
        self.clip_model = None
        self.preprocessor = None
        self.embedding_dim = 768  # will try to infer from model if available
        self._offline_semantics = False

        self._init_model()
        self.voxel_pcd = voxel_map

    def _init_model(self):
        """
        Initialize OwlViT with local-cache-first strategy and offline fallback.
        """
        # 1) Try local cache only
        try:
            self.preprocessor = AutoProcessor.from_pretrained(self.model_name, local_files_only=True)
            self.clip_model = OwlViTModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
            self._infer_dim()
            try:
                rospy.loginfo("VoxelMapLocalizer: OwlViT loaded from local cache")
            except Exception:
                pass
            self._offline_semantics = False
            return
        except Exception:
            pass

        # 2) Try online if cache not present
        try:
            self.preprocessor = AutoProcessor.from_pretrained(self.model_name)
            self.clip_model = OwlViTModel.from_pretrained(self.model_name).to(self.device)
            self._infer_dim()
            try:
                rospy.loginfo("VoxelMapLocalizer: OwlViT loaded online")
            except Exception:
                pass
            self._offline_semantics = False
        except Exception as e:
            # 3) Offline fallback (deterministic embeddings)
            self.clip_model = None
            self.preprocessor = None
            self._offline_semantics = True
            try:
                rospy.logwarn(f"VoxelMapLocalizer: OwlViT unavailable; enabling offline semantics. Reason: {e}")
            except Exception:
                pass

    def _infer_dim(self):
        try:
            if hasattr(self.clip_model, "text_model") and hasattr(self.clip_model.text_model, "config") and hasattr(self.clip_model.text_model.config, "hidden_size"):
                self.embedding_dim = int(self.clip_model.text_model.config.hidden_size)
            elif hasattr(self.clip_model, "config") and hasattr(self.clip_model.config, "text_config") and hasattr(self.clip_model.config.text_config, "hidden_size"):
                self.embedding_dim = int(self.clip_model.config.text_config.hidden_size)
        except Exception:
            pass

    def _embed_offline(self, class_names):
        """
        Produce deterministic, normalized embeddings using SHA-256 seeded RNG.
        Must match the offline logic used in buildVoxel to preserve embedding space consistency.
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
        return torch.from_numpy(np.stack(vecs, axis=0)).to(self.device)

    def load_voxel_map(self, file_path):
        with open(file_path, 'rb') as file:
            voxel_map_data = pickle.load(file)

        voxel_pcd = VoxelizedPointcloud()
        voxel_pcd.add(points=voxel_map_data._points,
                      features=voxel_map_data._features,
                      rgb=voxel_map_data._rgb,
                      weights=voxel_map_data._weights,
                      scale=voxel_map_data._scale)
        return voxel_pcd

    def calculate_clip_and_st_embeddings_for_queries(self, queries):
        # Normalize to list[str]
        if isinstance(queries, str):
            class_names = [queries]
        elif isinstance(queries, (list, tuple)):
            class_names = [str(q) for q in queries]
        else:
            class_names = [str(queries)]

        if self._offline_semantics or self.clip_model is None or self.preprocessor is None:
            return self._embed_offline(class_names)

        with torch.no_grad():
            inputs = self.preprocessor(text=class_names, return_tensors="pt")
            for k in list(inputs.keys()):
                inputs[k] = inputs[k].to(self.device)
            all_clip_tokens = self.clip_model.get_text_features(**inputs)
            all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        return all_clip_tokens
        
    def find_alignment_over_model(self, queries):
        clip_text_tokens = self.calculate_clip_and_st_embeddings_for_queries(queries)
        points, features, _, _, _ = self.voxel_pcd.get_pointcloud()
        if features is None or points is None:
            # Degenerate case; return zero alignments
            return torch.zeros((clip_text_tokens.shape[0], 0), device=self.device)
        features = F.normalize(features, p=2, dim=-1)
        features = features.to(self.device)

        point_alignments = clip_text_tokens.detach() @ features.T
        return point_alignments

    # Currently we only support compute one query each time, in the future we might want to support check many queries
    def localize_AonB(self, A="cushion", B="Couch", k_A=10, k_B=50, threshold=0.5, data_type='r3d'):
        if B is None or B == '':
            target_pos, target_scale = self.find_alignment_for_A([A], threshold=threshold)
            return target_pos, target_scale
        else:
            points, _, _, _, scale = self.voxel_pcd.get_pointcloud()
            alignments = self.find_alignment_over_model([A, B]).cpu()
            if points is None or alignments.numel() == 0:
                # Fallback: no points
                return torch.zeros((0, 3)), torch.zeros((0, 3))
            A_points_idx = alignments[0].topk(k=min(k_A, alignments.shape[1]), dim=-1).indices
            B_points_idx = alignments[1].topk(k=min(k_B, alignments.shape[1]), dim=-1).indices
            A_points = points[A_points_idx].reshape(-1, 3)
            B_points = points[B_points_idx].reshape(-1, 3)
            distances = torch.norm(A_points.unsqueeze(1) - B_points.unsqueeze(0), dim=2)
            target = A_points[torch.argmin(torch.min(distances, dim=1).values)]

            if data_type == 'r3d':
                target = target[[0, 2, 1]]
                target[1] = -target[1]

            # Best-effort scale: if we have scales for A_points_idx, take the first's scale; else ones
            if scale is not None and scale.shape[0] > 0:
                target_scale = scale[A_points_idx[0]].reshape(1, 3)
            else:
                target_scale = torch.ones_like(target).unsqueeze(0)
            return target.unsqueeze(0), target_scale

    def find_alignment_for_A(self, A, threshold=0.5):
        points, features, _, _, scale = self.voxel_pcd.get_pointcloud()
        if points is None or features is None:
            return torch.zeros((0, 3)), torch.zeros((0, 3))
        alignments = self.find_alignment_over_model(A).cpu()
        mask = alignments.squeeze() > threshold
        return points[mask].reshape(-1, 3), scale[mask].reshape(-1, 3)

def main():
    rospy.init_node('voxel_map_localizer')
    # Example/demo (not used in normal flow); requires an existing voxel_map.pkl
    try:
        vm_local = VoxelMapLocalizer(voxel_map=VoxelizedPointcloud(), device='cpu')
        A = "buddha decoration"
        B = ""
        target_point, target_scale = vm_local.localize_AonB(A, B, k_A=10, k_B=50, data_type='xyz')

        marker_pub = rospy.Publisher('/selected_object_marker', Marker, queue_size=10)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "selected_object"
        marker.id = 0
        marker.type = 1
        marker.action = 0
        if target_point.numel() >= 3:
            marker.pose.position.x = float(target_point[0, 0].item())
            marker.pose.position.y = float(target_point[0, 1].item())
            marker.pose.position.z = float(target_point[0, 2].item())
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        if target_scale.numel() >= 3:
            marker.scale.x = float(target_scale[0, 0].item())
            marker.scale.y = float(target_scale[0, 1].item())
            marker.scale.z = float(target_scale[0, 2].item())
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            marker_pub.publish(marker)
            rate.sleep()
    except Exception as e:
        rospy.logwarn(f"voxel_map_localizer demo failed: {e}")

if __name__ == "__main__":
    main()

