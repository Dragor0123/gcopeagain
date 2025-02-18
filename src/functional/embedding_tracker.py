import os
import datetime
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from torch_geometric.data import Data


class EmbeddingTracker:
    def __init__(self, tracked_nodes: dict, track_interval: int = 10,
                 dataset_name: str = 'unnamed', base_results_dir: str = None):
        self.tracked_nodes = tracked_nodes
        self.track_interval = track_interval
        self.tracking_data = {}
        self.tracked_indices = None  # (batch_idx, ptr_idx) 형태로 저장
        self.centroids = None
        # 결과 저장 디렉토리 생성
        current_dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_results_dir is None:
            base_results_dir = os.path.join('results', f'{dataset_name}', current_dt)
        self.tracking_log_dir = os.path.join(base_results_dir, 'tracking_log')
        self.visualization_dir = os.path.join(base_results_dir, 'visualization')
        os.makedirs(self.tracking_log_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)

    def set_centroids(self, centroids):
        self.centroids = centroids

    def setup_tracking(self, first_batch):
        """첫 epoch의 첫 배치에서 추적할 노드 인덱스 설정"""
        if self.tracked_indices is not None:
            return

        # batch의 ptr 정보를 사용하여 노드 위치 파악
        start_ptr = 0 if first_batch.ptr is None else first_batch.ptr[0].item()
        end_ptr = len(first_batch.x) if first_batch.ptr is None else first_batch.ptr[1].item()

        # 추적할 노드 선택 (예: 첫 배치에서 2개 노드)
        self.tracked_indices = [(0, start_ptr), (0, start_ptr + 1)]  # (batch_idx, ptr_idx)
        print(f"Set up tracking for nodes at positions: {self.tracked_indices}")

    def get_nodes_majority_labels(self, ptr_idx:int, all_majority_labels: dict) -> int:
        return all_majority_labels.get(ptr_idx, None)

    def track_embeddings(self, epoch: int, batch, original_features: torch.Tensor,
                         prompted_features: torch.Tensor, node_wise_prompt: torch.Tensor):
        """
        각 tracked node에 대해:
          - 원본, prompted, node-wise prompt feature를 저장
          - 원본과 prompted feature 사이의 L2 distance 계산
          - 해당 노드의 이웃을 edge_index에서 검색하여 neighbor_labels 추출 (자기 자신 제외)
          - 기본적으로 self-loop 제외한 이웃이 없으면, 자기 자신 label을 fallback으로 사용
          - DefaultNodeClassifier 로직에 따라 case (예: case1, case2, case3) 결정은 추후 visualize 단계에서 수행
          - 결정된 majority label과 centroid embedding과의 cosine similarity 계산
        """
        import torch.nn.functional as F
        from collections import Counter

        if self.tracked_indices is None:
            self.setup_tracking(batch)

        if epoch not in self.tracking_data:
            self.tracking_data[epoch] = {}

        for batch_idx, ptr_idx in self.tracked_indices:
            try:
                orig_feat = original_features[ptr_idx].detach().cpu().numpy()
                prompted_feat = prompted_features[ptr_idx].detach().cpu().numpy()
                prompt_feat = node_wise_prompt[ptr_idx].detach().cpu().numpy()
                distance = np.linalg.norm(orig_feat - prompted_feat)

                node_label = batch.y[ptr_idx].item()  # 노드의 라벨

                # 이웃 노드 검색 (self-loop 제외)
                edge_index = batch.edge_index
                mask = (edge_index[0] == ptr_idx) | (edge_index[1] == ptr_idx)
                relevant_edges = edge_index[:, mask]
                neighbors = []
                for i in range(relevant_edges.size(1)):
                    src = relevant_edges[0, i].item()
                    dst = relevant_edges[1, i].item()
                    if src == ptr_idx and dst != ptr_idx:
                        neighbors.append(dst)
                    elif dst == ptr_idx and src != ptr_idx:
                        neighbors.append(src)
                # fallback: 만약 self-loop 제외 이웃이 없다면, 모든 연결된 노드 사용
                if len(neighbors) == 0:
                    all_neighbors = []
                    for i in range(relevant_edges.size(1)):
                        src = relevant_edges[0, i].item()
                        dst = relevant_edges[1, i].item()
                        if src == ptr_idx:
                            all_neighbors.append(dst)
                        elif dst == ptr_idx:
                            all_neighbors.append(src)
                    neighbors = all_neighbors

                neighbor_labels = [batch.y[n].item() for n in neighbors] if len(neighbors) > 0 else [node_label]

                # 기존 다수결 majority label (추후 참고용)
                def get_majority_label(neighbor_list, fallback_label):
                    if len(neighbor_list) == 0:
                        return fallback_label
                    counter = Counter(neighbor_list)
                    max_count = max(counter.values())
                    candidates = [label for label, count in counter.items() if count == max_count]
                    return candidates[0]

                majority_label = get_majority_label(neighbor_labels, fallback_label=node_label)

                # 코사인 유사도 계산 (centroid은 tracker.centroids에 저장되어 있어야 함)
                cosine_sim = None
                if (self.centroids is not None) and (majority_label in self.centroids):
                    centroid_embedding = self.centroids[majority_label]
                    prompted_tensor = torch.tensor(prompted_feat, dtype=torch.float32)
                    centroid_tensor = torch.tensor(centroid_embedding, dtype=torch.float32)
                    cosine_sim = F.cosine_similarity(prompted_tensor.unsqueeze(0),
                                                     centroid_tensor.unsqueeze(0)).item()

                # tracking_data에 저장 (노드의 원래 라벨과 이웃 라벨도 저장)
                self.tracking_data[epoch][ptr_idx] = {
                    'original': orig_feat,
                    'prompted': prompted_feat,
                    'prompt': prompt_feat,
                    'distance': distance,
                    'node_label': node_label,
                    'neighbor_labels': neighbor_labels,
                    'majority_label': majority_label,
                    'cosine_similarity': cosine_sim
                }

                print(
                    f"Epoch {epoch}, Tracked node {ptr_idx} (Node Label: {node_label}, Majority Label: {majority_label}, Cosine Sim: {cosine_sim})")
            except IndexError:
                print(f"Warning: Could not track node at position {ptr_idx} in current batch")

    def save_tracking_logs(self, epoch: int):
        """각 노드별로 전체 epoch 동안의 변화를 하나의 파일에 저장"""
        if epoch not in self.tracking_data or not self.tracking_data[epoch]:
            print(f"No tracking data available for epoch {epoch}")
            return

        # 각 노드별로 파일 생성/업데이트
        for node_id, data in self.tracking_data[epoch].items():
            # 원본 특징 파일
            orig_file = os.path.join(self.tracking_log_dir, f'node_{node_id}_original_features.txt')
            with open(orig_file, 'a') as f:  # append 모드
                f.write(f"\nEpoch {epoch}:\n")
                np.savetxt(f, data['original'].reshape(1, -1))

            # 프롬프트된 특징 파일
            prompted_file = os.path.join(self.tracking_log_dir, f'node_{node_id}_prompted_features.txt')
            with open(prompted_file, 'a') as f:  # append 모드
                f.write(f"\nEpoch {epoch}:\n")
                np.savetxt(f, data['prompted'].reshape(1, -1))

            # node-wise prompt 특징 파일
            node_wise_prompt_file = os.path.join(self.tracking_log_dir, f'node_{node_id}_nodewiseprompt_features.txt')
            with open(node_wise_prompt_file, 'a') as f:  # append 모드
                f.write(f"\nEpoch {epoch}:\n")
                np.savetxt(f, data['prompt'].reshape(1, -1))

            # 거리 변화 파일
            distance_file = os.path.join(self.tracking_log_dir, f'node_{node_id}_distances.txt')
            with open(distance_file, 'a') as f:  # append 모드
                f.write(f"Epoch {epoch}: {data['distance']:.6f}\n")

            # 코사인 유사도 파일: 첫 줄에 자기 자신의 레이블과 최빈 이웃 레이블 명시
            cosine_file = os.path.join(self.tracking_log_dir, f'node_{node_id}_cosine_similarity.txt')
            with open(cosine_file, 'a') as f:  # append 모드
                node_label = data.get('node_label', 'N/A')
                majority_label = data.get('majority_label', 'N/A')
                # 헤더로 레이블 정보 출력
                f.write(f"\nEpoch {epoch}: Node Label: {node_label}, Majority Label: {majority_label}\n")
                np.savetxt(f, np.array([data.get('cosine_similarity', np.nan)]).reshape(1, -1))

    def visualize(self, epoch: int, method: str = 'PCA'):
        """각 노드별로 전체 epoch의 변화를 시각화
        - 원본(Original) 시각화: 원본 특징과 node-wise prompt를 하나의 플롯에 함께 표시
        - 프롬프트(Prompted) 특징 시각화
        - 거리 변화 시각화
        - 코사인 유사도 변화 시각화
        또한, 각 노드에 대해 DefaultNodeClassifier 로직에 따라 분류(case1, case2, case3)를 결정하여 타이틀에 출력합니다.
        """
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        if epoch not in self.tracking_data or not self.tracking_data[epoch]:
            print(f"No tracking data available for epoch {epoch}")
            return

        # 내부 함수: DefaultNodeClassifier의 로직을 간단히 구현
        def classify_node(node_label, neighbor_labels):
            # case1: 모든 이웃 노드의 라벨이 현재 노드와 동일한 경우
            if neighbor_labels and all(nl == node_label for nl in neighbor_labels):
                return 'case1'
            unique_neighbors = list(set(neighbor_labels))
            if len(unique_neighbors) == 1 and unique_neighbors[0] != node_label:
                return 'case2'
            elif len(unique_neighbors) > 1:
                return 'case3'
            else:
                return 'unknown'

        for node_id in self.tracking_data[epoch].keys():
            epochs = []
            orig_features_list = []
            prompt_features_list = []
            prompted_features = []
            distances = []
            cosine_sims = []

            # 각 epoch에 대해 데이터 수집
            for e in range(0, epoch + 1, self.track_interval):
                if e in self.tracking_data and node_id in self.tracking_data[e]:
                    epochs.append(e)
                    data_entry = self.tracking_data[e][node_id]
                    orig_features_list.append(data_entry['original'])
                    prompt_features_list.append(data_entry['prompt'])
                    prompted_features.append(data_entry['prompted'])
                    distances.append(data_entry['distance'])
                    cosine_val = data_entry.get('cosine_similarity', None)
                    cosine_sims.append(cosine_val if cosine_val is not None else np.nan)

            if not epochs:
                continue

            # 최종 epoch의 데이터를 사용하여 case type 결정
            final_data = self.tracking_data[epochs[-1]][node_id]
            node_label = final_data.get('node_label', None)
            neighbor_labels = final_data.get('neighbor_labels', [])
            case_type = classify_node(node_label, neighbor_labels)

            # --- Combined Original & Node-wise Prompt Visualization ---
            orig_array = np.array(orig_features_list)
            prompt_array = np.array(prompt_features_list)
            combined_data = np.vstack([orig_array, prompt_array])
            if combined_data.shape[0] < 2:
                combined_2d = np.hstack([combined_data, np.zeros((combined_data.shape[0], 1))])
                orig_2d = combined_2d[:len(orig_features_list)]
                prompt_2d = combined_2d[len(orig_features_list):]
            else:
                reducer = PCA(n_components=2)
                combined_2d = reducer.fit_transform(combined_data)
                orig_2d = combined_2d[:len(orig_features_list)]
                prompt_2d = combined_2d[len(orig_features_list):]

            plt.figure(figsize=(10, 6))
            plt.scatter(orig_2d[:, 0], orig_2d[:, 1], c='blue', marker='o', label='Original Feature')
            plt.scatter(prompt_2d[:, 0], prompt_2d[:, 1], c='green', marker='x', label='Node-wise Prompt')
            if len(orig_features_list) > 1:
                for i in range(len(orig_features_list) - 1):
                    plt.arrow(orig_2d[i, 0], orig_2d[i, 1],
                              orig_2d[i + 1, 0] - orig_2d[i, 0],
                              orig_2d[i + 1, 1] - orig_2d[i, 1],
                              color='blue', alpha=0.3)
            if len(prompt_features_list) > 1:
                for i in range(len(prompt_features_list) - 1):
                    plt.arrow(prompt_2d[i, 0], prompt_2d[i, 1],
                              prompt_2d[i + 1, 0] - prompt_2d[i, 0],
                              prompt_2d[i + 1, 1] - prompt_2d[i, 1],
                              color='green', alpha=0.3)
            plt.title(f'Node {node_id} Combined Original & Node-wise Prompt Features Evolution\nCase: {case_type}')
            plt.legend()
            plt.savefig(os.path.join(self.visualization_dir, f'node_{node_id}_original.png'))
            plt.close()

            # --- Prompted Features Visualization (기존 방식) ---
            prompted_data = np.vstack(prompted_features)
            if prompted_data.shape[0] < 2:
                prompted_2d = np.hstack([prompted_data, np.zeros((prompted_data.shape[0], 1))])
            else:
                reducer = PCA(n_components=2)
                prompted_2d = reducer.fit_transform(prompted_data)

            plt.figure(figsize=(10, 6))
            plt.scatter(prompted_2d[:, 0], prompted_2d[:, 1], c=epochs, cmap='Reds')
            if len(epochs) > 1:
                for i in range(len(epochs) - 1):
                    plt.arrow(prompted_2d[i, 0], prompted_2d[i, 1],
                              prompted_2d[i + 1, 0] - prompted_2d[i, 0],
                              prompted_2d[i + 1, 1] - prompted_2d[i, 1],
                              color='red', alpha=0.3)
            plt.colorbar(label='Epoch')
            plt.title(f'Node {node_id} Prompted Features Evolution\nCase: {case_type}')
            plt.savefig(os.path.join(self.visualization_dir, f'node_{node_id}_prompted.png'))
            plt.close()

            # --- Distance Plot ---
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, distances, marker='o', color='purple')
            plt.title(f'Node {node_id} Original-Prompted Distance\nCase: {case_type}')
            plt.xlabel('Epoch')
            plt.ylabel('Distance')
            plt.grid(True)
            plt.savefig(os.path.join(self.visualization_dir, f'node_{node_id}_distances.png'))
            plt.close()

            # --- Cosine Similarity Plot ---
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, cosine_sims, marker='o', color='orange')
            plt.title(f'Node {node_id} Cosine Similarity\nCase: {case_type}')
            plt.xlabel('Epoch')
            plt.ylabel('Cosine Similarity')
            plt.grid(True)
            plt.savefig(os.path.join(self.visualization_dir, f'node_{node_id}_cosine_similarity.png'))
            plt.close()

        print(f"Saved visualizations for epoch {epoch}")

    def get_tracking_results(self) -> dict:
        """전체 추적 결과 반환"""
        return self.tracking_data

    def should_track(self, epoch: int) -> bool:
        """주어진 에폭에서 추적을 수행할지 결정"""
        return epoch % self.track_interval == 0


