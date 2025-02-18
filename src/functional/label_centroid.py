import numpy as np


def compute_class_centroids(training_dataset):
    """
    training_dataset: torch_geometric.data.Data 객체들이 담긴 리스트
                      각 Data 객체는 .x (노드 raw feature, Tensor)와 .y (노드 label, Tensor)를 가짐
    반환:
      centroids: dict {class_label: centroid_embedding (NumPy array)}
    """
    all_features = []
    all_labels = []
    for data in training_dataset:
        # data.x: (num_nodes, feature_dim), data.y: (num_nodes,)
        all_features.append(data.x.cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    centroids = {}
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        # 해당 label에 해당하는 모든 노드의 raw feature 평균
        features_of_class = all_features[all_labels == label]
        centroids[label] = np.mean(features_of_class, axis=0)
    return centroids
