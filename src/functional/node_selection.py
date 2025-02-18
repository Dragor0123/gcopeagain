import torch
from .node_classifier import DefaultNodeClassifier
import random
from torch_geometric.data import Batch


def get_valid_neighbors_for_node(batch, local_idx, node_labels):
    """
    현재 노드(local_idx)와 연결된 유효한 이웃 노드 인덱스를 반환합니다.
    배치의 edge_index를 사용하여, 현재 노드와 연결된 edge의 대상 노드들을 추출하며,
    self-loop (자기 자신으로의 연결)는 배제합니다.

    :param batch: 노드 특성(batch.x), 라벨(batch.y) 및 edge_index(batch.edge_index)를 포함하는 배치.
    :param local_idx: 현재 노드의 인덱스 (정수 또는 스칼라 텐서).
    :param node_labels: (필요 시 사용, 여기서는 사용하지 않음)
    :return: 유효한 이웃 노드 인덱스를 담은 torch.Tensor.
    """
    # 현재 노드와 연결된 edge들을 찾음
    edge_mask = (batch.edge_index[0] == local_idx)
    neighbors = batch.edge_index[1][edge_mask]
    # self-loop(현재 노드와 자신과의 연결)를 배제
    valid_neighbors = neighbors[neighbors != local_idx]
    return valid_neighbors

def assign_global_node_indices(dataset, classifier=None):
    """
    전체 dataset을 하나의 Batch로 결합한 후, 전역적으로 노드 이웃을 분석하여 노드 매핑을 생성합니다.
    heterophilic 관계(즉, case2, case3)를 더 잘 드러내기 위한 방법.
    """
    global_node_count = 0
    node_mappings = {'case1': [], 'case2': [], 'case3': []}

    # 전체 dataset을 하나의 Batch로 결합
    full_data = Batch.from_data_list(dataset)
    if full_data.y.size(0) == full_data.x.size(0):
        print("DEBUG: Using node-level labels")
        node_labels = full_data.y
    else:
        print(
            "DEBUG: Using graph-level labels (full_data.y does not match full_data.x rows; using full_data.y[full_data.batch])")
        node_labels = full_data.y[full_data.batch]

    num_nodes = full_data.x.size(0)
    # 전체 edge_index 사용하여 각 노드의 이웃 계산
    for local_idx in range(num_nodes):
        # self-loop 배제
        edge_mask = (full_data.edge_index[0] == local_idx)
        neighbors = full_data.edge_index[1][edge_mask]
        valid_neighbors = neighbors[neighbors != local_idx]
        if len(valid_neighbors) == 0:
            continue

        node_label = node_labels[local_idx]
        neighbor_labels = node_labels[valid_neighbors]
        node_features = full_data.x[local_idx].clone().cpu()

        # 기존 분류 로직에 따라 case 분류
        if all(nl == node_label for nl in neighbor_labels):
            node_mappings['case1'].append((global_node_count, node_features))
        elif len(torch.unique(neighbor_labels)) == 1 and neighbor_labels[0] != node_label:
            node_mappings['case2'].append((global_node_count, node_features))
        elif len(torch.unique(neighbor_labels)) > 1:
            node_mappings['case3'].append((global_node_count, node_features))
        # (원하는 경우 classifier를 사용해도 됨)

        global_node_count += 1

    # 결과 출력
    for case, nodes in node_mappings.items():
        print(f"Found {len(nodes)} nodes for {case}")

    return node_mappings


# def select_nodes_for_tracking(node_mappings, num_per_case=2):
#     """
#     각 case별로 주어진 수(num_per_case)만큼 노드를 선택하여 반환합니다.
#
#     :param node_mappings: assign_global_node_indices에서 생성된 딕셔너리.
#     :param num_per_case: 각 case별 선택할 노드 수 (기본값 2).
#     :return: 선택된 노드들의 리스트 (튜플: (global_node_index, 노드 특성)).
#     """
#     selected_nodes = []
#     for case in node_mappings:
#         selected_nodes.extend(node_mappings[case][:num_per_case])
#     return selected_nodes


def select_nodes_for_tracking(node_mappings: dict, num_per_case: int = 2):
    """각 case별로 추적할 노드 선택"""
    tracked_nodes = {}
    for case_type, nodes in node_mappings.items():
        if len(nodes) >= num_per_case:
            tracked_nodes[case_type] = random.sample(nodes, num_per_case)
        else:
            tracked_nodes[case_type] = nodes
    return tracked_nodes
