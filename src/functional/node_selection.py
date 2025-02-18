import torch
from .node_classifier import DefaultNodeClassifier
import random


def get_valid_neighbors_for_node(batch, local_idx, node_labels):
    """
    현재 노드(local_idx)와 연결된 유효한 이웃 노드 인덱스를 반환합니다.

    배치의 edge_index를 사용하여, 현재 노드와 연결된 edge의 대상 노드들을 추출하고,
    이 중 노드 라벨 텐서 범위 내에 있는 인덱스만 반환합니다.

    :param batch: 노드 특성(batch.x), 라벨(batch.y) 및 edge_index(batch.edge_index)를 포함하는 배치.
    :param local_idx: 현재 노드의 인덱스 (정수 또는 스칼라 텐서).
    :param node_labels: 노드 라벨 텐서 (이미 노드별 라벨이 존재하거나 확장된 라벨).
    :return: 유효한 이웃 노드 인덱스를 담은 torch.Tensor.
    """
    # 현재 노드와 연결된 edge들을 찾음
    edge_mask = (batch.edge_index[0] == local_idx)
    neighbors = batch.edge_index[1][edge_mask]
    # 노드 라벨 범위 내의 인덱스만 선택 (안전하게 처리)
    valid_neighbors = neighbors[neighbors < len(node_labels)]
    return valid_neighbors


def assign_global_node_indices(data_loader, classifier=None):
    global_node_count = 0
    node_mappings = {'case1': [], 'case2': [], 'case3': []}

    for batch in data_loader:
        if batch.y.size(0) == batch.x.size(0):
            node_labels = batch.y
        else:
            node_labels = batch.y[batch.batch]

        batch_size = int(batch.batch.max().item() + 1)
        for graph_idx in range(batch_size):
            graph_mask = batch.batch == graph_idx
            graph_nodes = torch.where(graph_mask)[0]

            for local_idx in graph_nodes:
                valid_neighbors = get_valid_neighbors_for_node(batch, local_idx, node_labels)
                if len(valid_neighbors) == 0:
                    continue

                node_label = node_labels[local_idx]
                neighbor_labels = node_labels[valid_neighbors]

                # CPU로 이동하는 부분을 공통으로 처리
                node_features = batch.x[local_idx].clone().cpu()

                #if classifier is not None:
                if False:
                    case = classifier.classify(node_label, neighbor_labels)
                    if case in node_mappings:
                        node_mappings[case].append((global_node_count, node_features))
                else:
                    # 기존 
                    if all(nl == node_label for nl in neighbor_labels):
                        node_mappings['case1'].append((global_node_count, node_features))
                    elif len(torch.unique(neighbor_labels)) == 1 and neighbor_labels[0] != node_label:
                        node_mappings['case2'].append((global_node_count, node_features))
                    elif len(torch.unique(neighbor_labels)) > 1:
                        node_mappings['case3'].append((global_node_count, node_features))

                global_node_count += 1

    # 결과 출력 추가
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
