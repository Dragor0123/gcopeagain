import torch

class BaseNodeClassifier:
    """
    노드 분류를 위한 추상 기본 클래스.
    하위 클래스는 classify 메서드를 구현해야 합니다.
    """
    def classify(self, node_label, neighbor_labels, **kwargs):
        """
        주어진 노드의 라벨과 이웃 노드들의 라벨을 바탕으로 분류(case)를 결정합니다.
        :param node_label: 현재 노드의 라벨
        :param neighbor_labels: 이웃 노드들의 라벨 (iterable 또는 torch.Tensor)
        :return: 분류 결과(case 이름, 예: 'case1', 'case2', 'case3')
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")


class DefaultNodeClassifier(BaseNodeClassifier):
    """
    기본 노드 분류기:
      - case1: 모든 이웃 노드의 라벨이 현재 노드의 라벨과 동일한 경우.
      - case2: 이웃 노드들의 라벨이 모두 동일하지만, 현재 노드의 라벨과 다른 경우.
      - case3: 이웃 노드들에 두 개 이상의 고유 라벨이 존재하는 경우.
    """

    def classify(self, node_label, neighbor_labels, **kwargs):
        # neighbor_labels가 torch.Tensor가 아니라면 list로 변환
        if not isinstance(neighbor_labels, torch.Tensor):
            neighbor_labels_tensor = torch.tensor(list(neighbor_labels))
        else:
            neighbor_labels_tensor = neighbor_labels

        # case1: 모든 이웃 노드의 라벨이 현재 노드와 동일한 경우
        if all(nl == node_label for nl in neighbor_labels_tensor.tolist()):
            return 'case1'

        # 고유한 이웃 라벨 계산
        unique_neighbors = torch.unique(neighbor_labels_tensor)
        if len(unique_neighbors) == 1 and unique_neighbors[0] != node_label:
            return 'case2'
        elif len(unique_neighbors) > 1:
            return 'case3'
        else:
            return 'unknown'