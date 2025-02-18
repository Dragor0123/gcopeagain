def find_subgraph_case_by_label_count(data, mask):
    """
    input : data - graph dataset
    output : case_type을 key값, case_type에 해당하는 sample로 구성된 list를 value로 삼는 dictionary.
    하는 일 : data sample 중 mask로 지정된 일부 set(e.g., training set)에 포함된 sample 각각에 대하여 sample의 case_type을 판단함.
        case1 : 자기 자신의 label과 자신과 1-hop 연결된 모든 이웃의 label이 같은 경우.
        case2 : 자기 자신의 label과 모든 이웃의 label이 같지 않음. 그러나 이웃끼리는 label이 모두 같은 경우.
        case3 : 자기 자신과 label과 모든 이웃의 label이 같지 않음. 그리고 이웃끼리도 label이 같지 않은 경우.
        case_type별로 샘플 리스트를 구성하고 난 후, key를 case_type, value를 그 case에 해당하는 sample로 구성된 리스트로 된 dictionary를 반환함.
    """
    G = nx.Graph()
    # Convert edge_index tensor to list of tuples and add edges
    edges = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    G.add_edges_from(edges)
    # Initialize result dictionary
    case1_nodes, case2_nodes, case3_nodes = [], [], []
    # Process each node
    for node in G.nodes():
        # Create a set to store unique labels
        label_set = {int(data.y[node])}  # Include the center node label
        # Get neighbor labels
        neighbors = list(G.neighbors(node))
        node_class = int(data.y[node])
        neighbor_classes = [int(data.y[neighbor]) for neighbor in neighbors]
        # Case 1: 중심 노드와 모든 이웃 노드의 클래스가 동일
        if all(nc == node_class for nc in neighbor_classes):
            case1_nodes.append(node)
        # Case 2: 중심 노드와 이웃 노드의 클래스는 다르지만, 이웃 노드들끼리는 동일한 클래스
        elif len(set(neighbor_classes)) == 1 and neighbor_classes[0] != node_class:
            case2_nodes.append(node)
        # Case 3: 중심 노드의 이웃 노드들이 2개 이상의 서로 다른 클래스를 가짐
        elif len(set(neighbor_classes)) > 1:
            case3_nodes.append(node)
    return {"case1": case1_nodes, "case2": case2_nodes, "case3": case3_nodes}

