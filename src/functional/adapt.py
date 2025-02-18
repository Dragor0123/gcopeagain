import copy
from fastargs.decorators import param
from fastargs import Param
import torch
from .graph_prompt_feature import *

from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
from tqdm import tqdm
from copy import deepcopy
import random
from .embedding_tracker import EmbeddingTracker
from .node_selection import assign_global_node_indices, select_nodes_for_tracking

from collections import Counter
from torch_geometric.data import Data



@param('data.name', 'dataset')
@param('adapt.batch_size')
@param('data.supervised.ratios')
@param('adapt.method')
@param('model.backbone.model_type', 'backbone_model')
@param('model.saliency.model_type', 'saliency_model')
@param('model.answering.model_type', 'answering_model')
@param('adapt.pretrained_file')
@param('general.save_dir')
@param('adapt.repeat_times')
def run(
    dataset,
    batch_size,
    ratios,
    method,
    backbone_model,
    saliency_model,
    answering_model,
    pretrained_file,
    save_dir,
    repeat_times,
    ):
    
    # load data
    from data import get_supervised_data
    from torch_geometric.loader import DataLoader
    datasets, num_classes = get_supervised_data(dataset[0], ratios=ratios)
    loaders = { k: DataLoader(v, batch_size=batch_size, shuffle=True, num_workers=4) for k, v in datasets.items() }

    # init model
    from model import get_model
    model = get_model(
        backbone_kwargs = {
            'name': backbone_model,
            'num_features': datasets['train'][0].x.size(-1),
        },
        answering_kwargs = {
            'name': answering_model,
            'num_class': num_classes,
        },
        saliency_kwargs = {
            'name': saliency_model,
            'feature_dim': datasets['train'][0].x.size(-1),
        } if saliency_model != 'none' else None,
    )

    model.load_state_dict(torch.load(pretrained_file,map_location=lambda storage, loc: storage.cuda(0)), strict=False)

    # train
    all_results = []
    for _ in range(repeat_times):
        if method == 'finetune':
            results = finetune(loaders, model)
        elif method == 'prog':
            from model import get_prompt_model
            # statistic the average node number of dataset
            total_graph = sum([len(v) for k, v in datasets.items()])
            train_node_num = sum([g.num_nodes for g in datasets['train']])
            val_node_num = sum([g.num_nodes for g in datasets['val']])
            test_node_num = sum([g.num_nodes for g in datasets['test']])
            prompt_node_num = int((train_node_num + val_node_num + test_node_num) / total_graph)
            prompt_model = get_prompt_model(num_features=datasets['train'][0].x.size(-1), prompt_node_num=prompt_node_num)
            results = prog(loaders, model, prompt_model, dataset)
        elif method == 'gpf':
            results = gpf(loaders=loaders,
                          model=model,
                          dataset=dataset)
        else:
            raise NotImplementedError(f'Unknown method: {method}')
        
        results.pop('model')
        all_results.append(results)        

    # print acc, auroc, f1 with std
    import numpy as np
    for k in all_results[0].keys():
        print(f'{k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}')
        
    import os

    if(method!='prog'):
        with open(os.path.join(save_dir, dataset[0]+'_results.txt'), 'a+') as f:
            f.write('-------------------------------------------------\n')
            for k in all_results[0].keys():
                f.write(method+f'FT on All, Target Dataset: {dataset[0]}, {k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}\n')
    else:
        with open(os.path.join(save_dir, dataset[0]+'_results.txt'), 'a+') as f:
            f.write('-------------------------------------------------\n')
            for k in all_results[0].keys():
                f.write(method+f' on All, Target Dataset: {dataset[0]}, {k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}\n')                        

    # torch.save(results, os.path.join(save_dir, dataset[0]+'_results.pt'))

@param('adapt.finetune.backbone_tuning')
@param('adapt.finetune.saliency_tuning')
@param('adapt.finetune.learning_rate')
@param('adapt.finetune.weight_decay')
@param('adapt.epoch')
def finetune(
        loaders,
        model,
        backbone_tuning,
        saliency_tuning,
        learning_rate,
        weight_decay,
        epoch,
        ):

    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = learning_rate,
        weight_decay = weight_decay,
        )

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)

    best_acc = 0.
    best_model = None

    for e in range(epoch):
        model.train()

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()

        pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100, desc=f'Epoch {e} Training, Loss: inf')

        for batch in pbar:
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = model(batch)
            loss = torch.nn.functional.cross_entropy(pred, batch.y)
            loss.backward()
            optimizer.step()

            loss_metric.update(loss.detach(), batch.size(0))
            pbar.set_description(f'Epoch {e} Training Loss: {loss_metric.compute():.4f}', refresh=True)
        pbar.close()

        model.eval()

        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                pred = model(batch).argmax(dim=-1)

                acc_metric.update(pred, batch.y)
                f1_metric.update(pred, batch.y)
                auroc_metric.update(model(batch), batch.y)
                pbar.set_description(f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
            pbar.close()

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_model = deepcopy(model)
    
    model = best_model if best_model is not None else model

    # test
    model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=-1)

            acc_metric.update(pred, batch.y)
            f1_metric.update(pred, batch.y)
            auroc_metric.update(model(batch), batch.y)
            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        pbar.close()
    
    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
    }

@param('adapt.epoch')
@param('adapt.prog.prompt_lr')
@param('adapt.prog.prompt_weight_decay')
@param('adapt.prog.ans_lr')
@param('adapt.prog.ans_weight_decay')
@param('adapt.prog.backbone_tuning')
@param('adapt.prog.saliency_tuning')
def prog(
        loaders,
        model,
        prompt_model,      
        dataset,
        epoch,
        backbone_tuning,
        saliency_tuning,          
        prompt_lr,
        prompt_weight_decay,
        ans_lr,
        ans_weight_decay,
        ):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.backbone.to(device)
    model.answering.to(device)
    prompt_model.to(device)
    
    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)

    opi_pg = torch.optim.Adam(
        prompt_model.parameters(),
        lr = prompt_lr,
        weight_decay = prompt_weight_decay,
        )
    
    opi_answer = torch.optim.Adam(
        model.answering.parameters(),
        lr = ans_lr,
        weight_decay = ans_weight_decay,
        )

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)
    
    # load prompting data

    from torch_geometric.loader import DataLoader

    best_acc = 0.
    best_backbone = None
    best_prompt_model = None
    best_answering = None

    for e in range(epoch):

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()
        
        print(("{}/{} frozen gnn | *tune prompt and tune answering function...".format(e, epoch)))
        prompt_model.train()
        model.backbone.eval()
        model.answering.train()

        from tqdm import tqdm

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        running_loss = 0.
        
        ans_pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100, desc=f'Epoch {e} / Total Epoch {epoch} Training, Loss: inf')

        for batch_id, train_batch in enumerate(ans_pbar):  # bar2       
            
            train_batch = train_batch.to(device)
            prompted_graph = prompt_model(train_batch)

            graph_emb = model.backbone(prompted_graph)

            # print(graph_emb)
            pred = model.answering(graph_emb)
            # print(pre)
            train_loss = torch.nn.functional.cross_entropy(pred, train_batch.y)

            opi_answer.zero_grad()
            opi_pg.zero_grad()
            train_loss.backward()
            opi_answer.step()
            opi_pg.step()
            running_loss += train_loss.item()

            current_avg_last_loss = running_loss / (batch_id+1)  # loss per batch

            ans_pbar.set_description('Epoch {} / Total Epoch {} | avg loss: {:.8f}'.format(e, epoch, current_avg_last_loss), refresh=True)
        
        ans_pbar.close()        
                
        model.backbone.eval()
        prompt_model.eval()
        model.answering.eval()

        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():
            for batch in pbar:              
                batch = batch.to(device)
                prompted_graph = prompt_model(batch)
                z = model.backbone(prompted_graph)
                pred = model.answering(z).argmax(dim=-1)

                acc_metric.update(pred, batch.y)
                f1_metric.update(pred, batch.y)
                auroc_metric.update(model(prompted_graph), batch.y)
                pbar.set_description(f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
            pbar.close()

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_backbone = deepcopy(model.backbone)
            best_answering = deepcopy(model.answering)
            best_prompt_model = deepcopy(prompt_model)
    
    model.backbone = best_backbone if best_backbone is not None else model.backbone
    model.answering = best_answering if best_answering is not None else model.answering
    prompt_model = best_prompt_model if best_prompt_model is not None else prompt_model

    # test
    model.backbone.eval()
    model.answering.eval()
    prompt_model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            prompted_graph = prompt_model(batch)
            z = model.backbone(prompted_graph)
            pred = model.answering(z).argmax(dim=-1)

            acc_metric.update(pred, batch.y)
            f1_metric.update(pred, batch.y)
            auroc_metric.update(model(prompted_graph), batch.y)
            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        pbar.close()

    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
    }

import numpy as np


def compute_class_centroids(data_loader):
    """
    data_loader: torch_geometric.data.DataLoader 객체 (예: loaders['train'])
      각 배치에서, 만약 batch.y.size(0) == batch.x.size(0)라면 node-level label을 직접 사용하고,
      그렇지 않으면 batch.y[batch.batch]를 사용하여 각 노드의 label을 결정합니다.
    반환:
      centroids: dict {class_label: centroid_embedding (NumPy array)}
    """
    import numpy as np

    all_features = []
    all_labels = []

    for batch in data_loader:
        # 노드 레이블 결정: 각 배치마다
        if batch.y.size(0) == batch.x.size(0):
            node_labels = batch.y.cpu().numpy()
        else:
            node_labels = batch.y[batch.batch].cpu().numpy()

        all_features.append(batch.x.cpu().numpy())
        all_labels.append(node_labels)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    centroids = {}
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        features_of_class = all_features[all_labels == label]
        centroids[label] = np.mean(features_of_class, axis=0)

    return centroids


import os
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC

@param('adapt.epoch')
@param('adapt.gpf.prompt_lr')
@param('adapt.gpf.prompt_weight_decay')
@param('adapt.gpf.prompt_basis_num')
@param('adapt.gpf.ans_lr')
@param('adapt.gpf.ans_weight_decay')
@param('adapt.gpf.backbone_tuning')
@param('adapt.gpf.saliency_tuning')
def gpf(
        loaders,
        model,
        dataset,
        epoch,
        backbone_tuning,
        saliency_tuning,
        prompt_lr,
        prompt_weight_decay,
        prompt_basis_num,
        ans_lr,
        ans_weight_decay,
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 백본 동결 및 디바이스 설정
    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)
    model.backbone.to(device)
    model.answering.to(device)

    raw_feature_dim = loaders['train'].dataset[0].x.size(-1)
    # GPF 모듈 초기화 (모델 내부의 add() 메서드가 프롬프트 적용을 수행한다고 가정)
    prompt_model = GPF(
        in_channels=raw_feature_dim,
        p_num=prompt_basis_num
    ).to(device)

    # 옵티마이저 설정
    prompt_optimizer = torch.optim.Adam(
        prompt_model.parameters(),
        lr=prompt_lr,
        weight_decay=prompt_weight_decay
    )
    answering_optimizer = torch.optim.Adam(
        model.answering.parameters(),
        lr=ans_lr,
        weight_decay=ans_weight_decay
    )

    # 평가 메트릭 초기화
    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)


    # ----- 모듈화된 노드 선정/트래킹 파트 시작 -----
    # 첫 번째 epoch에서 노드 분류 (모듈화된 함수 사용)
    node_mappings = assign_global_node_indices(loaders['train'])
    # 추적할 노드 선택 (각 case별 2개)
    tracked_nodes = select_nodes_for_tracking(node_mappings, num_per_case=2)
    # 임베딩 추적기 초기화 (예: 10 epoch마다 추적)
    tracker = EmbeddingTracker(
        tracked_nodes=tracked_nodes,
        track_interval=10,
        dataset_name=dataset[0]
    )
    # ----- 모듈화된 노드 선정/트래킹 파트 끝 -----

    best_acc = 0.
    best_model = None
    best_prompt = None

    centroids = compute_class_centroids(loaders['train'])
    tracker.set_centroids(centroids)

    for e in range(epoch):
        prompt_model.train()
        model.backbone.eval()  # 백본 고정
        model.answering.train()

        loss_metric.reset()
        pbar = tqdm(loaders['train'], desc=f'Epoch {e} Training')

        for batch in pbar:
            batch = batch.to(device)

            # 원본 특징 저장
            original_features = batch.x.clone()

            # Node-wise prompting
            prompted_features, node_wise_prompt = prompt_model.add(batch.x)
            batch.x = prompted_features

            # 추적 데이터 저장 (해당 epoch이 추적 시점인 경우)
            if tracker.should_track(e):
                tracker.track_embeddings(e, batch, original_features, prompted_features, node_wise_prompt)

            # Frozen backbone으로 노드 임베딩 생성
            with torch.no_grad():
                node_embeddings = model.backbone(batch)

            # 노드 분류 및 손실 계산
            pred = model.answering(node_embeddings)
            loss = F.cross_entropy(pred, batch.y)

            # 역전파
            prompt_optimizer.zero_grad()
            answering_optimizer.zero_grad()
            loss.backward()
            prompt_optimizer.step()
            answering_optimizer.step()

            loss_metric.update(loss.item())
            pbar.set_description(f'Epoch {e}, Loss: {loss_metric.compute():.4f}')

        # 검증 단계
        prompt_model.eval()
        model.backbone.eval()
        model.answering.eval()

        val_pbar = tqdm(loaders['val'], desc='Validation')
        with torch.no_grad():
            for batch in val_pbar:
                batch = batch.to(device)
                prompted_features, _ = prompt_model.add(batch.x)
                batch.x = prompted_features

                node_embeddings = model.backbone(batch)
                pred = model.answering(node_embeddings)

                acc_metric.update(pred.argmax(dim=-1), batch.y)
                f1_metric.update(pred.argmax(dim=-1), batch.y)
                auroc_metric.update(pred, batch.y)

                val_pbar.set_description(
                    f'Validation Acc: {acc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}'
                )

        # epoch마다 추적 데이터 저장 및 시각화 (추적 대상 epoch인 경우)
        if tracker.should_track(e):
            tracker.save_tracking_logs(e)
            tracker.visualize(e)

        # 최고 성능 모델 저장
        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_model = copy.deepcopy(model)
            best_prompt = copy.deepcopy(prompt_model)

    # 테스트 단계
    model = best_model if best_model is not None else model
    prompt_model = best_prompt if best_prompt is not None else prompt_model

    model.eval()
    prompt_model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    test_pbar = tqdm(loaders['test'], desc='Testing')
    with torch.no_grad():
        for batch in test_pbar:
            batch = batch.to(device)
            prompted_features, _ = prompt_model.add(batch.x)
            batch.x = prompted_features

            node_embeddings = model.backbone(batch)
            pred = model.answering(node_embeddings)

            acc_metric.update(pred.argmax(dim=-1), batch.y)
            f1_metric.update(pred.argmax(dim=-1), batch.y)
            auroc_metric.update(pred, batch.y)

            test_pbar.set_description(
                f'Test Acc: {acc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}'
            )

    #tracking_results = tracker.get_tracking_results()

    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
    }
