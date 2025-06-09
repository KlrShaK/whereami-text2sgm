import time
import argparse
import sys
import torch
import torch.cuda
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import wandb
import random
import matplotlib.pyplot as plt

sys.path.append('../data_processing') # sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
sys.path.append('../../../') # sys.path.append('/home/julia/Documents/h_coarse_loc/')
from scene_graph import SceneGraph
from data_distribution_analysis.helper import get_matching_subgraph, calculate_overlap
from model_graph2graph import BigGNN
from train_utils import k_fold, cross_entropy, k_fold_by_scene

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.current_device())

random.seed(42)

from args import get_args
args = get_args()

def format_to_latex(acc):
    # Turn acc, which is a dict, into a string where each key-value pair is a line
    acc_string = ''
    for k, v in acc.items():
        # format the string like latex: $0.00\pm0.00$, also as percentages
        acc_string += f'{k}: ${v[0] * 100:.2f} \pm {v[1] * 100:.2f}$\n'
    return acc_string

def train(model, optimizer, database_3dssg, dataset, batch_size, fold):
    assert(type(dataset) == list)
    indices = [i for i in range(len(dataset))]
    random.shuffle(indices)
    
    if args.contrastive_loss:
        # Improved batching - avoid creating tiny batches at the end
        batched_indices = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
        # Handle last batch potentially being smaller
        if len(batched_indices[-1]) < batch_size // 2:
            batched_indices = batched_indices[:-1]
            
        skipped = 0
        total = 0
        
        for batch in batched_indices:
            batch_size_actual = len(batch)
            # Pre-allocate tensors on GPU
            loss1 = torch.zeros((batch_size_actual, batch_size_actual), device='cuda')
            loss3 = torch.zeros((batch_size_actual, batch_size_actual), device='cuda')
            
            # Pre-process and cache all batch graphs at once to reduce loop overhead
            batch_data = []
            valid_pairs = []
            
            # Preprocess batch data outside inner loops
            for i in range(batch_size_actual):
                query = dataset[batch[i]]
                db = database_3dssg[query.scene_id]
                
                if args.subgraph_ablation:
                    query_subgraph, db_subgraph = query, db
                else:
                    query_subgraph, db_subgraph = get_matching_subgraph(query, db)
                    if db_subgraph is None or len(db_subgraph.nodes) <= 1: db_subgraph = db
                    if query_subgraph is None or len(query_subgraph.nodes) <= 1: query_subgraph = query
                
                x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
                p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()
                
                # Convert to tensors once and store
                batch_data.append({
                    'x_node_ft': torch.tensor(np.array(x_node_ft), dtype=torch.float32, device='cuda'),
                    'p_node_ft': torch.tensor(np.array(p_node_ft), dtype=torch.float32, device='cuda'),
                    'x_edge_idx': torch.tensor(x_edge_idx, dtype=torch.int64, device='cuda'),
                    'p_edge_idx': torch.tensor(p_edge_idx, dtype=torch.int64, device='cuda'),
                    'x_edge_ft': torch.tensor(np.array(x_edge_ft), dtype=torch.float32, device='cuda'),
                    'p_edge_ft': torch.tensor(np.array(p_edge_ft), dtype=torch.float32, device='cuda'),
                    'valid': len(x_edge_idx[0]) >= 1 and len(p_edge_idx[0]) >= 1
                })
                
            # Process pairs with GPU parallelization where possible
            for i in range(batch_size_actual):
                for j in range(i, batch_size_actual):
                    total += 1
                    
                    # Skip invalid graphs
                    if not batch_data[i]['valid'] or not batch_data[j]['valid']:
                        skipped += 1
                        loss1[i][j] = 1
                        loss1[j][i] = 1
                        loss3[i][j] = 0.5
                        loss3[j][i] = 0.5
                        continue
                    
                    # Forward pass with pre-loaded tensors
                    x_p, p_p, m_p = model(
                        batch_data[i]['x_node_ft'], batch_data[j]['p_node_ft'],
                        batch_data[i]['x_edge_idx'], batch_data[j]['p_edge_idx'],
                        batch_data[i]['x_edge_ft'], batch_data[j]['p_edge_ft']
                    )

                    # Calculate similarity and store in pre-allocated tensor
                    loss1[i][j] = 1 - F.cosine_similarity(x_p, p_p, dim=0)
                    loss1[j][i] = loss1[i][j]
                    loss3[i][j] = m_p
                    loss3[j][i] = loss3[i][j]
            
            # Create target tensors directly on GPU
            loss1_t = (torch.ones((batch_size_actual, batch_size_actual), device='cuda') - 
                      torch.eye(batch_size_actual, device='cuda')) * 2
            loss3_t = torch.eye(batch_size_actual, device='cuda')

            # Calculate batch statistics
            avg_mp = torch.diag(loss3).mean()
            avg_mn = (torch.sum(loss3) - torch.diag(loss3).sum()) / max(1, batch_size_actual * (batch_size_actual - 1))
            avg_cos_sim_p = torch.diag(loss1).mean()
            avg_cos_sim_n = (torch.sum(loss1) - torch.diag(loss1).sum()) / max(1, batch_size_actual * (batch_size_actual - 1))
            
            # Cross entropy and total loss
            loss1 = cross_entropy(loss1, loss1_t, reduction='mean', dim=1)
            loss3 = cross_entropy(loss3, loss3_t, reduction='mean', dim=1)
            
            if args.loss_ablation_m:
                loss = loss1  # cosine similarity only
            elif args.loss_ablation_c:
                loss = loss3  # matching probability only
            else:
                loss = (loss1 + loss3) / 2.0  # average of both

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics
            wandb.log({
                f'loss1_{fold}': loss1.item(),
                f'loss3_{fold}': loss3.item(),
                f'loss_{fold}': loss.item(),
                f'avg_matching_pos_{fold}': avg_mp.item(),
                f'avg_matching_neg_{fold}': avg_mn.item(),
                f'avg_cos_sim_pos_{fold}': avg_cos_sim_p.item(),
                f'avg_cos_sim_neg_{fold}': avg_cos_sim_n.item()
            })
            
            # Clear batch data to release GPU memory
            for data in batch_data:
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = None
            batch_data = None
            torch.cuda.empty_cache()
            
        print(f'Skipped {skipped} graphs out of {total} because one of the subgraphs had too few edges')
    return model

def eval_loss(model, database_3dssg, dataset, fold):
    model.eval()
    loss1_across_batches = []
    loss3_across_batches = []
    loss_across_batches = []
    avg_mp_across_batches = []
    avg_mn_across_batches = []
    avg_cos_sim_p_across_batches = []
    avg_cos_sim_n_across_batches = []
    
    with torch.no_grad():
        assert(type(dataset) == list)
        indices = [i for i in range(len(dataset))]
        random.shuffle(indices)
        
        if args.contrastive_loss:
            # Improved batching - avoid creating tiny batches at the end
            batched_indices = [indices[i:i+args.batch_size] for i in range(0, len(indices), args.batch_size)]
            if len(batched_indices[-1]) < args.batch_size // 2:
                batched_indices = batched_indices[:-1]
                
            print(f'Number of batches in evaluation: {len(batched_indices)}')
            skipped = 0
            total = 0
            
            for batch in batched_indices:
                batch_size_actual = len(batch)
                # Pre-allocate tensors on GPU
                loss1 = torch.zeros((batch_size_actual, batch_size_actual), device='cuda')
                loss3 = torch.zeros((batch_size_actual, batch_size_actual), device='cuda')
                
                # Pre-process and cache all batch graphs at once
                batch_data = []
                
                # Preprocess batch data outside inner loops
                for i in range(batch_size_actual):
                    query = dataset[batch[i]]
                    db = database_3dssg[query.scene_id]
                    
                    if args.subgraph_ablation:
                        query_subgraph, db_subgraph = query, db
                    else:
                        query_subgraph, db_subgraph = get_matching_subgraph(query, db)
                        if db_subgraph is None or len(db_subgraph.nodes) <= 1: db_subgraph = db
                        if query_subgraph is None or len(query_subgraph.nodes) <= 1: query_subgraph = query
                    
                    x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
                    p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()
                    
                    # Convert to tensors once and store
                    batch_data.append({
                        'x_node_ft': torch.tensor(np.array(x_node_ft), dtype=torch.float32, device='cuda'),
                        'p_node_ft': torch.tensor(np.array(p_node_ft), dtype=torch.float32, device='cuda'),
                        'x_edge_idx': torch.tensor(x_edge_idx, dtype=torch.int64, device='cuda'),
                        'p_edge_idx': torch.tensor(p_edge_idx, dtype=torch.int64, device='cuda'),
                        'x_edge_ft': torch.tensor(np.array(x_edge_ft), dtype=torch.float32, device='cuda'),
                        'p_edge_ft': torch.tensor(np.array(p_edge_ft), dtype=torch.float32, device='cuda'),
                        'valid': len(x_edge_idx[0]) >= 1 and len(p_edge_idx[0]) >= 1
                    })
                
                # Process all pairs
                for i in range(batch_size_actual):
                    for j in range(i, batch_size_actual):
                        total += 1
                        
                        # Skip invalid graphs
                        if not batch_data[i]['valid'] or not batch_data[j]['valid']:
                            skipped += 1
                            loss1[i][j] = 1
                            loss1[j][i] = 1
                            loss3[i][j] = 0.5
                            loss3[j][i] = 0.5
                            continue
                        
                        # Forward pass with pre-loaded tensors
                        x_p, p_p, m_p = model(
                            batch_data[i]['x_node_ft'], batch_data[j]['p_node_ft'],
                            batch_data[i]['x_edge_idx'], batch_data[j]['p_edge_idx'],
                            batch_data[i]['x_edge_ft'], batch_data[j]['p_edge_ft']
                        )

                        # Store results
                        loss1[i][j] = 1 - F.cosine_similarity(x_p, p_p, dim=0)
                        loss1[j][i] = loss1[i][j]
                        loss3[i][j] = m_p
                        loss3[j][i] = loss3[i][j]
                
                # Target tensors
                loss1_t = (torch.ones((batch_size_actual, batch_size_actual), device='cuda') - 
                          torch.eye(batch_size_actual, device='cuda')) * 2
                loss3_t = torch.eye(batch_size_actual, device='cuda')

                # Calculate metrics
                avg_mp = torch.diag(loss3).mean()
                avg_mn = (torch.sum(loss3) - torch.diag(loss3).sum()) / max(1, batch_size_actual * (batch_size_actual - 1))
                avg_cos_sim_p = torch.diag(loss1).mean()
                avg_cos_sim_n = (torch.sum(loss1) - torch.diag(loss1).sum()) / max(1, batch_size_actual * (batch_size_actual - 1))
                
                # Cross entropy loss
                loss1 = cross_entropy(loss1, loss1_t, reduction='mean', dim=1)
                loss3 = cross_entropy(loss3, loss3_t, reduction='mean', dim=1)
                
                if args.loss_ablation_m or args.eval_only_c:
                    loss = loss1  # cosine similarity only
                elif args.loss_ablation_c:
                    loss = loss3  # matching probability only
                else:
                    loss = (loss1 + loss3) / 2.0  # average of both

                # Save batch results
                loss1_across_batches.append(loss1.item())
                loss3_across_batches.append(loss3.item())
                loss_across_batches.append(loss.item())
                avg_mp_across_batches.append(avg_mp.item())
                avg_mn_across_batches.append(avg_mn.item())
                avg_cos_sim_p_across_batches.append(avg_cos_sim_p.item())
                avg_cos_sim_n_across_batches.append(avg_cos_sim_n.item())
                
                # Clear memory
                for data in batch_data:
                    for key in data:
                        if isinstance(data[key], torch.Tensor):
                            data[key] = None
                batch_data = None
                torch.cuda.empty_cache()

            # Log metrics
            wandb.log({
                f'eval_across_batch_loss1_{fold}': np.mean(loss1_across_batches),
                f'eval_across_batch_loss3_{fold}': np.mean(loss3_across_batches),
                f'eval_across_batch_loss_{fold}': np.mean(loss_across_batches),
                f'eval_across_batch_avg_matching_pos_{fold}': np.mean(avg_mp_across_batches),
                f'eval_across_batch_avg_matching_neg_{fold}': np.mean(avg_mn_across_batches),
                f'eval_across_batch_avg_cos_sim_pos_{fold}': np.mean(avg_cos_sim_p_across_batches),
                f'eval_across_batch_avg_cos_sim_neg_{fold}': np.mean(avg_cos_sim_n_across_batches)
            })
            
            print(f'During evaluation fold {fold} skipped {skipped} graphs out of {total} because one of the subgraphs had too few edges')
            print(f'Loss across batches was {np.mean(loss_across_batches)}')
    
    model.train()
    return np.mean(loss_across_batches)

def eval_acc(model, database_3dssg, dataset, fold, mode='scanscribe', eval_iter_count=args.eval_iter_count, out_of=args.out_of, valid_top_k=[1, 2, 3, 5], timer=None):
    model.eval()

    # Make sure the dataset is properly sampled
    buckets = {}
    for idx, g in enumerate(dataset):
        if g.scene_id not in buckets: buckets[g.scene_id] = []
        buckets[g.scene_id].append(idx)

    if args.eval_entire_dataset:
        out_of = len(buckets)
        valid_top_k = [1, 5, 10, 20, 30, 40]
        if mode == 'human' or mode == 'human_test':
            valid_top_k.extend([50, 75])

    # Prepare evaluation metrics
    all_valid = {k: [] for k in valid_top_k}
    
    for _ in range(args.eval_iters):
        valid = {k: [] for k in valid_top_k}
        
        # Sample test indices efficiently
        sampled_test_indices = [
            [random.sample(buckets[g], 1)[0] for g in random.sample(list(buckets.keys()), out_of)] 
            for _ in range(eval_iter_count)
        ]
        
        for t_set in sampled_test_indices:
            # Precompute query once and reuse
            query = dataset[t_set[0]]
            query_subgraph = query  # Initial value
            
            # Pre-process query graph only once
            if not args.subgraph_ablation:
                x_node_ft_base, x_edge_idx_base, x_edge_ft_base = query.to_pyg()
                query_tensor_base = {
                    'x_node_ft': torch.tensor(np.array(x_node_ft_base), dtype=torch.float32, device='cuda'),
                    'x_edge_idx': torch.tensor(x_edge_idx_base, dtype=torch.int64, device='cuda'),
                    'x_edge_ft': torch.tensor(np.array(x_edge_ft_base), dtype=torch.float32, device='cuda'),
                }
            
            # Pre-process all database entries in one batch where possible
            batch_data = []
            scene_ids = []
            true_match_flags = []
            
            for i in t_set:
                db = database_3dssg[dataset[i].scene_id]
                scene_ids.append(db.scene_id)
                is_match = (query.scene_id == db.scene_id)
                true_match_flags.append(1 if is_match else 0)
                
                # Process query-specific subgraph only when needed
                if not args.subgraph_ablation:
                    if is_match:  # Only get matching subgraph for true match
                        query_subgraph, db_subgraph = get_matching_subgraph(query, db)
                        if db_subgraph is None or len(db_subgraph.nodes) <= 1 or len(db_subgraph.edge_idx[0]) < 1:
                            db_subgraph = db
                        if query_subgraph is None or len(query_subgraph.nodes) <= 1 or len(query_subgraph.edge_idx[0]) < 1:
                            query_subgraph = query
                            
                        # Convert to PyG format and tensors
                        x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
                        query_tensor = {
                            'x_node_ft': torch.tensor(np.array(x_node_ft), dtype=torch.float32, device='cuda'),
                            'x_edge_idx': torch.tensor(x_edge_idx, dtype=torch.int64, device='cuda'),
                            'x_edge_ft': torch.tensor(np.array(x_edge_ft), dtype=torch.float32, device='cuda'),
                        }
                    else:
                        # Reuse base query for non-matches
                        query_tensor = query_tensor_base
                else:
                    # No subgraphing, use full graph
                    query_subgraph = query
                    x_node_ft, x_edge_idx, x_edge_ft = query_subgraph.to_pyg()
                    query_tensor = {
                        'x_node_ft': torch.tensor(np.array(x_node_ft), dtype=torch.float32, device='cuda'),
                        'x_edge_idx': torch.tensor(x_edge_idx, dtype=torch.int64, device='cuda'),
                        'x_edge_ft': torch.tensor(np.array(x_edge_ft), dtype=torch.float32, device='cuda'),                        
                    }
                
                # Process database graph
                if not args.subgraph_ablation and is_match:
                    # Already computed above
                    p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()
                else:
                    if args.subgraph_ablation:
                        db_subgraph = db
                    else:
                        _, db_subgraph = get_matching_subgraph(query, db)
                        if db_subgraph is None or len(db_subgraph.nodes) <= 1 or len(db_subgraph.edge_idx[0]) < 1:
                            db_subgraph = db
                            
                    p_node_ft, p_edge_idx, p_edge_ft = db_subgraph.to_pyg()
                
                # Add processed tensors to batch
                batch_data.append({
                    'query': query_tensor,
                    'db': {
                        'p_node_ft': torch.tensor(np.array(p_node_ft), dtype=torch.float32, device='cuda'),
                        'p_edge_idx': torch.tensor(p_edge_idx, dtype=torch.int64, device='cuda'),
                        'p_edge_ft': torch.tensor(np.array(p_edge_ft), dtype=torch.float32, device='cuda'),
                    }
                })
            
            # Process entire batch efficiently
            match_prob = []
            cos_sims = []
            
            for i, data in enumerate(batch_data):
                t1 = time.time()
                x_p, p_p, m_p = model(
                    data['query']['x_node_ft'], data['db']['p_node_ft'],
                    data['query']['x_edge_idx'], data['db']['p_edge_idx'],
                    data['query']['x_edge_ft'], data['db']['p_edge_ft']
                )
                
                if timer is not None:
                    timer.text2graph_text_embedding_matching_score_time.append(time.time() - t1)
                    timer.text2graph_text_embedding_matching_score_iter.append(1)

                cos_sims.append((1 - F.cosine_similarity(x_p, p_p, dim=0)).item())
                match_prob.append(m_p.item())
            
            # Scoring and ranking
            t1 = time.time()
            if args.loss_ablation_m or args.eval_only_c:
                # Use cosine similarity (lower is better)
                scores = np.array(cos_sims)
                sorted_indices = np.argsort(scores)
            else:
                # Use matching probability (higher is better)
                scores = np.array(match_prob)
                sorted_indices = np.argsort(scores)[::-1]
                
            if timer is not None:
                timer.text2graph_matching_time.append(time.time() - t1)
                timer.text2graph_matching_iter.append(1)
            
            true_match = np.array(true_match_flags)
            scene_ids = [scene_ids[i] for i in sorted_indices]
            true_match = true_match[sorted_indices]
            
            # Calculate accuracy metrics
            for k in valid_top_k:
                if 1 in true_match[:k]:
                    valid[k].append(1)
                else:
                    valid[k].append(0)
            
            # Clean up GPU memory
            for data in batch_data:
                for key in data['query']:
                    if isinstance(data['query'][key], torch.Tensor):
                        data['query'][key] = None
                for key in data['db']:
                    if isinstance(data['db'][key], torch.Tensor):
                        data['db'][key] = None
            batch_data = None
            torch.cuda.empty_cache()
        
        # Aggregate metrics
        for k in valid_top_k:
            all_valid[k].append(np.mean(valid[k]))

    # Calculate final metrics
    accuracy = {k: (np.mean(all_valid[k]), np.std(all_valid[k])) for k in valid_top_k}
    
    # Log metrics
    if fold is not None:
        for k in accuracy:
            wandb.log({f'accuracy_{str(mode)}_top_{k}_fold_{fold}': accuracy[k]})
    else:
        for k in accuracy:
            wandb.log({f'accuracy_{str(mode)}_top_{k}': accuracy[k]})
            
    print(f'accuracies: {accuracy}')
    model.train()
    
    return accuracy


def train_with_cross_val(dataset, database_3dssg, model, folds, epochs, batch_size, entire_training_set):
    if entire_training_set:
        if args.continue_training:
            model = BigGNN(args.N, args.heads).to('cuda')
            model_dict = torch.load(f'../model_checkpoints/graph2graph/{args.continue_training_model}.pt')
            model.load_state_dict(model_dict)
        else: model = BigGNN(args.N, args.heads).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        starting_epoch = 1
        if (args.continue_training): 
            starting_epoch = args.continue_training
        epochs = epochs + starting_epoch
        for epoch in tqdm(range(starting_epoch, epochs)):
            _ = train(model=model, 
                               optimizer=optimizer, 
                               database_3dssg=database_3dssg, 
                               dataset=dataset, 
                               batch_size=batch_size, 
                               fold=None)
            if epoch % 2 == 0:
                torch.save(model.state_dict(), f'../model_checkpoints/graph2graph/{args.model_name}_epoch_{epoch}_checkpoint.pt')
        return model
    
    # else we do k-fold, or with 1 fold and validation set
    # assert(type(dataset) == list)
    val_losses, accs, durations = [], [], []
    for fold, (train_idx, val_idx) in enumerate(k_fold_by_scene(dataset, folds)):
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]

        print(f'length of training set in fold {fold}: {len(train_dataset)}')
        print(f'length of validation set in fold {fold}: {len(val_dataset)}')
        
        if args.continue_training:
            model = BigGNN(args.N, args.heads).to('cuda')
            model_dict = torch.load(f'../model_checkpoints/graph2graph/{args.continue_training_model}.pt')
            model.load_state_dict(model_dict)
        else: model = BigGNN(args.N, args.heads).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # if torch.cuda.is_available(): torch.cuda.synchronize()
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     try:
        #         import torch.mps
        #         torch.mps.synchronize()
        #     except ImportError: pass

        # t_start = time.perf_counter()
        starting_epoch = 1
        if (args.continue_training): 
            starting_epoch = args.continue_training
        epochs = epochs + starting_epoch
        for epoch in tqdm(range(starting_epoch, epochs)):
            _ = train(model=model, 
                               optimizer=optimizer, 
                               database_3dssg=database_3dssg, 
                               dataset=train_dataset, 
                               batch_size=batch_size, 
                               fold=fold)
            if epoch % 2 == 0:
                torch.save(model.state_dict(), f'../model_checkpoints/graph2graph/{args.model_name}_epoch_{epoch}_checkpoint.pt')
            val_losses.append(eval_loss(model=model, 
                                        database_3dssg=database_3dssg, 
                                        dataset=val_dataset,
                                        fold=fold))
            accs.append(eval_acc(model=model,
                                 database_3dssg=database_3dssg, 
                                 dataset=val_dataset,
                                 fold=fold,
                                 eval_iter_count=30))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': _,
                'val_loss': val_losses[-1],
                'val_acc_from_train': accs[-1],
            }
            print(f'Evaluation information: {eval_info}')

            # if epoch % lr_decay_step_size == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_decay_factor * param_group['lr']

        # if torch.cuda.is_available(): torch.cuda.synchronize()
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.synchronize()

        # t_end = time.perf_counter()
        # durations.append(t_end - t_start)
        if (args.skip_k_fold): break # only use the first fold to speed up training, but we still see a validation

    # loss, acc, duration = torch.tensor(val_losses), torch.tensor(accs), torch.tensor(durations)
    # loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    # loss, argmin = loss.min(dim=1)
    # acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    # loss_mean = loss.mean().item()
    # acc_mean = acc.mean().item()
    # acc_std = acc.std().item()
    # duration_mean = duration.mean().item()
    # print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
    #       f'± {acc_std:.3f}, Duration: {duration_mean:.3f}')

    return model#, loss_mean, acc_mean, acc_std

###################################### OLD ######################################

def train_without_val(_3dssg_graphs, scanscribe_graphs):
    model = BigGNN(args.N, args.heads).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    current_keys = list(scanscribe_graphs.keys())
    # assert(all([len(scanscribe_graphs[g].nodes) >= args.graph_size_min for g in scanscribe_graphs]))

    # batched contrastive Loss
    if (args.contrastive_loss):
        for epoch in tqdm(range(args.epoch)):
            random.shuffle(current_keys)
            current_keys_batched = [current_keys[i:i+args.batch_size] for i in range(0, len(current_keys) - args.batch_size, args.batch_size)]
            # print(f'len(current_keys): {len(current_keys_batched)}, num batches {int(len(current_keys) / args.batch_size)}')
            # assert(len(current_keys_batched) == int(len(current_keys) / args.batch_size)) # TODO: Check the indexing is okay here, but for now should be fine we just skip a few graphs
            assert(len(current_keys_batched[0]) == args.batch_size)
            skipped = 0
            total = 0
            for batch in current_keys_batched:
                loss1 = torch.zeros((len(batch), len(batch))).to('cuda')
                loss3 = torch.zeros((len(batch), len(batch))).to('cuda')
                for i in range(len(batch)):
                    for j in range(i, len(batch)):
                        total += 1
                        scribe_g = scanscribe_graphs[batch[i]]
                        _3dssg_g = _3dssg_graphs[batch[j].split('_')[0]]
                        scribe_g_subgraph, _3dssg_g_subgraph = get_matching_subgraph(scribe_g, _3dssg_g)
                        if _3dssg_g_subgraph is None or len(_3dssg_g_subgraph.nodes) <= 1: _3dssg_g_subgraph = _3dssg_g
                        if scribe_g_subgraph is None or len(scribe_g_subgraph.nodes) <= 1: scribe_g_subgraph = scribe_g # TODO: why is scribe g None now?

                        x_node_ft, x_edge_idx, x_edge_ft = scribe_g_subgraph.to_pyg()
                        p_node_ft, p_edge_idx, p_edge_ft = _3dssg_g_subgraph.to_pyg()
                        if len(x_edge_idx[0]) <= 2 or len(p_edge_idx[0]) <= 2: 
                            skipped += 1
                            loss1[i][j] = 1
                            loss1[j][i] = loss1[i][j]
                            loss3[i][j] = 0.5
                            loss3[j][i] = loss3[i][j]
                            continue
                        x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                                torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                                torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
                        # remove from cuda to free space
                        x_node_ft, x_edge_idx, x_edge_ft = None, None, None
                        loss1[i][j] = 1 - F.cosine_similarity(x_p, p_p, dim=0) # [0, 2] 0 is good
                        loss1[j][i] = loss1[i][j]
                        loss3[i][j] = m_p
                        loss3[j][i] = loss3[i][j]
                loss1_t = (torch.ones((len(batch), len(batch))).to('cuda') - torch.eye(len(batch)).to('cuda')) * 2
                loss3_t = torch.eye(len(batch)).to('cuda')

                # Average m_p across diagonal
                avg_mp = torch.diag(loss3).mean()
                avg_mn = (torch.sum(loss3) - torch.diag(loss3).sum()) / (len(batch) * (len(batch) - 1))
                avg_cos_sim_p = torch.diag(loss1).mean()
                avg_cos_sim_n = (torch.sum(loss1) - torch.diag(loss1).sum()) / (len(batch) * (len(batch) - 1))
                # Cross entropy
                loss1 = cross_entropy(loss1, loss1_t, reduction='mean', dim=1)
                loss3 = cross_entropy(loss3, loss3_t, reduction='mean', dim=1)
                loss = (loss1 + loss3) / 2.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({"loss1": loss1.item(),
                            "loss3": loss3.item(),
                            "loss": loss.item(),
                            "avg_matching_pos": avg_mp.item(),
                            "avg_matching_neg": avg_mn.item(),
                            "avg_cos_sim_pos": avg_cos_sim_p.item(),
                            "avg_cos_sim_neg": avg_cos_sim_n.item()})
                
            wandb.log({"loss_per_epoch": loss.item()})
            if epoch % 2 == 0:
                # evaluate_model(model, scanscribe_graphs_test, _3dssg_graphs, 'test')
                evaluate_model(model, human_graphs_test, _3dssg_graphs, 'test_human')
                print(f'x_p first 10: {x_p[:10]}')
                print(f'p_p first 10: {p_p[:10]}')
            print(f'Skipped {skipped} graphs out of {total} because one of the subgraphs had too few edges')
        return model
    else: 
        batch_size = args.batch_size
        for epoch in tqdm(range(args.epoch)):
            curr_batch = 0
            loss = 0
            skipped = 0

            for scribe_id in scanscribe_graphs:
                scribe_g = scanscribe_graphs[scribe_id]
                _3dssg_g = _3dssg_graphs[scribe_id.split('_')[0]]

                _3dssg_g_n = _3dssg_graphs[np.random.choice([k.split('_')[0] for k in current_keys if k.split('_')[0] != scribe_id.split('_')[0]])]
                scribe_g_subgraph_n, _3dssg_g_subgraph_n = get_matching_subgraph(scribe_g, _3dssg_g_n)
                if scribe_g_subgraph_n is None or len(scribe_g_subgraph_n.nodes) <= 1: scribe_g_subgraph_n = scribe_g
                if _3dssg_g_subgraph_n is None or len(_3dssg_g_subgraph_n.nodes) <= 1: _3dssg_g_subgraph_n = _3dssg_g_n

                scribe_g_subgraph, _3dssg_g_subgraph = get_matching_subgraph(scribe_g, _3dssg_g) # TODO: 3) check what the graph neural network is doing
                if _3dssg_g_subgraph is None or len(_3dssg_g_subgraph.nodes) <= 1: _3dssg_g_subgraph = _3dssg_g
                if scribe_g_subgraph is None or len(scribe_g_subgraph.nodes) <= 1: scribe_g_subgraph = scribe_g
                # x = torch.tensor([scribe_g.nodes[i].features for i in scribe_g.nodes]).to('cuda') # TODO: Why is x not the same as x_node_ft?
                # p = torch.tensor([_3dssg_g.nodes[i].features for i in _3dssg_g.nodes]).to('cuda')

                x_node_ft, x_edge_idx, x_edge_ft = scribe_g.to_pyg() # scribe_g.to_pyg()
                xn_node_ft, xn_edge_idx, xn_edge_ft = scribe_g_subgraph_n.to_pyg() # TODO: change this so that model gets an equal chance with a subgraphed scribe negative example
                p_node_ft, p_edge_idx, p_edge_ft = _3dssg_g_subgraph.to_pyg() # _3dssg_g.to_pyg()
                n_node_ft, n_edge_idx, n_edge_ft = _3dssg_g_subgraph_n.to_pyg()
                if len(x_edge_idx[0]) <= 2 or len(p_edge_idx[0]) <= 2 or len(n_edge_idx[0]) <= 2: 
                    skipped += 1
                    continue

                x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                        torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                        torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
                x_n, n_n, m_n = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(n_node_ft), dtype=torch.float32).to('cuda'),
                                        torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(n_edge_idx, dtype=torch.int64).to('cuda'),
                                        torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(n_edge_ft), dtype=torch.float32).to('cuda'))
                
                curr_batch += 1

                loss1 = 1 - F.cosine_similarity(x_p, p_p, dim=0) # [0, 2] 0 is good
                loss2 = 2 - (1 - F.cosine_similarity(x_n, n_n, dim=0)) # [0, 2] 2 is good
                loss3 = (1 - m_p) + m_n

                loss += loss1 + loss2 + loss3
                
                if (curr_batch % batch_size == 0):
                    optimizer.zero_grad()
                    loss = loss / batch_size
                    epoch_loss = loss
                    loss.backward()
                    optimizer.step()
                    wandb.log({"loss1": loss1.item(),
                                "loss2": loss2.item(),
                                "loss3": loss3.sum().item(),
                                "loss": loss.item(),
                                "match_prob_pos": m_p.item(),
                                "match_prob_neg": m_n.item()})
                    loss = 0
                    curr_batch = 0

            wandb.log({"loss_per_epoch": epoch_loss.item()})
            if epoch % 2 == 0:
                evaluate_model(model, scanscribe_graphs_test, _3dssg_graphs, 'test')
                print(f'x_p first 10: {x_p[:10]}')
                print(f'x_n first 10: {x_n[:10]}')
                print(f'p_p first 10: {p_p[:10]}')
                print(f'n_n first 10: {n_n[:10]}')
            print(f'Skipped {skipped} graphs because one of the subgraphs had too few edges')
        return model

def evaluate_model(model, scanscribe, _3dssg, mode='test'):
    model.eval()
    valid_top_k = args.valid_top_k
    valid = {k: [] for k in valid_top_k}

    _3dssg = {k.split('_')[0]: _3dssg[k.split('_')[0]] for k in scanscribe}
    with torch.no_grad():
        for scribe_id in scanscribe:
            match_prob = []
            true_match = []
            scribe_g = scanscribe[scribe_id]
            for _3dssg_id in _3dssg:
                _3dssg_g = _3dssg[_3dssg_id]
                scribe_g_subgraph, _3dssg_g_subgraph = get_matching_subgraph(scribe_g, _3dssg_g)
                if _3dssg_g_subgraph is None or len(_3dssg_g_subgraph.nodes) <= 1: _3dssg_g_subgraph = _3dssg_g
                x_node_ft, x_edge_idx, x_edge_ft = scribe_g.to_pyg()
                p_node_ft, p_edge_idx, p_edge_ft = _3dssg_g_subgraph.to_pyg()
                x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                        torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                        torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
                match_prob.append(m_p.item())
                if (scribe_id.split('_')[0] == _3dssg_id): true_match.append(1)
                else: true_match.append(0)
            
            # sort w indices
            match_prob = np.array(match_prob)
            true_match = np.array(true_match)
            sorted_indices = np.argsort(match_prob)
            match_prob = match_prob[sorted_indices]
            true_match = true_match[sorted_indices]
            print(f'match_prob: {match_prob}')
            print(f'true_match: {true_match}')
            for k in valid_top_k:
                if (1 in true_match[-k:]): valid[k].append(1)
                else: valid[k].append(0)

    accuracy = {k: np.mean(valid[k]) for k in valid_top_k}
    for k in accuracy: wandb.log({f'accuracy_{str(mode)}_top{k}': accuracy[k]})
    print(f'accuracies: {accuracy}')
    model.train()

if __name__ == '__main__':
    # In[0]: argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, default='online')
    # parser.add_argument('--epoch', type=int, default=10)
    # parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--weight_decay', type=float, default=5e-5)
    # parser.add_argument('--N', type=int, default=1)
    # parser.add_argument('--overlap_thr', type=float, default=0.8)
    # parser.add_argument('--cos_sim_thr', type=float, default=0.5)
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--training_set_size', type=int, default=2847)
    # parser.add_argument('--test_set_size', type=int, default=712)
    # parser.add_argument('--graph_size_min', type=int, default=4, help='minimum number of nodes in a graph')
    # parser.add_argument('--contrastive_loss', type=bool, default=True)
    # parser.add_argument('--valid_top_k', nargs='+', type=int, default=[1, 2, 3, 5])
    # parser.add_argument('--use_attributes', type=bool, default=True)
    # parser.add_argument('--training_with_cross_val', type=bool, default=True)
    # parser.add_argument('--folds', type=int, default=5)
    # parser.add_argument('--skip_k_fold', type=bool, default=False)
    # parser.add_argument('--entire_training_set', action='store_true')
    # parser.add_argument('--subgraph_ablation', action='store_true')
    # parser.add_argument('--eval_iters', type=int, default=100)
    # parser.add_argument('--model_name', type=str, default=None)
    # parser.add_argument('--loss_ablation_m', action='store_true')
    # parser.add_argument('--loss_ablation_c', action='store_true')
    # args = parser.parse_args()

    if (args.model_name is None):
        print("Must define a model name")
        print("Exiting...")
        exit()
    # make sure only 1 out of 2 loss ablations is true
    if (args.loss_ablation_m and args.loss_ablation_c):
        print("Can only have one loss ablation true at a time")
        print("Exiting...")
        exit()
    # In[1]
    wandb.config = { "architecture": "self attention cross attention",
                     "dataset": "ScanScribe_cleaned"} # ScanScribe_1 is the cleaned dataset with ada_002 embeddings
    for arg in vars(args): wandb.config[arg] = getattr(args, arg)
    wandb.init(project="graph2graph",
                mode=args.mode,
                config=wandb.config)

    # _3dssg_graphs = torch.load('../data_checkpoints/processed_data/training/3dssg_graphs_train_graph_min_size_4.pt')                # Len 1323   
    _3dssg_graphs = {}
    _3dssg_scenes = torch.load('../data_checkpoints/processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt')
    for sceneid in tqdm(_3dssg_scenes):
        _3dssg_graphs[sceneid] = SceneGraph(sceneid, 
                                            graph_type='3dssg', 
                                            graph=_3dssg_scenes[sceneid], 
                                            max_dist=1.0, embedding_type='word2vec',
                                            use_attributes=args.use_attributes)


    # scanscribe_graphs = torch.load('../data_checkpoints/processed_data/training/scanscribe_graphs_train_graph_min_size_4.pt')       # 80% split len 2847
    scanscribe_graphs = {}
    scanscribe_scenes = torch.load('../data_checkpoints/processed_data/training/scanscribe_graphs_train_final_no_graph_min.pt')
    for scene_id in tqdm(scanscribe_scenes):
        txtids = scanscribe_scenes[scene_id].keys()
        assert(len(set(txtids)) == len(txtids)) # no duplicate txtids
        assert(len(set(txtids)) == len(range(max([int(id) for id in txtids]) + 1))) # no missing txtids
        for txt_id in txtids:
            txt_id_padded = str(txt_id).zfill(5)
            scanscribe_graphs[scene_id + '_' + txt_id_padded] = SceneGraph(scene_id,
                                                                        txt_id=txt_id,
                                                                        graph_type='scanscribe', 
                                                                        graph=scanscribe_scenes[scene_id][txt_id], 
                                                                        embedding_type='word2vec',
                                                                        use_attributes=args.use_attributes)

    # preprocess so that the graphs all at least 1 edge
    print(f'number of scanscribe graphs before removing graphs with 1 edge: {len(scanscribe_graphs)}')
    to_remove = []
    for g in scanscribe_graphs:
        if len(scanscribe_graphs[g].edge_idx[0]) <= 1: # TODO: turn into strict inequality
            to_remove.append(g)
    for g in to_remove: del scanscribe_graphs[g]
    print(f'number of scanscribe graphs after removing graphs with 1 edge: {len(scanscribe_graphs)}')
    scanscribe_graphs = list(scanscribe_graphs.values()) # NOTE
    args.training_set_size = len(scanscribe_graphs)

    # scanscribe_graphs_test = torch.load('../data_checkpoints/processed_data/testing/scanscribe_graphs_test_graph_min_size_4.pt')    # 20% split len 712
    scanscribe_graphs_test = {}
    scanscribe_scenes_test = torch.load('../data_checkpoints/processed_data/testing/scanscribe_graphs_test_final_no_graph_min.pt')
    for scene_id in tqdm(scanscribe_scenes_test):
        txtids = scanscribe_scenes_test[scene_id].keys()
        assert(len(set(txtids)) == len(txtids)) # no duplicate txtids
        assert(len(set(txtids)) == len(range(max([int(id) for id in txtids]) + 1))) # no missing txtids
        for txt_id in txtids:
            txt_id_padded = str(txt_id).zfill(5)
            scanscribe_graphs_test[scene_id + '_' + txt_id_padded] = SceneGraph(scene_id,
                                                                        txt_id=txt_id,
                                                                        graph_type='scanscribe', 
                                                                        graph=scanscribe_scenes_test[scene_id][txt_id], 
                                                                        embedding_type='word2vec',
                                                                        use_attributes=args.use_attributes)
    
    print(f'number of scanscribe test graphs before removing: {len(scanscribe_graphs_test)}')
    to_remove = []
    for g in scanscribe_graphs_test:
        if len(scanscribe_graphs_test[g].edge_idx[0]) < 1:
            to_remove.append(g)
    for g in to_remove: del scanscribe_graphs_test[g]
    print(f'number of scanscribe test graphs after removing: {len(scanscribe_graphs_test)}')
    args.test_set_size = len(scanscribe_graphs_test)

    # human_graphs_test = torch.load('../data_checkpoints/processed_data/testing/human_graphs_test_graph_min_size_4.pt')              # Len 35
    h_graphs_test = torch.load('../data_checkpoints/processed_data/human/human_graphs_processed.pt')
    h_graphs_remove = [k for k in h_graphs_test if k.split('_')[0] not in _3dssg_graphs]
    print(f'to remove human_graphs, hopefully none: {h_graphs_remove}')
    for k in h_graphs_remove: del h_graphs_test[k]
    assert(all([k.split('_')[0] in _3dssg_graphs for k in h_graphs_test]))
    human_graphs_test = {k: SceneGraph(k.split('_')[0], 
                                   graph_type='human',
                                   graph=h_graphs_test[k],
                                   embedding_type='word2vec',
                                   use_attributes=args.use_attributes) for k in h_graphs_test}



    ###################### MEMORY SIZE ANALYSIS ######################
    b_n = 0
    b_e = 0
    b_f = 0
    b_n_h = 0
    b_e_h = 0
    b_f_h = 0
    scanscribe_graphs_list_of_ids = [a.split('_')[0] for a in list(scanscribe_graphs_test.keys())]
    human_graphs_list_of_ids = [a.split('_')[0] for a in list(human_graphs_test.keys())]
    assert(all([a in _3dssg_graphs for a in scanscribe_graphs_list_of_ids]))
    assert(all([a in _3dssg_graphs for a in human_graphs_list_of_ids]))

    # analyze the graph memory size
    for g in _3dssg_graphs:
        if g in scanscribe_graphs_list_of_ids:
            graph = _3dssg_graphs[g]
            # bytes for self.nodes
            for n in graph.nodes:
                n = graph.nodes[n]
                b_n += np.array(n.features).size * np.array(n.features).itemsize

            # bytes for self.edge_idx
            b_e += np.array(graph.edge_idx).size * np.array(graph.edge_idx).itemsize

            # bytes for self.edge_features
            b_f += np.array(graph.edge_features).size * np.array(graph.edge_features).itemsize
        if g in human_graphs_list_of_ids:
            graph = _3dssg_graphs[g]
            # bytes for self.nodes
            for n in graph.nodes:
                n = graph.nodes[n]
                b_n_h += np.array(n.features).size * np.array(n.features).itemsize

            # bytes for self.edge_idx
            b_e_h += np.array(graph.edge_idx).size * np.array(graph.edge_idx).itemsize

            # bytes for self.edge_features
            b_f_h += np.array(graph.edge_features).size * np.array(graph.edge_features).itemsize

    print(f'SCANSCRIBE b_n: {b_n}, b_e: {b_e}, b_f: {b_f}, total: {b_n + b_e + b_f}')
    print(f'HUMAN b_n_h: {b_n_h}, b_e_h: {b_e_h}, b_f_h: {b_f_h}, total: {b_n_h + b_e_h + b_f_h}')
    # exit()# Why EXIT?





    if args.training_with_cross_val:
        if args.continue_training: 
            model = BigGNN(args.N, args.heads).to('cuda')
            model_dict = torch.load(f'../model_checkpoints/graph2graph/{args.continue_training_model}.pt')
            model.load_state_dict(model_dict)
        else: model = BigGNN(args.N, args.heads).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = train_with_cross_val(database_3dssg=_3dssg_graphs, 
                                        dataset=scanscribe_graphs,
                                        model=model,
                                        folds=args.folds,
                                        epochs=args.epoch,
                                        batch_size=args.batch_size,
                                        entire_training_set=args.entire_training_set)
    
    ######### SAVE SOME THINGS #########
    model_name = args.model_name
    args_str = ''
    for arg in vars(args): args_str += f'\n{arg}_{getattr(args, arg)}'
    with open(f'../model_checkpoints/graph2graph/{model_name}_args.txt', 'w') as f: f.write(args_str)
    torch.save(model.state_dict(), f'../model_checkpoints/graph2graph/{model_name}.pt')
    ####################################

    # model = BigGNN(args.N, args.heads).to('cuda')
    # model.load_state_dict(torch.load('../model_checkpoints/graph2graph/model_100epochs.pt'))

    t_start = time.perf_counter()
    # Final test sets evaluation
    scanscribe_test_accuracy = eval_acc(model=model,
                                     database_3dssg=_3dssg_graphs,
                                     dataset=list(scanscribe_graphs_test.values()),
                                     fold=None,
                                     mode='scanscribe_test')
    human_test_accuracy = eval_acc(model=model,
                                     database_3dssg=_3dssg_graphs,
                                     dataset=list(human_graphs_test.values()),
                                     fold=None,
                                     mode='human_test')
    t_end = time.perf_counter()
    print(f'Time elapsed in minutes: {(t_end - t_start) / 60}')
    
    print(f'Final test set accuracies: scanscribe {scanscribe_test_accuracy}, human {human_test_accuracy}')
