import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from our_model import MMoE 
from our_model import get_cancer_datasets  
from LRP_utils import LRP  
from utils import set_seed

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dict = get_cancer_datasets()
data = data_dict["dataset"]

in_dim = [19199, 335886, 640, 24776]
bottleneck_dim = 64 
emb_dim = 32
task_output_sizes = [3,1,1]
num_shared_experts = 4
num_specific_experts = 2
num_blocks=8
hidden_dim=8
Dropout = 0
task_type_map = {
    "multiclass": 0,
    "survival": 1,
    "regression": 2
}

task_types = ["multiclass","survival","regression"]
task_feat_nums = [len(task_output_sizes),3]


data = data_dict["dataset"]

data_type = data.labels[1].cpu().numpy()
n_splits = 5
kf = StratifiedShuffleSplit(n_splits=n_splits, random_state=42)

lrp_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(data, data_type)):
    print(f"Processing fold {fold+1}")
    
    model = MMoE(in_dim, bottleneck_dim, emb_dim=emb_dim, num_shared_experts=num_shared_experts,
                 num_specific_experts=num_specific_experts, task_output_sizes=task_output_sizes, 
                 num_blocks=num_blocks, hidden_dim=hidden_dim, task_feat_nums=task_feat_nums,
                 Dropout=Dropout).to(device)
    model.load_state_dict(torch.load(f'./saved/best_model_fold_{fold}.pt'))
    model.eval()
    
    lrp = LRP(model)
    
    val_sampler = SubsetRandomSampler(val_idx)
    val_loader = DataLoader(data, batch_size=9999, sampler=val_sampler) 
    
    for task_inputs1, task_inputs2, task_inputs3, task_inputs4, label in val_loader:
        task_inputs = [task_inputs1.to(device), task_inputs2.to(device), 
                       task_inputs3.to(device), task_inputs4.to(device)]
        
        domain_feat = label[-1].unsqueeze(1).to(device)
        
        task_feats = [torch.tensor(task_type_map[i], device=device).repeat(task_inputs1.size(0)) 
                      for i in task_types]
        
        for task_id in range(len(task_types)):
            relevance = lrp.explain((task_inputs,domain_feat, task_feats), task_id)
            for i in range(len(relevance[0])):  
                lrp_results.append({
                    'fold': fold,
                    'sample_id': val_idx[i],
                    'task_id': task_id,
                    'relevance': [r[i] for r in relevance]  
                })



def get_feature_names(cancer_dataset_dict):
    return [
        cancer_dataset_dict["rnaseq_feature_name"],
        cancer_dataset_dict["methy_feature_name"],
        cancer_dataset_dict["mirna_feature_name"],
        cancer_dataset_dict["scnv_feature_name"]
    ]

def visualize_lrp(lrp_results, cancer_dataset_dict, num_samples=5, top_k=20):
    feature_names = get_feature_names(cancer_dataset_dict)
    num_omics = len(lrp_results[0]['relevance'])
    num_tasks = len(task_types)
    fig, axes = plt.subplots(num_samples, num_omics * num_tasks, 
                             figsize=(7 * num_omics * num_tasks, 5 * num_samples))
    
    for i in range(num_samples):
        sample = np.random.choice(lrp_results)
        for task in range(num_tasks):
            for j in range(num_omics):
                relevance = sample['relevance'][j].detach().cpu().numpy()
                ax = axes[i, task * num_omics + j]
                
                top_indices = np.argsort(np.abs(relevance))[-top_k:]
                top_relevance = relevance[top_indices]
                top_feature_names = [feature_names[j][idx] for idx in top_indices]
                
                bars = ax.bar(range(top_k), top_relevance, align='center')
                
                for bar, value in zip(bars, top_relevance):
                    bar.set_color('red' if value > 0 else 'blue')
                
                ax.set_title(f"Fold {sample['fold']}, Task {task}, Omics {j+1}")
                ax.set_xlabel('Feature name')
                ax.set_ylabel('Relevance')
                ax.set_xticks(range(top_k))
                ax.set_xticklabels(top_feature_names, rotation=90, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('lrp_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()



def analyze_lrp(lrp_results, cancer_dataset_dict):
    feature_names = get_feature_names(cancer_dataset_dict)
    all_feature_importance = []

    for task_id in range(len(task_types)):
        task_relevance = [result['relevance'] for result in lrp_results if result['task_id'] == task_id]
        avg_relevance = [torch.stack([r[i].detach() for r in task_relevance]).mean(dim=0) for i in range(len(task_relevance[0]))]
        
        print(f"Analysis for Task: {task_types[task_id]}")
        
        task_feature_importance = []
        
        for i, relevance in enumerate(avg_relevance):
            relevance = relevance.cpu().numpy()
            
            sorted_indices = np.argsort(np.abs(relevance))[::-1]
            
            for idx in sorted_indices:
                feature_name = feature_names[i][idx]
                importance = relevance[idx]
                task_feature_importance.append({
                    'Task': task_types[task_id],
                    'Omics': f'Omics_{i+1}',
                    'Feature': feature_name,
                    'Importance': importance,
                    'Abs_Importance': abs(importance)
                })
            
            top_positive = sorted_indices[relevance[sorted_indices] > 0][:5]
            top_negative = sorted_indices[relevance[sorted_indices] < 0][-5:][::-1]
            
            print(f"  Top 5 positive contributing features for Omics {i+1}:")
            for j, idx in enumerate(top_positive):
                feature_name = feature_names[i][idx]
                print(f"    {j+1}. Feature {feature_name}: {relevance[idx]:.4f}")
            
            print(f"  Top 5 negative contributing features for Omics {i+1}:")
            for j, idx in enumerate(top_negative):
                feature_name = feature_names[i][idx]
                print(f"    {j+1}. Feature {feature_name}: {relevance[idx]:.4f}")
            
            print()

        all_feature_importance.extend(task_feature_importance)

        total_relevance = np.concatenate([rel.cpu().numpy() for rel in avg_relevance])
        print(f"Overall statistics for Task {task_types[task_id]}:")
        print(f"  Mean relevance: {np.mean(total_relevance):.4f}")
        print(f"  Median relevance: {np.median(total_relevance):.4f}")
        print(f"  Std relevance: {np.std(total_relevance):.4f}")
        print(f"  % of positive relevance: {(total_relevance > 0).mean() * 100:.2f}%")
        print()

    df = pd.DataFrame(all_feature_importance)
    df = df.sort_values(['Task', 'Omics', 'Abs_Importance'], ascending=[True, True, False])
    df.to_csv('feature_importance.csv', index=False)
    print("Feature importance has been saved to 'feature_importance.csv'")

cancer_dataset_dict = get_cancer_datasets()

visualize_lrp(lrp_results, cancer_dataset_dict)
analyze_lrp(lrp_results, cancer_dataset_dict)
