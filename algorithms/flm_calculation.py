import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
import torch.nn.functional as F

def get_dist_matrix(fx, y):
    # Get the predicted class labels
    predicted_labels = torch.argmax(fx, dim=1)
    
    # Convert to one-hot encoding
    fx_one_hot = one_hot(predicted_labels, num_classes=fx.size(1))
    
    # Ensure the dimensions match the original fx tensor
    fx = fx_one_hot.permute(0, 3, 1, 2).float()
    
    # Resize y to match the spatial resolution of fx
    y = F.interpolate(y.unsqueeze(1).float(), size=(fx.size(2), fx.size(3)), mode='nearest').squeeze(1).long()
    
    num_target_classes = fx.size(1)
    dist_matrix = torch.zeros((num_target_classes, y.max().item() + 1), device=fx.device)
    
    for i in range(num_target_classes):
        mask = (predicted_labels == i)
        for j in y.unique():
            dist_matrix[i, j] = (mask & (y == j)).float().sum()
    
    return dist_matrix

def predictive_distribution_based_multi_label_mapping(dist_matrix, mlm_num: int):
    assert mlm_num * dist_matrix.size(0) <= dist_matrix.size(1), "source label number not enough for mapping"
    mapping_matrix = torch.zeros_like(dist_matrix, dtype=int)
    dist_matrix_flat = dist_matrix.flatten() # same memory
    for _ in range(mlm_num * dist_matrix.size(0)):
        loc = dist_matrix_flat.argmax().item()
        loc = [loc // dist_matrix.size(1), loc % dist_matrix.size(1)]
        mapping_matrix[loc[0], loc[1]] = 1
        dist_matrix[:, loc[1]] = -1
        if mapping_matrix[loc[0]].sum() == mlm_num:
            dist_matrix[loc[0]] = -1
    return mapping_matrix

def generate_label_mapping_by_frequency(visual_prompt, network, data_loader, mapping_num = 1):
    device = next(visual_prompt.parameters()).device
    if hasattr(network, "eval"):
        network.eval()
    fx0s = []
    ys = []
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Frequency Label Mapping", ncols=100) if len(data_loader) > 20 else data_loader
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            fx0 = network(visual_prompt(x))
        fx0s.append(fx0)
        ys.append(y)
    fx0s = torch.cat(fx0s).cpu().float()
    ys = torch.cat(ys).cpu().int()
    if ys.size(0) != fx0s.size(0):
        assert fx0s.size(0) % ys.size(0) == 0
        ys = ys.repeat(int(fx0s.size(0) / ys.size(0)))
    dist_matrix = get_dist_matrix(fx0s, ys)
    pairs = torch.nonzero(predictive_distribution_based_multi_label_mapping(dist_matrix, mapping_num))
    mapping_sequence = pairs[:, 0][torch.sort(pairs[:, 1]).indices.tolist()]
    
    return mapping_sequence.long()
