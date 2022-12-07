import torch
import torch.nn.functional as F
import numpy as np
import bisect
from kmeans_pytorch import kmeans


def contrast_loss_clust(logits_w, logits_s, feats_ulb_w, n_clusts,
                        n_feats=64, temperature=1.0, clust_cutoff=0.8, device_id=0):
    logits_softmax = torch.softmax(logits_w.detach(), dim=-1)

    # feats_ulb_w = feats_ulb_w.detach().cpu().numpy()

    n_clusts = int(n_clusts)
    #     print(logits_softmax.shape, feats_ulb_w.shape, feats_ulb_w.dtype, n_clusts)

    # 1) get only selected logits and feats
    max_probs, max_idx = torch.max(logits_softmax, dim=-1)
    mask_bool = max_probs.ge(clust_cutoff)  # .cpu().numpy()
    del max_probs, max_idx

    # 2) 1. kmeans
    new_pseudo, _ = kmeans(X=feats_ulb_w, num_clusters=n_clusts)
    pseudo_onehot = torch.nn.functional.one_hot(new_pseudo, num_classes=n_clusts).float()
    pseudo_onehot = pseudo_onehot.cuda()

    # 3) get distribution for each cluster
    clust_dist = []
    for i in range(n_clusts):
        mask = new_pseudo==i
        tmp_dist_clust = logits_softmax[mask]
        if tmp_dist_clust.numel()==0:
            return torch.tensor(0.0),new_pseudo
        clust_dist.append(tmp_dist_clust.mean(0))
    dists = torch.stack(clust_dist)

    sim = torch.mm(torch.softmax(logits_s, dim=-1), dists.t() / temperature)  # B*N, K*N --> B *K
    sim_probs = sim / sim.sum(1, keepdim=True)

    loss_c = - ((torch.log(sim_probs + 1e-6) * pseudo_onehot)).sum(1)
    loss_c = loss_c * mask_bool.float()

    loss_c = loss_c.mean()

    return loss_c,new_pseudo

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        return F.cross_entropy(logits, targets.long(), reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return nll_loss

        
class AverageMeter(object):
    """
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    
    Args
        output: logits or probs (num of batch, num of classes)
        target: (num of batch, 1) or (num of batch, )
        topk: list of returned k
    
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    
    with torch.no_grad():
        maxk = max(topk) #get k in top-k
        batch_size = target.size(0) #get batch size of target

        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
        # return: value, index
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True) # pred: [num of batch, k]
        pred = pred.t() # pred: [k, num of batch]
        
        #[1, num of batch] -> [k, num_of_batch] : bool
        correct = pred.eq(target.view(1, -1).expand_as(pred)) 

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        #np.shape(res): [k, 1]
        return res         


class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value

        
def consistency_loss(logits_w, logits_s, y_ulb, name='ce', T=1.0, p_cutoff=0.0, 
                     use_hard_labels=True, use_sharpen=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask_bool = max_probs.ge(p_cutoff)
        mask = mask_bool.float()
        
        ulb_acc_num = (max_idx == y_ulb).float() * mask
        
        with torch.no_grad():
            dist_ulb = pseudo_label.detach().cpu().mean(0).tolist()
            if mask.sum() >= 1:
                dist_ulb_high = pseudo_label[mask_bool].detach().cpu().mean(0).tolist()
            else:
                dist_ulb_high = None

        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            if use_sharpen:
                # pseudo_label = torch.softmax(logits_w/T, dim=-1)
                pseudo_label = torch.softmax(pseudo_label/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), mask.sum(), ulb_acc_num.sum(), dist_ulb, dist_ulb_high

    else:
        assert Exception('Not Implemented consistency_loss')
        

def consistency_loss_backup(logits_w, logits_s, y_ulb, name='ce', T=1.0, p_cutoff=0.0, 
                     use_hard_labels=True, use_sharpen=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        
        ulb_acc_num = (max_idx == y_ulb).float() * mask

        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            if use_sharpen:
                # pseudo_label = torch.softmax(logits_w/T, dim=-1)
                pseudo_label = torch.softmax(pseudo_label/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), mask.sum(), ulb_acc_num.sum()

    else:
        assert Exception('Not Implemented consistency_loss')


# method: dgr-> distribution ratio, dgc-> distribution gradient with ce, dgl -> distributi gradient with l2
def consistency_loss_wt_dist(logits_w, logits_s, y_ulb, lst_dist, len_lst, eta, dist_temp=0.2, method="dgr", pre_dist=None,
    name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True, use_sharpen=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)

        # distribution alignment
        tmp_dist = pseudo_label.mean(0)
        num_samples = len(pseudo_label)
        num_class = len(tmp_dist)
        lst_dist.append(tmp_dist)
        if len(lst_dist) > len_lst:
            lst_dist.pop(0)
        dist_avg = torch.stack(lst_dist,dim=0).mean(0)
        
        # pre-distr
        if pre_dist is None:
            pre_dist = torch.ones_like(tmp_dist) / num_class
        
        # revise pseudo-label
        if method == "dgr":
            pseudo_label = pseudo_label * pre_dist / dist_avg
            pseudo_label = pseudo_label /pseudo_label.sum(dim=1, keepdim=True)
        
        if method == "dgrw":
            tmp_ent = -1 * pseudo_label * torch.log(pseudo_label + 1e-10)
            tmp_w = tmp_ent.sum(dim=-1, keepdim=True) / np.log(num_class)
            tmp_adjust = pre_dist / dist_avg
            tmp_adjust = tmp_w * tmp_adjust
            # print("="*10, tmp_w)
            # print("="*10, tmp_ent)
            # print("="*10, tmp_adjust)
            pseudo_label = pseudo_label * tmp_adjust * eta
            pseudo_label = pseudo_label /pseudo_label.sum(dim=1, keepdim=True)
        
        elif method == "dgc":
            tmp_diff = pre_dist / dist_avg
            tmp_ent = -1 * pseudo_label * torch.log(pseudo_label + 1e-10)
            tmp_w = tmp_ent.sum(dim=-1, keepdim=True) / np.log(num_class)
            tmp_grad = eta * tmp_w * tmp_diff
#             print("="*10, tmp_grad)
            pseudo_label = pseudo_label + tmp_grad
            # pseudo_label = pseudo_label /pseudo_label.sum(dim=1, keepdim=True)
            pseudo_label = torch.softmax(pseudo_label / dist_temp, dim=-1)
        
        elif method == "dgl":
            tmp_diff = 2* (dist_avg - pre_dist)
            tmp_ent = -1 * pseudo_label * torch.log(pseudo_label + 1e-10)
            tmp_w = tmp_ent.sum(dim=-1, keepdim=True) / np.log(num_class)
            tmp_grad = eta * tmp_w * tmp_diff
            # print("="*10, tmp_grad)
            pseudo_label = pseudo_label - tmp_grad
            pseudo_label = torch.softmax(pseudo_label / dist_temp, dim=-1)

        else:
            assert Exception('"please propoerly set distritbution consistency: [dgr, dgc, dgl]"')

        # get high-confidence res
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        
        # calculate the quality
        ulb_acc_num = (max_idx == y_ulb).float() * mask

        # calculate loss
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            if use_sharpen:
                # pseudo_label = torch.softmax(logits_w/T, dim=-1)
                pseudo_label = torch.softmax(pseudo_label/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), mask.sum(), ulb_acc_num.sum(), lst_dist

    else:
        assert Exception('Not Implemented consistency_loss')



class NT_Xent(torch.nn.Module):
    def __init__(self, batch_size, temperature, world_size=1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)


        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)


        labels = torch.zeros(N).to(positive_samples).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper


def linear_rampup(rampup_length):
    """Linear rampup"""
    def warpper(epoch):
        if epoch < rampup_length:
            return epoch / rampup_length
        else:
            return 1.0
    return warpper

def step_rampup(k):
    list_k = [3,5,10]
    i = bisect.bisect(list_k,k)
    return list_k[i-1]