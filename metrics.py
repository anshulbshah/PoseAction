import numpy as np
import torch
from sklearn.metrics import average_precision_score

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.dim() == 3:
        target = target.max(dim=1)[0]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if len(target.shape) == 1:
        #print('computing accuracy for single-label case')
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    else:
        #print('computing accuracy for multi-label case')
        correct = torch.zeros(*pred.shape)
        for i in range(correct.shape[0]):
            for j in range(correct.shape[1]):
                correct[i, j] = target[j, pred[i, j]] > 0.5

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #breakpoint()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def mean_classwise_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.dim() == 3:
        target = target.max(dim=1)[0]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if len(target.shape) == 1:
        #print('computing accuracy for single-label case')
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    else:
        #print('computing accuracy for multi-label case')
        correct = torch.zeros(*pred.shape)
        for i in range(correct.shape[0]):
            for j in range(correct.shape[1]):
                correct[i, j] = target[j, pred[i, j]] > 0.5

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #breakpoint()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.nanmean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps


def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return map(fix, gt_array)

def charades_submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))

def weighted_scores(scores,weighted_logits,k=1):
    _,max_indices = torch.topk(scores,k,dim=1,largest=True,sorted=True)
    new_logits = weighted_logits.new_ones((weighted_logits.shape[0],weighted_logits.shape[-1]))
    #breakpoint()
    for b in range(weighted_logits.shape[0]):
        new_logits[b] = torch.sum(weighted_logits[b,max_indices[b]],0)

    return new_logits,max_indices
    #breakpoint()

def get_map_ava(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap,aps
