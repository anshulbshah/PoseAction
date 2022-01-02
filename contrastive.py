from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, contrastive_loss_type='all', debug=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        # features --> J x B x views x 128

        batch_size = features.shape[1]
        if not contrastive_loss_type in ['all_charades_atleast_one','all_charades_weighted']:
            labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size and contrastive_loss_type in ['all']:
            raise ValueError('Num of labels does not match num of features')            
        if contrastive_loss_type == 'only_aug':
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif contrastive_loss_type == 'all':
            mask = torch.eq(labels, labels.T).float().to(device)
        elif contrastive_loss_type == 'all_charades_atleast_one':
            dot_product = labels@labels.T
            mask = (dot_product > 0.0).float().to(device)
        elif contrastive_loss_type == 'all_charades_weighted':
            dot_product = labels@labels.T   
            dir0_sum = torch.sum(labels,1)
            max_val = torch.max(dir0_sum.unsqueeze(0).repeat(batch_size,1),dir0_sum.unsqueeze(1).repeat(1,batch_size))
            mask = dot_product/(max_val + 1E-3)
            # import pdb; pdb.set_trace()



        # features : J x B x 2 x D
        contrast_count = features.shape[2]
        contrast_feature = torch.cat(torch.unbind(features, dim=2), dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits

        anchor_dot_contrast = torch.div(
            torch.bmm(anchor_feature, contrast_feature.permute(0,2,1)),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=2, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask.unsqueeze(0)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask.unsqueeze(0) * log_prob).sum(1) / (mask.unsqueeze(0).sum(2) + 1E-8)

        # loss
        loss = -1.0*mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        # if debug:
        #     import pdb; pdb.set_trace()
        return loss.permute(1,0), mask