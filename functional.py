import torch
import torch.nn as nn
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1,topk=1,test_mode=False, use_gumbel_noise=True):
    # type: (Tensor, float, bool, float, int) -> Tensor
    """
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if use_gumbel_noise:
      y_soft = gumbels.softmax(dim)
    else:
      logits = logits/tau
      y_soft = logits.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        max_values, max_indices = torch.topk(y_soft,topk,dim=dim,largest=True,sorted=True)
        max_values_normalized = max_values/torch.sum(max_values,1,keepdim=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, max_indices, max_values_normalized)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def gumbel_sigmoid(logits, tau=1, hard=False,use_gumbel_noise=True):

    gumbels1 = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels2 = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    
    sigmoid = nn.Sigmoid()
    gumbels = (logits + gumbels1 - gumbels2) / tau 
    
    if use_gumbel_noise:
      y_soft = sigmoid(gumbels)
    else:
      y_soft = sigmoid(logits/tau)
      
    if hard:
        y_hard = (y_soft>0.5).type(torch.float)
        ret = (y_hard - y_soft).detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = x*torch.log(x)
        b = -1.0 * b.mean()
        return b

if __name__ == '__main__':
  logits = torch.randn([128,5]).cuda()
  gumbel_softmax(logits,hard=True,topk=2)
  print(logits.shape)
  breakpoint()