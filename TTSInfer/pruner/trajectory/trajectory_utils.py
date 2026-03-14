import torch


def expand_pruner_head_for_rollout_cache(pruner, init_bias_trajectory: float):
    """
    Expand a 3-way pruner head into a 4-way head with rollout-cache reuse.

    Args:
        pruner: `TransformerPruner` instance.
        init_bias_trajectory: Initial bias for the rollout-cache logit.

    Returns:
        The pruner with an updated output head.
    """
    device = next(pruner.parameters()).device
    old_head = pruner.head

    old_weight = old_head.weight.data
    old_bias = old_head.bias.data
    hidden_dim = old_weight.shape[1]

    new_head = torch.nn.Linear(hidden_dim, 4).to(device)

    with torch.no_grad():
        new_head.weight.data[:3, :] = old_weight
        if old_bias is not None:
            new_head.bias.data[:3] = old_bias

        torch.nn.init.xavier_uniform_(new_head.weight.data[3:4, :])
        if new_head.bias is not None:
            torch.nn.init.zeros_(new_head.bias.data[3:4])
            new_head.bias.data[3:4] += init_bias_trajectory

    pruner.head = new_head
    return pruner
