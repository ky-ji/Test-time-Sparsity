

def get_task_ckpt(task_name: str, ckpt_id: int = 0):
    return get_task_ckpt_with_train_version(task_name, ckpt_id)


def get_task_ckpt_with_train_version(task_name: str, train_version: int = 0) -> str:
    """Get task checkpoint path for specified train version."""
    LOW_DIM_TASKS = {'kitchen', 'block_pushing'}
    TASK_ALIASES = {'tool': 'tool_hang_ph'}
    
    resolved = TASK_ALIASES.get(task_name, task_name)
    prefix = 'checkpoint/low_dim' if resolved in LOW_DIM_TASKS else 'checkpoint'
    return f'{prefix}/{resolved}/diffusion_policy_transformer/train_{train_version}/checkpoints/latest.ckpt'

