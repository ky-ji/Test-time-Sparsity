import json
import numpy as np
import torch
import os


class Logger:
    def __init__(self, args):
        self.logs = []
        self.global_step = 0
        self.best_valid_loss = float('inf')
        self.best_valid_pruning_ratio = 0.0
        self.patience_counter = 0
        self.patience = args.patience
        self.previous_hard_gates = None  
        self.use_swanlab = args.use_swanlab
        self.use_wandb = getattr(args, 'use_wandb', False)  # wandb
        self.use_tensorboard = getattr(args, 'use_tensorboard', False)  # tensorboard
        self.tb_writer = None  # TensorBoard writer
        self.global_step = 0

        # setupConfigSwanLabWandB
        config = {
            'task_name': args.task_name,
            'target_prune_ratio': args.target_prune_ratio,
            'lr': args.lr,
            'warmup_steps': args.warmup_steps,
            'min_lr': args.min_lr,
            'hidden_dim': args.hidden_dim,
            'consistency': args.consistency,
            'lamb_sparse': args.lamb_sparse,
            'seed': args.seed,
            'epochs': args.epochs,
            'patience': args.patience,
            'use_target_action': args.use_target_action,
        }

        # swanlab
        if self.use_swanlab:
            import swanlab
            # swanlab
            swanlab.init(
                project="diffusion-policy-pruner",
                experiment_name=f"{args.task_name}_prune{args.target_prune_ratio:.2f}_seed{args.seed}",
                config=config,
                description=f"Training pruner for {args.task_name} with target pruning ratio {args.target_prune_ratio}"
            )
            print(f"SwanLab: {args.task_name}_prune{args.target_prune_ratio:.1f}_seed{args.seed}")

        # wandb
        if self.use_wandb:
            import wandb
            # wandb
            wandb.init(
                project="diffusion-policy-pruner",
                name=f"{args.task_name}_prune{args.target_prune_ratio:.2f}_seed{args.seed}",
                config=config,
                notes=f"Training pruner for {args.task_name} with target pruning ratio {args.target_prune_ratio}",
                tags=[args.task_name, f"prune_{args.target_prune_ratio:.3f}", f"seed_{args.seed}"]
            )
            print(f"WandB: {args.task_name}_prune{args.target_prune_ratio:.1f}_seed{args.seed}")

        # tensorboard
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            import time
            # Timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_dir = f"runs/{timestamp}_{args.task_name}"
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            
            hparams = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
            self.tb_writer.add_hparams(hparams, {})
            print(f"TensorBoard: {log_dir}")

    def log_train(self, epoch, current_loss, Lc, Ls,current_pruning_ratio, gate_change_info=None, learning_rate=None, gate_stats=None, stage=None, target_step=None):     
        def log():
            msg = {
                'global_step': self.global_step,
                'loss': float(current_loss),
                'Lc': float(Lc),
                'Ls': float(Ls),
                'pruning_ratio': float(current_pruning_ratio),
            }
                  
            if learning_rate is not None:
                msg['learning_rate'] = float(learning_rate)
            
            # Add detailed gate statistics
            if gate_stats is not None:
                # Ternary gate
                if 'p1_reuse_3cache' in gate_stats:
                    msg['p1'] = float(gate_stats['p1_reuse_3cache'])
                if 'p2_reuse_24cache' in gate_stats:
                    msg['p2'] = float(gate_stats['p2_reuse_24cache'])
                if 'p3_reuse_rlcache' in gate_stats:
                    msg['p3'] = float(gate_stats['p3_reuse_rlcache'])
               
            self.logs.append(msg)
            print(json.dumps(msg))
            
            # SwanLabWandB
            log_dict = {
                'train/loss': float(current_loss),
                'train/Lc': float(Lc), 
                'train/Ls': float(Ls),
                'train/pruning_ratio': float(current_pruning_ratio),
            }
            
            # Add detailed gate statistics
            if gate_stats is not None:
                if 'p1_reuse_3cache' in gate_stats:
                    log_dict['train/p1'] = float(gate_stats['p1_reuse_3cache'])
                if 'p2_reuse_24cache' in gate_stats:
                    log_dict['train/p2'] = float(gate_stats['p2_reuse_24cache'])
                if 'p3_reuse_rlcache' in gate_stats:
                    log_dict['train/p3'] = float(gate_stats['p3_reuse_rlcache'])
            
            if learning_rate is not None:
                log_dict['train/learning_rate'] = float(learning_rate)
            

            # SwanLab
            if self.use_swanlab:
                import swanlab
                swanlab.log(log_dict, step=self.global_step)

            # WandB
            if self.use_wandb:
                import wandb
                wandb.log(log_dict, step=self.global_step)

            # TensorBoard
            if self.use_tensorboard and self.tb_writer is not None:
                for key, value in log_dict.items():
                    self.tb_writer.add_scalar(key, value, self.global_step)
            
        
        if stage == "Stage1":
            if(self.global_step % 10) == 0:
                log()
        else:
            if (self.global_step % 1) == 0:             
                log()

    def log_valid(self, epoch, valid_metrics):
        valid_msg = {
            'epoch': epoch + 1,
            'valid_loss': float(valid_metrics['valid_loss']),
            'valid_Lc': float(valid_metrics['valid_consistency_loss']),
            'valid_Ls': float(valid_metrics['valid_sparse_loss']),
            'valid_pruning_ratio': float(valid_metrics['valid_pruning_ratio'])
        }
        
        # Add detailed gate statistics
        if 'gate_stats' in valid_metrics and valid_metrics['gate_stats'] is not None:
            gate_stats = valid_metrics['gate_stats']
            # Ternary gate
            if 'p1_reuse_3cache' in gate_stats:
                valid_msg['p1'] = float(gate_stats['p1_reuse_3cache'])
            if 'p2_reuse_24cache' in gate_stats:
                valid_msg['p2'] = float(gate_stats['p2_reuse_24cache'])
            if 'p3_reuse_rlcache' in gate_stats:
                valid_msg['p3'] = float(gate_stats['p3_reuse_rlcache'])
        
        self.logs.append(valid_msg)
        print(json.dumps(valid_msg))
        
        # SwanLabWandB
        log_dict = {
            'valid/loss': float(valid_metrics['valid_loss']),
            'valid/Lc': float(valid_metrics['valid_consistency_loss']),
            'valid/Ls': float(valid_metrics['valid_sparse_loss']),
            'valid/pruning_ratio': float(valid_metrics['valid_pruning_ratio']),
        }
        
        # Add detailed gate statistics
        if 'gate_stats' in valid_metrics and valid_metrics['gate_stats'] is not None:
            gate_stats = valid_metrics['gate_stats']
            if 'p1_reuse_3cache' in gate_stats:
                log_dict['valid/p1'] = float(gate_stats['p1_reuse_3cache'])
            if 'p2_reuse_24cache' in gate_stats:
                log_dict['valid/p2'] = float(gate_stats['p2_reuse_24cache'])
            if 'p3_reuse_rlcache' in gate_stats:
                log_dict['valid/p3'] = float(gate_stats['p3_reuse_rlcache'])

        # SwanLab
        if self.use_swanlab:
            import swanlab
            swanlab.log(log_dict, step=self.global_step)

        # WandB
        if self.use_wandb:
            import wandb
            wandb.log(log_dict, step=self.global_step)

        # TensorBoard
        if self.use_tensorboard and self.tb_writer is not None:
            for key, value in log_dict.items():
                self.tb_writer.add_scalar(key, value, self.global_step)

    def early_stop(self, valid_metrics,output_dir,epoch,args,pruner=None):

        pruner_model_path = os.path.join(output_dir, f'pruner_model_{epoch}_{valid_metrics["valid_loss"]:.4f}_{valid_metrics["valid_pruning_ratio"]:.2f}.pt')
        os.makedirs(os.path.dirname(pruner_model_path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': pruner.state_dict(),
            'structure': args.structure,
            'hidden_dim': args.hidden_dim,
            'attn_heads': args.attn_heads,
            'dim_feedforward': args.dim_feedforward,
            'block_encoder_type': args.block_encoder_type,
            'valid_loss': valid_metrics['valid_loss'],
            'valid_pruning_ratio': valid_metrics['valid_pruning_ratio'],
            'valid_consistency_loss': valid_metrics['valid_consistency_loss'],
            'valid_sparse_loss': valid_metrics['valid_sparse_loss'],
            'epoch': epoch,
        }
        
        torch.save(save_dict, pruner_model_path)
        print(f"  pruner: {pruner_model_path}")

        if valid_metrics['valid_loss'] < self.best_valid_loss:
            self.patience_counter = 0
            self.best_valid_loss = valid_metrics['valid_loss']
            print(f"  valid_loss: {valid_metrics['valid_loss']:.6f}")
        else:
            self.patience_counter += 1
             
        if self.patience_counter >= self.patience:
            print(f"  {self.patience}batchvalid_loss")
            return True
        return False

    def calculate_gate_change_rate(self, current_hard_gates):
        """
        hard gate
         [batch, T, B, 2]batchgate
        : gategate
        """
        if isinstance(current_hard_gates, torch.Tensor):
            return self._calculate_gate_change_rate_tensor(current_hard_gates)
        else:
            return self._calculate_gate_change_rate_dict(current_hard_gates)
    
    def _calculate_gate_change_rate_tensor(self, current_hard_gates):
        """
         [batch, T, B, 2] gate
        """
        if self.previous_hard_gates is None:
            # gates0
            self.previous_hard_gates = current_hard_gates.clone().detach()
            batch_size, T, B, _ = current_hard_gates.shape
            total_gates = T * B
            return {
                'changed_gates': 0.0,  # gate
                'total_gates': total_gates,
                'change_rate': 0.0,    # 
                'batch_change_rates': [0.0] * batch_size  # batch
            }
        
        batch_size, T, B, _ = current_hard_gates.shape
        total_gates_per_sample = T * B
        
        #  [batch, T, B]
        # gate0=reuse, 1=compute
        prev_decisions = torch.argmax(self.previous_hard_gates, dim=-1)  # [batch, T, B]
        curr_decisions = torch.argmax(current_hard_gates, dim=-1)        # [batch, T, B]
        
        #  [batch, T, B] -> [batch]
        changes_per_sample = (prev_decisions != curr_decisions).sum(dim=(1, 2)).float()
        
        change_rates_per_sample = changes_per_sample / total_gates_per_sample
        
        avg_changed_gates = changes_per_sample.mean().item()
        avg_change_rate = change_rates_per_sample.mean().item()
        
        # previous_hard_gates
        self.previous_hard_gates = current_hard_gates.clone().detach()
        
        return {
            'batch_change_rates': change_rates_per_sample.tolist(),  # batch
            'batch_changed_counts': changes_per_sample.tolist()      # batch
        }
    
    def _calculate_gate_change_rate_dict(self, current_hard_gates):
        """
        gate
        """
        if self.previous_hard_gates is None:
            # gates0
            self.previous_hard_gates = self._deep_copy_gates(current_hard_gates)
            return {'changed_gates': 0, 'total_gates': 0, 'change_rate': 0.0}
        
        changed_gates = 0
        total_gates = 0
        
        # stepblockgate
        steps = sorted(current_hard_gates.keys())
        for step in steps:
            if step in self.previous_hard_gates:
                blocks = current_hard_gates[step].keys()
                for block in blocks:
                    if block in self.previous_hard_gates[step]:
                        total_gates += 1
                        
                        # gateargmax
                        prev_decision = torch.argmax(self.previous_hard_gates[step][block]).item()
                        curr_decision = torch.argmax(current_hard_gates[step][block]).item()
                        
                        if prev_decision != curr_decision:
                            changed_gates += 1
        
        change_rate = changed_gates / total_gates if total_gates > 0 else 0.0
        
        # previous_hard_gates
        self.previous_hard_gates = self._deep_copy_gates(current_hard_gates)
        
        return {
            'changed_gates': changed_gates,
            'change_rate': change_rate
        }
    
    def _deep_copy_gates(self, gates):
        """gates"""
        copied_gates = {}
        for step, step_gates in gates.items():
            copied_gates[step] = {}
            for block, gate in step_gates.items():
                copied_gates[step][block] = gate.clone().detach()
        return copied_gates

    def save_pruner_ckpt(self, ckpt_path, pruner_state_dict):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(pruner_state_dict, ckpt_path)
    
    def finish(self):
        """swanlabwandbtensorboard"""
        if self.use_swanlab:
            import swanlab
            swanlab.finish()
            print("SwanLab")
        
        if self.use_wandb:
            import wandb
            wandb.finish()
            print("WandB")
        
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()
            print("TensorBoard")

    def update_global_step(self):
        self.global_step +=1