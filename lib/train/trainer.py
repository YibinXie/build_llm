from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader
from lib.loss.loss import CrossEntropyLoss
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import json
import random
from typing import Union
from torchinfo import summary
from lib.params.config import TrainingConfig


class Trainer:

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1)

    def save_hyperparams(self, hp):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')

        with open(f'./runs/{self.run_name}/hyperparams.json', 'w') as fp:
            json.dump(hp, fp, indent=4)

    def save_metrics(self, metrics):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        with open(f'./runs/{self.run_name}/metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=4)

    def save_states(self, step, is_last=False):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        file_name = f'{self.run_name}_final.pt' if is_last else f'{self.run_name}_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict':
                    self.model.state_dict(),  # Save the unoptimized model
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'./runs/{self.run_name}/{file_name}')


class SFTTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"sft_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = device
        # assert self.device == 'cuda'
        self.max_steps = cfg.max_steps
        self.eval_freq = 1
        self.save_freq = 20000
        self.train_dataloader = iter(
            DataLoader(train_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=8,
                       pin_memory=True))
        self.test_dataloader = iter(
            DataLoader(test_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=8,
                       pin_memory=True))
        self.model = model
        self.criterion = CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float16

        self.finetune_method = cfg.finetune_method

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)
        summary(self.model, input_data=torch.ones(1, 1024).long())

        # opt_model = torch.compile(self.model)
        self.model.to(self.device)
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        self.model.train()
        step = 0

        t0 = time.time()
        while step < self.max_steps:
            x, y = next(self.train_dataloader)
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.autocast(device_type=self.device, dtype=self.dtype):
                y_hat, loss = self.model(x, y)  # (B, 1)
                # loss = self.criterion(y_hat, y)  # (B, 1)

            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.grad_clip)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            lossf = loss.item()

            iter_time = time.time() - t0
            t0 = time.time()
            print(
                f"step {step}, batch loss {round(lossf, 3)}, {round(1.0 / iter_time, 2)} iters/s"
            )
            writer.add_scalar('Loss/train/step', lossf, step)

            if step != 0 and step % self.save_freq == 0:
                self.save_states(step)

            step += 1

        self.save_states(step, True)


class SFTTrainerV2:
    def __init__(self, model, data_loader: DataLoader, device, log_dir):
        self.model = model
        self.data_loader = iter(data_loader)
        self.device = device
        self.dtype = torch.bfloat16
        self.grad_clip = 1.0
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.log_dir = log_dir

    def train(self, max_steps):
        scaler = GradScaler(enabled=self.dtype != torch.float32)
        for step in range(max_steps):
            start_time = time.time()
            last_step = (step == max_steps - 1)
            x, y, loss_mask = next(self.data_loader)
            x = x.to(self.device)
            y = y.to(self.device)
            loss_mask = loss_mask.to(self.device)
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                y_hat, loss = self.model(x, y)
                loss_mask = loss_mask.view(-1)
                loss = torch.sum(loss * loss_mask) / loss_mask.sum()
            scaler.scale(loss).backward()
            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if step % 1 == 0:
                spend_time = time.time() - start_time
                print(
                    f"step {step}/{max_steps}, lr {self.optimizer.param_groups[-1]['lr']}, "
                    f"batch loss {round(loss.item(), 3)}, time {round(spend_time, 2)}s"
                )
            if step > 0 and (step % 100 == 0 or last_step):
                checkpoint_path = os.path.join(self.log_dir, f"model_sft_{step:04d}.pt")
                checkpoint = {
                    'model': self.model.state_dict(),
                    'config': self.model.config,
                    'step': step,
                    'train_loss': loss.item()
                }
                torch.save(checkpoint, checkpoint_path)


class RewardModelTrainer:
    def __init__(self, model, train_data_loader: DataLoader, test_data_loader: DataLoader, device, log_dir):
        self.model = model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.device = device
        self.dtype = torch.bfloat16
        self.grad_clip = 1.0
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.log_dir = log_dir
        self.log_path = os.path.join(self.log_dir, 'log.txt')
        self.criterion = KPairwiseLoss()
        self.eval_freq = 100
        # clear file
        with open(self.log_path, 'w'):
            pass

    def train(self, max_epochs):
        scaler = GradScaler(enabled=self.dtype != torch.float32)
        max_steps = max_epochs * len(self.train_data_loader)
        step = 0
        for epoch in range(max_epochs):
            for pair in self.train_data_loader:
                # eval part
                start_time = time.time()
                if step % self.eval_freq == 0:
                    self.model.eval()
                    with torch.no_grad():
                        tp = 0
                        total = 0
                        losses = []
                        for pair in self.test_data_loader:
                            pos = pair[:, 0, :].to(self.device)
                            neg = pair[:, 1, :].to(self.device)
                            pos_scores = self.model(pos)
                            neg_scores = self.model(neg)
                            loss = self.criterion(torch.cat((pos_scores, neg_scores), dim=-1))  # (B, 2)
                            loss_val = loss.item()
                            losses.append(loss_val)
                            tp += torch.count_nonzero(pos_scores > neg_scores)
                            total += pos_scores.shape[0]
                        eval_loss = np.mean(losses)
                        acc = (tp / total).item()
                        spend_time = time.time() - start_time
                        print(
                            f"eval {step}, eval loss {round(eval_loss, 4)}, "
                            f"eval acc {round(acc, 3)}, time {round(spend_time, 2)}s"
                        )
                        with open(self.log_path, "a") as f:
                            f.write(f"{step} val eval_loss: {eval_loss:.4f} acc: {acc:.4f}\n")
                # train part
                start_time = time.time()
                last_step = (step == max_steps - 1)
                self.model.train()
                pos = pair[:, 0, :].to(self.device)
                neg = pair[:, 1, :].to(self.device)
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    pos_scores = self.model(pos)
                    neg_scores = self.model(neg)
                    loss = self.criterion(torch.cat((pos_scores, neg_scores), dim=-1))  # (B, 2)
                scaler.scale(loss).backward()
                if self.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if step % 1 == 0:
                    spend_time = time.time() - start_time
                    print(
                        f"step {step}/{max_steps}, lr {self.optimizer.param_groups[-1]['lr']}, "
                        f"batch loss {round(loss.item(), 3)}, time {round(spend_time, 2)}s"
                    )
                    with open(self.log_path, "a") as f:
                        f.write(f"{step} train {loss.item():.6f}\n")
                if step > 0 and (step % 2000 == 0 or last_step):
                    checkpoint_path = os.path.join(self.log_dir, f"model_rm_{step}.pt")
                    checkpoint = {
                        'model': self.model.state_dict(),
                        'config': self.model.config,
                        'step': step,
                        'train_loss': loss.item()
                    }
                    torch.save(checkpoint, checkpoint_path)
                step += 1


@dataclass
class Experience:
    completion: torch.Tensor
    actor_log_probs: torch.Tensor
    attention_mask: torch.Tensor
    kL_penalized_reward: torch.Tensor
    advantage: torch.Tensor
    num_actions: int
    estimated_kl: torch.Tensor
    values: torch.Tensor
    action_mask: torch.Tensorclass


class PPOTrainer:
    def __init__(self, sft_model, reward_model, actor, critic, train_data_loader: DataLoader,
                 test_data_loader: DataLoader, device, log_dir):
        self.sft_model = sft_model
        self.reward_model = reward_model
        self.actor = actor
        self.critic = critic
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.device = device
        self.dtype = torch.bfloat16
        self.grad_clip = 1.0
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-4)
        self.log_dir = log_dir
        self.log_path = os.path.join(self.log_dir, 'log.txt')

        self.actor_criterion = PolicyLoss()
        self.critic_criterion = ValueLoss()
        self.eval_freq = 100
        # clear file
        with open(self.log_path, 'w'):
            pass

    def kl_penalized_reward(
            self,
            reward: torch.Tensor,
            log_prob_rl: torch.Tensor,
            log_prob_sft: torch.Tensor,
            action_mask: torch.Tensor = None
    ) -> Union[torch.Tensor, torch.Tensor]:
        # Log(T_RL(y|x) / T_SFT(y|x)) = Log(T_RL(y|x)) - Log(T_SFT(y|x))
        ratio = log_prob_rl - log_prob_sft
        # k3 in http://joschu.net/blog/kl-approx.html
        estimated_kl = (torch.exp(ratio) - 1) - ratio
        if action_mask is not None:
            estimated_kl *= action_mask
        estimated_kl = estimated_kl.sum(dim=1) / action_mask.sum(dim=1)
        estimated_kl = estimated_kl.mean(dim=1, keepdim=True)  # estimated_kl -> (B, 1)
        return reward - estimated_kl, estimated_kl

    def make_experience(self, idx):
        self.sft_model.eval()
        self.reward_model.eval()
        self.actor.eval()
        self.critic.eval()
        completions, num_actions = self.actor.generate(idx, max_new_tokens=32, top_k=50)
        actor_log_probs = self.actor.forward_actor(completions, num_actions)
        sft_log_probs = self.sft_model.forward_actor(completions, num_actions)
        values = self.critic.forward_critic(completions, num_actions).view(-1, 1)
        reward = self.reward_model(completions)
        kL_reward, kL_div = self.kl_penalized_reward(reward, actor_log_probs, sft_log_probs)
        advantage = kL_reward - values
        return Experience(
            completion=completions,
            actor_log_probs=actor_log_probs,
            attention_mask=None,
            kL_penalized_reward=kL_reward,
            advantage=advantage,
            num_actions=num_actions,
            estimated_kl=kL_div,
            values=values,
            action_mask=None
        )

    def train(self, max_epochs):
        scaler = GradScaler(enabled=self.dtype != torch.float32)
        max_steps = max_epochs * len(self.train_data_loader)
        step = 0
        for epoch in range(max_epochs):
            for x, _, _ in self.train_data_loader:
                # train part
                start_time = time.time()
                last_step = (step == max_steps - 1)
                x = x.to(self.device)
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    # train actor
                    self.actor.train()
                    experience = self.make_experience(x)
                    curr_actor_log_probs = self.actor.forward_actor(experience.completion, experience.num_actions)
                    actor_loss = self.actor_criterion(curr_actor_log_probs, experience.actor_log_probs,
                                                      experience.advantage)
                    scaler.scale(actor_loss).backward()
                    if self.grad_clip != 0.0:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                    scaler.step(self.actor_opt)
                    scaler.update()
                    self.actor_opt.zero_grad(set_to_none=True)

                    # train critic
                    self.critic.train()
                    new_values = self.critic.forward_critic(experience.completion, experience.num_actions).view(-1, 1)
                    critic_loss = self.critic_criterion(new_values, experience.kL_penalized_reward, experience.values)
                    scaler.scale(critic_loss).backward()
                    if self.grad_clip != 0.0:
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                    scaler.step(self.critic_opt)
                    scaler.update()
                    self.critic_opt.zero_grad(set_to_none=True)

                if step % 1 == 0:
                    spend_time = time.time() - start_time
                    print(
                        f"step {step}/{max_steps}, "
                        f"actor lr {self.actor_opt.param_groups[-1]['lr']}, "
                        f"critic lr {self.critic_opt.param_groups[-1]['lr']}, "
                        f"actor loss {round(actor_loss.item(), 3)}, "
                        f"critic loss {round(critic_loss.item(), 3)}, "
                        f"time {round(spend_time, 2)}s"
                    )

                    with open(self.log_path, "a") as f:
                        f.write(f"{step} train actor {actor_loss.item():.6f} critic {critic_loss.item():.6f}\n")

                if step > 0 and (step % 2000 == 0 or last_step):
                    checkpoint_path = os.path.join(self.log_dir, f"model_rm_{step}.pt")
                    checkpoint = {
                        'actor': self.actor.state_dict(),
                        'critic': self.critic.state_dict(),
                        'actor_config': self.actor.config,
                        'critic_config': self.critic.config,
                        'step': step,
                        'actor_train_loss': actor_loss.item(),
                        'critic_train_loss': critic_loss.item()
                    }
                    torch.save(checkpoint, checkpoint_path)
                step += 1
