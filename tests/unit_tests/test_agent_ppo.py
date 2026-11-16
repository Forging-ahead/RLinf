# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous agent PPO targets and placement-independent loss scaling."""

import asyncio
import copy
from types import SimpleNamespace

import pytest
import torch

from rlinf.algorithms.losses import (
    compute_ppo_actor_loss,
    compute_ppo_critic_loss,
)
from rlinf.algorithms.registry import get_adv_and_returns
from rlinf.data.schema.agent_trajectory import prepare_agent_ppo_targets
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.scheduler import Channel
from rlinf.scheduler.channel.channel_worker import LocalChannel


def _local_channel():
    channel = Channel()
    channel._initialize("ppo-targets", None, None, None, local_channel=LocalChannel())
    return channel


def _prepare_targets(results, gamma, gae_lambda, normalize_advantages):
    input_channel, output_channel = _local_channel(), _local_channel()
    for result in results:
        input_channel.put(result)
    prepare_agent_ppo_targets(
        input_channel,
        output_channel,
        len(results),
        gamma,
        gae_lambda,
        normalize_advantages,
    )
    for result in results:
        assert output_channel.get() is result


def test_masked_gae_carries_across_observations_and_bootstraps_truncation():
    values = torch.tensor([[0.2, 0.1], [99.0, 0.3], [0.4, 99.0]])
    rewards = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    mask = torch.tensor([[True, True], [False, True], [True, False]])
    advantages, returns = get_adv_and_returns("actor_gae")(
        rewards,
        values=values,
        step_mask=mask,
        bootstrap_values=torch.tensor([0.0, 0.5]),
        normalize_advantages=False,
    )
    torch.testing.assert_close(
        advantages, torch.tensor([[0.8, 0.4], [0.0, 0.2], [0.6, 0.0]])
    )
    torch.testing.assert_close(
        returns, torch.tensor([[1.0, 0.5], [0.0, 0.5], [1.0, 0.0]])
    )


@pytest.mark.parametrize("role", ["actor", "critic"])
@pytest.mark.parametrize("dp_size,microbatches", [(2, 1), (4, 4)])
def test_megatron_valid_token_scale_matches_unsplit_gradient(
    monkeypatch, tmp_path, role, dp_size, microbatches
):
    pytest.importorskip("megatron.training")
    from megatron.core import parallel_state
    from omegaconf import OmegaConf

    from rlinf.scheduler import Worker
    from rlinf.utils.utils import masked_mean
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic

    class TokenModel(torch.nn.Module):
        pre_process = True
        post_process = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.01))

        def forward(self, input_ids, attention_mask, position_ids, packed_seq_params):
            output = (input_ids.float() * self.weight).unsqueeze(-1)
            if role == "actor":
                output = output * torch.arange(8, dtype=torch.float32)
            return output

    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    torch.distributed.init_process_group(
        "gloo", init_method=(tmp_path / "loss-group").as_uri(), rank=0, world_size=1
    )
    try:
        for name in (
            "_TENSOR_MODEL_PARALLEL_GROUP",
            "_CONTEXT_PARALLEL_GROUP",
            "_DATA_PARALLEL_GROUP",
        ):
            monkeypatch.setattr(parallel_state, name, torch.distributed.group.WORLD)
        worker_cls = MAMegatronActor if role == "actor" else MAMegatronCritic
        worker = object.__new__(worker_cls)
        worker.cfg = OmegaConf.create(
            {
                "runner": {"task_type": "reasoning"},
                "rollout": {"model": {"model_path": "/test/model"}},
                "tools": {},
                "critic": {},
                "algorithm": {
                    "loss_type": "actor",
                    "use_valid_token_scale": False,
                    "sampling_params": {"temperature": 1.0},
                    "early_stop_imp_ratio": None,
                },
            }
        )
        worker.transformer_config = SimpleNamespace(sequence_parallel=False)
        worker.return_loss = True
        worker.loss_agg_func = masked_mean
        worker.calculate_entropy = False
        worker.kl_beta = 0
        worker.clip_ratio_c = None
        worker.clip_ratio_low = worker.clip_ratio_high = 0.2
        worker.value_clip = 0.5
        worker.num_microbatches = 1
        worker.response_only_logits = False
        worker.lm_head_chunk_size = 0
        input_ids = torch.arange(128).reshape(32, 4) % 7 + 1
        positions = torch.arange(4).expand_as(input_ids).clone()
        mask = (positions > 0) & (positions <= (torch.arange(32) % 3 + 1)[:, None])
        mask[-1] = False
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(mask),
            "position_ids": positions,
            "response_mask": mask,
            "loss_scales": mask.float(),
            "advantages": input_ids.float() / 7 - 0.5,
            "recomputed_logprobs": torch.zeros_like(mask, dtype=torch.float32),
            "values": torch.zeros_like(mask, dtype=torch.float32),
            "returns": torch.full_like(mask, 0.7, dtype=torch.float32),
        }
        expected_model = TokenModel()
        forward = worker.get_forward_step_func()
        output, loss_func = forward(iter([copy.deepcopy(batch)]), expected_model)
        expected_loss, _ = loss_func(output)
        expected_loss.backward()

        worker.cfg.algorithm.use_valid_token_scale = True
        worker._setup_valid_token_scale(batch)
        assert worker.global_valid_token.item() == mask.sum().item()
        worker.num_microbatches = microbatches
        monkeypatch.setattr(
            parallel_state, "get_data_parallel_world_size", lambda: dp_size
        )
        model = TokenModel()
        for rank in range(dp_size):
            indices = torch.arange(rank, len(input_ids), dp_size)
            for rows in torch.tensor_split(indices, microbatches):
                microbatch = {key: value[rows] for key, value in batch.items()}
                output, loss_func = forward(iter([microbatch]), model)
                loss, metrics = loss_func(output)
                assert loss.requires_grad and torch.isfinite(loss)
                metric = (
                    "actor/final_loss" if role == "actor" else "critic/final_value_loss"
                )
                torch.testing.assert_close(metrics[metric].squeeze(), loss)
                # Megatron averages over microbatches, then DP gradients.
                (loss / microbatches / dp_size).backward()
        torch.testing.assert_close(model.weight.grad, expected_model.weight.grad)
    finally:
        torch.distributed.destroy_process_group()


def make_result():
    return DynamicRolloutResult(
        num_sequence=3,
        group_size=2,
        idx_to_traj=[0, 0, 1],
        input_ids=[[10, 11, 12], [10, 11, 12, 50, 51, 13], [20, 21, 22]],
        prompt_lengths=[1, 5, 1],
        response_lengths=[2, 1, 2],
        is_end=[False, True, False],
        rewards=[1.0, 1.0, 0.0],
        values=torch.tensor(
            [[0.0, 0.2, 0.3, 0, 0, 0], [0, 0, 0, 0, 0, 0.4], [0, 0.1, 0.2, 0, 0, 0]]
        ),
    )


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_trajectory_packing_preserves_ppo_targets_loss_and_gradients(monkeypatch, role):
    from rlinf.scheduler import Worker

    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    result = make_result()
    _prepare_targets([result], 0.99, 0.95, True)
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Embedding(64, 4), torch.nn.Linear(4, 64 if role == "actor" else 1)
    )

    def predict(model, batch):
        hidden = model[0](batch["input_ids"]).cumsum(dim=1)
        raw = model[1](hidden) * 0.05
        predictions = torch.zeros_like(batch["input_ids"], dtype=torch.float32)
        if role == "actor":
            next_logprobs = (
                raw[:, :-1]
                .log_softmax(-1)
                .gather(-1, batch["input_ids"][:, 1:].unsqueeze(-1))
                .squeeze(-1)
            )
            predictions[:, 1:] = next_logprobs
        else:
            predictions[:, 1:] = raw[:, :-1, 0]
        return predictions

    batch = result.to_actor_batch(6, 0)
    batch["recomputed_logprobs"] = predict(model, batch).detach() - 0.02
    original = copy.deepcopy(batch)
    packed = DynamicRolloutResult.pack_traj_batch({}, batch)
    assert packed["idx_to_traj"] == [0, 1]
    assert packed["input_ids"].shape[0] == 2
    assert packed["attention_mask"].sum() < original["attention_mask"].sum()
    assert packed["response_mask"].sum() == original["response_mask"].sum()
    for key in ("recomputed_logprobs", "advantages", "values", "returns"):
        torch.testing.assert_close(
            packed[key][packed["response_mask"]],
            original[key][original["response_mask"]],
        )
    for key in batch:
        if torch.is_tensor(batch[key]):
            torch.testing.assert_close(batch[key], original[key])

    packed_model = copy.deepcopy(model)

    def loss(model, batch):
        predictions = predict(model, batch)
        if role == "actor":
            return compute_ppo_actor_loss(
                logprobs=predictions,
                old_logprobs=batch["recomputed_logprobs"],
                advantages=batch["advantages"],
                clip_ratio_low=0.2,
                clip_ratio_high=0.2,
                loss_mask=batch["response_mask"],
            )[0]
        return compute_ppo_critic_loss(
            values=predictions,
            prev_values=batch["values"],
            returns=batch["returns"],
            value_clip=0.5,
            huber_delta=10000,
            loss_mask=batch["response_mask"],
        )[0]

    original_loss, packed_loss = loss(model, original), loss(packed_model, packed)
    torch.testing.assert_close(packed_loss, original_loss)
    original_loss.backward()
    packed_loss.backward()
    for before, after in zip(
        model.parameters(), packed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(after.grad, before.grad, rtol=1e-5, atol=1e-7)


# A Qwen3.5 shape small enough to test, with Gated DeltaNet sizes that differ
# from Megatron's defaults.
QWEN3_5_SHAPE = {
    "hidden_size": 8,
    "num_attention_heads": 4,
    "num_query_groups": 2,
    "kv_channels": 3,
    "attention_output_gate": True,
    "linear_key_head_dim": 2,
    "linear_num_key_heads": 2,
    "linear_value_head_dim": 3,
    "linear_num_value_heads": 4,
}


@pytest.mark.parametrize("presence_penalty", [0.0, 1.5])
def test_tis_corrects_penalized_behavior_without_differentiating_the_weights(
    presence_penalty,
):
    """Enumerating the vocabulary gives an exact expected policy gradient."""
    from rlinf.algorithms.utils import truncated_importance_weights

    temperature, clip = 0.7, 2.0
    logits = torch.tensor([[0.4, -0.2, 0.8, 0.1]], requires_grad=True)
    seen = torch.tensor([[True, False, True, False]])
    target = (logits / temperature).log_softmax(-1)
    behavior = ((logits.detach() - presence_penalty * seen) / temperature).softmax(-1)
    mask = torch.ones_like(seen)
    weights = truncated_importance_weights(target, behavior.log(), mask, clip)
    assert not weights.requires_grad
    torch.testing.assert_close(
        weights, (target.detach().exp() / behavior).clamp(max=clip)
    )

    advantages = torch.tensor([[1.0, -0.5, -1.0, 0.25]])
    loss, _ = compute_ppo_actor_loss(
        logprobs=target,
        old_logprobs=target.detach(),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages * weights,
        loss_mask=mask,
        loss_agg_func=lambda values, loss_mask, *args: (
            values * behavior * loss_mask
        ).sum(),
    )
    gradient = torch.autograd.grad(loss, logits)[0]
    probabilities = target.detach().exp()
    mass = torch.minimum(probabilities, clip * behavior) * advantages
    expected = -(mass - probabilities * mass.sum()) / temperature
    torch.testing.assert_close(gradient, expected, atol=1e-6, rtol=1e-6)


def test_rollout_failure_cancels_and_drains_sibling_episodes():
    from rlinf.workers.agent.agent_loop import MultiAgentLoopWorker

    cleaned = []

    async def sibling():
        try:
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0)
            cleaned.append("sibling")

    async def failing():
        await asyncio.sleep(0)
        raise RuntimeError("episode failed")

    async def main():
        tasks = [asyncio.create_task(sibling()), asyncio.create_task(failing())]
        with pytest.raises(RuntimeError, match="episode failed"):
            await MultiAgentLoopWorker._gather_rollout_tasks(tasks)
        # The sibling finished its cleanup before the failure reached the caller.
        assert cleaned == ["sibling"] and tasks[0].cancelled()

    asyncio.run(asyncio.wait_for(main(), timeout=5))
