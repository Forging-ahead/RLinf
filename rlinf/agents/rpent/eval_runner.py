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

"""Evaluate an RPent planner on fixed tasks with their native metrics."""

import json
import os
from collections import defaultdict

from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.runners.agent_eval_runner import AgentEvalRunner
from rlinf.utils.logging import get_logger
from rlinf.utils.runner_utils import local_mkdir_safe

logger = get_logger()
_EPISODE_FIELDS = (
    "task",
    "success",
    "init_state_id",
    "termination_reason",
    "turns",
    "episode_id",
    "infra_retries",
)


def _metrics(episodes: list[dict]) -> dict:
    successes = [e["success"] for e in episodes if e["success"] is not None]
    returns = [
        e["episode_return"] for e in episodes if e.get("episode_return") is not None
    ]
    metrics = {
        "episodes": len(episodes),
        "episodes_with_success_metric": len(successes),
    }
    if successes:
        metrics["success_rate"] = sum(successes) / len(successes)
    if returns:
        metrics["mean_episode_return"] = sum(returns) / len(returns)
    return metrics


def summarize_episodes(episodes: list[dict]) -> dict:
    """Summarize LIBERO by suite and other simulators by native task."""
    suites, tasks = defaultdict(list), defaultdict(list)
    for episode in episodes:
        task = episode["task"]
        if "suite" in task:
            suites[task["suite"]].append(episode)
        else:
            tasks[f"{task['environment']}/{task['task_name']}"].append(episode)
    summary = _metrics(episodes)
    summary["suites"] = {key: _metrics(rows) for key, rows in sorted(suites.items())}
    if tasks:
        summary["tasks"] = {key: _metrics(rows) for key, rows in sorted(tasks.items())}
    return summary


class RPentEvalRunner(AgentEvalRunner):
    """Run the validation tasks and write ``eval_results.json``."""

    def pre_process(self) -> dict:
        return {"episodes": []}

    def update(self, context: dict, eval_pbar, input_channel, batch_idx, batch):
        received = 0
        while received < self.total_batch_size:
            result: DynamicRolloutResult = input_channel.get()
            traj = result.extra_fields_traj
            episodes = len(traj["success"])
            for i in range(episodes):
                row = {key: traj[key][i] for key in _EPISODE_FIELDS}
                if row["task"].get("environment") == "maniskill":
                    for key in (
                        "episode_return",
                        "terminated",
                        "fail",
                        "truncated",
                        "terminal",
                    ):
                        row[key] = traj[key][i]
                context["episodes"].append(row)
            received += episodes
            eval_pbar.update(episodes)

    def update_batch(self, context: dict, eval_pbar, time_metrics):
        summary = summarize_episodes(context["episodes"])
        eval_pbar.set_postfix(
            {
                key: f"{summary[key]:.4f}"
                for key in ("success_rate", "mean_episode_return")
                if key in summary
            }
        )

    def post_process(self, context: dict) -> None:
        summary = summarize_episodes(context["episodes"])
        output_dir = os.path.join(
            self.cfg.runner.output_dir, self.cfg.runner.experiment_name
        )
        local_mkdir_safe(output_dir)
        output_file = os.path.join(output_dir, "eval_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "episodes": context["episodes"]}, f, indent=2
            )
        logger.info(
            f"Evaluated {summary['episodes']} RPent episodes. Results in {output_file}"
        )
