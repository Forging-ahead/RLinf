RL with Behavior Benchmark
==========================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
   :align: center
   :width: 90%

   The BEHAVIOR benchmark (image: `BEHAVIOR <https://behavior.stanford.edu>`__).

`BEHAVIOR <https://behavior.stanford.edu>`__ is a benchmark of everyday household
activities built on NVIDIA IsaacSim / OmniGibson. A dual-arm R1 Pro robot performs
long-horizon manipulation; RLinf uses it to RL-fine-tune vision-language-action (VLA)
policies.

Overview
--------

RL-finetune a VLA on BEHAVIOR household tasks with PPO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      50 BEHAVIOR-1K tasks

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · ray-tracing GPUs

| **You'll do:** install IsaacSim deps → download assets + base model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Isaac Sim 5.1 + BEHAVIOR-1K assets (>30 GB) · a base checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Detail
   * - Tasks
     - 50 household manipulation tasks from BEHAVIOR-1K (select via ``omni_config.task.activity_name`` and its scene).
   * - Robot
     - Dual-arm R1 Pro on IsaacSim / OmniGibson.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Head-camera RGB (720×720) plus left/right wrist RealSense RGB (480×480).
   * - Action
     - 23-dim continuous: 3-DOF base (x, y, rz), 4-DOF torso, 2×7-DOF arms, and 2×1-DOF parallel-jaw grippers.


Installation
------------

The BEHAVIOR simulator uses the official ``v3.9.3-post2`` SDK with OmniGibson 3.9.3,
Isaac Sim 5.1, and PyTorch 2.7.0 on Linux. Check the
`Isaac Sim 5.1 hardware and driver requirements <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_
before installing. Use a supported GPU with ray tracing for camera observations.

.. include:: _setup_common.rst

The simulator and model need different versions of LeRobot and Hugging Face Hub.
From the RLinf checkout, the following command installs the model in ``.venv``
and the simulator in ``.venv-behavior``. Activating the model environment sets
``BEHAVIOR_ENV_PYTHON`` to the simulator's interpreter.

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env behavior --no-flash-attn
   source .venv/bin/activate

For OpenVLA-OFT, pass ``--model openvla-oft`` and omit ``--no-flash-attn``
to install the model's attention backend. Both environments use the same Python 3.11
patch release and Ray version so RLinf can start workers across interpreters.
Use interpreter paths without spaces or shell metacharacters. The Docker build
targets ``embodied-behavior-openpi`` and ``embodied-behavior-openvlaoft`` use the
same installer. Mount the model weights and simulation assets at runtime.

.. code-block:: bash

   docker build -f docker/Dockerfile --build-arg BUILD_TARGET=embodied-behavior-openpi -t rlinf:behavior-openpi .

To reuse an existing installation of the pinned SDK, prepare the two empty
environments with its Python interpreter *before* running the installer above.

.. code-block:: bash

   export BEHAVIOR_PATH=/path/to/BEHAVIOR-1K
   uv venv --python /path/to/sdk/bin/python --system-site-packages .venv-behavior
   uv venv --python /path/to/sdk/bin/python .venv

For an application that only needs the simulator, install the environment with
``bash requirements/install.sh embodied --env behavior --venv .venv-behavior --no-flash-attn``.
The installer builds from a separate SDK checkout and pins LeRobot to
``436812bd8ee39b768c645c248c91f1330834e687`` to support Python 3.11.
Use ``--no-root`` only if system libraries are already installed.

To audit the simulator's dependencies, including packages inherited from an existing
SDK, run ``"$BEHAVIOR_ENV_PYTHON" -m pip check``. With the pinned stack the command
reports four metadata conflicts and exits with status 1. Isaac Sim pins older
``packaging``, ``websockets``, and ``wrapt`` than its co-installed dependencies allow,
and OmniGibson requests Pillow 11.0.x while Isaac Sim uses 11.3.0.

Download the Assets
-------------------

Run the SDK's asset downloader with the simulator interpreter. The data root must
contain the robot assets, BEHAVIOR-1K assets, dataset key, and the 2026 task instances.
The pip installation of Isaac Sim locates its own runtime. Set ``ISAAC_PATH`` only for
a separately installed, matching Isaac Sim.

.. code-block:: bash

   export OMNI_KIT_ACCEPT_EULA=YES
   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   "$BEHAVIOR_ENV_PYTHON" -m omnigibson.utils.asset_utils \
     --download_omnigibson_robot_assets --download_behavior_1k_assets \
     --download_2026_challenge_task_instances --accept_license

On the self-hosted CI runner, stage these assets in ``/workspace/dataset/behavior-datasets``
or set the repository variable ``BEHAVIOR_DATA_PATH`` to their absolute location.

Download the Model
------------------

The standalone π₀.₅ evaluation accepts a PyTorch checkpoint with BEHAVIOR
normalization statistics. A public checkpoint is available from
`RLinf on Hugging Face <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_.

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 \
     --local-dir /path/to/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32

For the other training recipes, download ``RLinf/RLinf-OpenVLAOFT-Behavior``
or ``RLinf/RLinf-Pi0-Behavior`` for the corresponding model.

.. include:: _model_path.rst

Run It
------

Set the actor and rollout model paths in your training config, activate the model
environment, and keep ``OMNIGIBSON_DATA_PATH`` set. For π₀.₅ PPO, run:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh behavior_ppo_openpi_pi05

The other training configs are ``behavior_ppo_openvlaoft`` and ``behavior_ppo_openpi``.
The env workers use the simulator interpreter to step OmniGibson, while the actor
and rollout workers run in the model environment. For a standalone evaluation or
a 64-step radio run, follow the :doc:`BEHAVIOR evaluation guide <../../evaluations/guides/behavior>`.

Prior BEHAVIOR training runs reported that OpenVLA-OFT and π₀ could remain at zero
on ``env/success_once``. These results predate the 3.9.3 simulator update.

Configure Further
-----------------

``examples/embodiment/config/env/behavior_r1pro.yaml`` supplies the shared R1Pro configuration.
Set ``omni_config.task.activity_name`` and ``omni_config.scene.scene_model`` together
when changing tasks. The 2026 train split starts at instance 0. Public test IDs
start at 301. ``activity_instance_dir`` selects a custom directory in preference
to ``instance_split``. Use instance files and assets from the same SDK release.

Each environment has its own simulator process by default. This allows partial
resets to leave other scenes untouched. An explicit positive ``num_env_subprocess``
can batch environments in each process when the whole batch resets together.
With ``auto_reset: True``, allocate a process per environment. Enable offload only
for explicitly reset environments. ``torch_num_threads`` limits PyTorch CPU threads.
``skip_intermediate_renders`` saves camera rendering inside action chunks while
preserving per-step physics, rewards, and termination.

To generate local task instances, run the generator from the simulator environment.
Select ``tro_state`` or ``template`` according to the file format your config loads:

.. code-block:: bash

   "$BEHAVIOR_ENV_PYTHON" -m rlinf.envs.sim.behavior.instance_generator \
     --config examples/embodiment/config/env/behavior_r1pro.yaml \
     --output-dir /path/to/instances --output-format tro_state --start-idx 1 --end-idx 10

For an application that owns its simulator process, import ``BehaviorSimulator``
from the same package used by RLinf. Run this example in the main thread of a
dedicated process with the data root set:

.. code-block:: python

   from pathlib import Path
   from hydra import compose, initialize_config_dir
   from rlinf.envs.sim.behavior import BehaviorSimulator

   with initialize_config_dir(
       version_base=None, config_dir=str(Path('examples/embodiment/config/env').resolve())
   ):
       cfg = compose(config_name="behavior_r1pro")
   sim = BehaviorSimulator(cfg, num_envs=1, seed=0)
   obs, info = sim.reset()
   print(sim.get_metrics(), flush=True)
   sim.close()

``reset`` returns observations and info, one entry per environment.
``step(actions)`` accepts finite native actions shaped ``[num_envs, 23]`` and returns
observations, rewards, terminations, truncations, and info.
``observe(render=True)`` refreshes images without advancing physics.
Saved observations, state, camera metadata, and metrics are copied. ``get_robot()``
borrows the native robot until reset or close. Save results before calling
``close()`` as the final operation because Isaac shutdown can terminate the process.


Visualization and Results
-------------------------

Launch TensorBoard to watch training live:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

The key signal to watch is **``env/success_once``** — the task success rate. For every
logged metric, see :doc:`Training metrics <../../reference/metrics>`.

To save evaluation videos, enable them in the config:

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval


For the Behavior experiment, we were inspired by
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_,
with only minor modifications. We thank the authors for releasing their open-source code.
