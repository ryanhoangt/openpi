"""Roll out a pi0-FAST checkpoint on a LIBERO task and record per-token uncertainty.

This is the client half of the INSIGHT-style introspection experiment. It mirrors the
normal LIBERO eval loop (`main.py`) but expects the policy server to have been started with
`--return-uncertainty`, so every `infer()` result carries a `token_uncertainty` array of shape
`[n_tokens, 6]` (columns: entropy, neg_logp, au, eu, alpha_min, alpha_sum -- see
`openpi.models.pi0_fast.UNCERTAINTY_FEATURE_NAMES`).

It runs entirely in the LIBERO (Python 3.8) venv and only writes a `.npz`; plotting happens
separately in the main venv via `scripts/plot_uncertainty.py`, which has matplotlib.

Start the server first, e.g.:

    uv run scripts/serve_policy.py --return-uncertainty \
        policy:checkpoint --policy.config pi0_fast_libero \
        --policy.dir checkpoints/pi0_fast_libero/finetune_pi0fast_libero_90_job/59999

Then, in the LIBERO venv:

    python examples/libero/collect_uncertainty.py --task-suite-name libero_10 --task-id 0 \
        --num-trials 10 --out data/libero/uncertainty/libero10_task0.npz
"""

import collections
import dataclasses
import logging
import pathlib

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# Must match openpi.models.pi0_fast.UNCERTAINTY_FEATURE_NAMES (that package is not importable here).
FEATURE_NAMES = ("entropy", "neg_logp", "au", "eu", "alpha_min", "alpha_sum")

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_10"
    task_id: int = 0  # single task within the suite
    num_trials: int = 10
    num_steps_wait: int = 10
    seed: int = 7

    # Output .npz path. Per-token features are stored in a flat, columnar layout so grouping by
    # episode / inference-step is trivial downstream.
    out: str = "data/libero/uncertainty/uncertainty.npz"


def _get_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat):
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def collect(args: Args) -> None:
    np.random.seed(args.seed)
    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = MAX_STEPS[args.task_suite_name]

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    logging.info(f"Task {args.task_id}: {task_description}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Flat columnar accumulators over every decoded token of the whole run.
    feats, episode_id, infer_step, env_step, token_pos = [], [], [], [], []
    episode_success = []

    for ep in tqdm.tqdm(range(args.num_trials), desc=f"task {args.task_id}"):
        env.reset()
        obs = env.set_init_state(initial_states[ep])
        action_plan = collections.deque()
        t = 0
        ep_infer_step = 0
        done = False

        while t < max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                if not action_plan:
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    element = {
                        "observation/image": image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        ),
                        "observation/wrist_image": image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        ),
                        "observation/state": np.concatenate(
                            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                        "prompt": str(task_description),
                    }

                    result = client.infer(element)
                    if "token_uncertainty" not in result:
                        raise RuntimeError(
                            "Server did not return 'token_uncertainty'. Start serve_policy.py with "
                            "--return-uncertainty."
                        )
                    u = np.asarray(result["token_uncertainty"], dtype=np.float32)  # [n_tokens, 6]
                    n = u.shape[0]
                    feats.append(u)
                    episode_id.append(np.full(n, ep, dtype=np.int32))
                    infer_step.append(np.full(n, ep_infer_step, dtype=np.int32))
                    env_step.append(np.full(n, t, dtype=np.int32))
                    token_pos.append(np.arange(n, dtype=np.int32))
                    ep_infer_step += 1

                    action_chunk = result["actions"]
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    break
                t += 1
            except Exception as e:
                logging.error(f"Caught exception: {e}")
                break

        episode_success.append(bool(done))
        logging.info(f"Episode {ep}: {'success' if done else 'failure'} ({ep_infer_step} inference steps)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        features=np.concatenate(feats, axis=0),
        episode_id=np.concatenate(episode_id),
        infer_step=np.concatenate(infer_step),
        env_step=np.concatenate(env_step),
        token_pos=np.concatenate(token_pos),
        episode_success=np.asarray(episode_success, dtype=bool),
        feature_names=np.asarray(FEATURE_NAMES),
        task_suite_name=np.asarray(args.task_suite_name),
        task_id=np.asarray(args.task_id),
        task_description=np.asarray(str(task_description)),
    )
    n_succ = int(np.sum(episode_success))
    logging.info(f"Saved {out}  ({args.num_trials} episodes, {n_succ} success / {args.num_trials - n_succ} failure)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    collect(tyro.cli(Args))
