"""
Cooking Experiment 7: human/environment completes garlic for
Qwen2.5-VL + Franka Panda + MuJoCo + manual strategy-template shielding.

Experiment
----------
The normal partial-order recipe begins with seven required ingredients. After the
first accepted and executed robot action, the environment moves garlic into the
mixing bowl without adding a corresponding robot command to the prompt history.
The task state used by the shield comes from the current MuJoCo scene, not from the
list of previous robot commands. This tests whether Qwen can use the current image
when the image and command history differ.

Example intended trajectory:
    1. Qwen/robot adds onion.
    2. Environment/human completes garlic by moving it into the bowl.
    3. The active obligation should become tomato, even though the previous command
       history contains only onion.

If Qwen happens to add garlic as the first accepted action, the disturbance is logged
as already satisfied because garlic is already in the bowl. In that case, rerun for a
cleaner history/image-conflict trial where the first robot action is onion.

Crucially, the shield never executes a replacement action on the model's behalf.
Only a Qwen-returned action that passes the template is executed.

Run this file from the Franka directory containing panda.xml, after copying:
    cooking_stars_scene.xml
    cooking_task.py
    cooking_strategy_template_guide.py
    cooking_qwen25_stars.py

Example:
    python -u cooking_qwen25_exp07_human_completed_garlic.py \
      --planner qwen_ollama \
      --shield-mode template_guidance \
      --ollama-url http://volta13:11434 \
      --ollama-model qwen2.5vl:3b \
      --ollama-timeout 1800 \
      --ollama-http-retries 0 \
      --ollama-num-predict 1024 \
      --ollama-keep-alive 2h \
      --no-ollama-think \
      --shield-gamma 0.20 \
      --shield-theta-prune 0.10 \
      --shield-guidance-threshold 0.95 \
      --max-steps 50 \
      --viewer

There is no fixed reprompt limit for well-formed but template-noncompliant model
proposals. The shield keeps rejecting such proposals, updating counters and
probabilities, and reprompting until Qwen returns a compliant action. Once an
individual live action crosses the effective theta_guide, that action is exposed
as explicit prompt guidance. Actual model-serving failures or malformed/invalid
model output still fail the run. No oracle/deterministic fallback is used.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from cooking_automaton import (
    FINISH,
    NO_MOTION,
    UNCONSTRAINED,
    UNSAFE,
    CookingAutomaton,
)

from cooking_unshielded_policy import assess_unshielded_action

from cooking_strategy_template_guide import (
    ManualCookingStrategyTemplateGuide,
    TemplateSnapshot,
    snapshot_to_jsonable,
)
from cooking_task import (
    ACTION_LABELS,
    BOWL_CENTER,
    INGREDIENTS,
    INGREDIENT_ORDER,
    TABLE_TOP_Z,
    abstract_state_key,
    active_stage,
    bowl_place_position,
    classify_action,
    get_task_state,
    hard_unsafe_actions,
    print_task_state,
    reset_ingredients,
    set_body_pose,
    task_complete_from_state,
)


BASE_SCENE_PATH = Path("cooking_stars_scene.xml").resolve()
SCENE_PATH = BASE_SCENE_PATH
CAMERA_NAME = "table_cam"

ENV_COMPLETED_ITEM = "garlic"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

VALID_ACTIONS = {"add", "stir", "finish"}
VALID_ITEMS = set(INGREDIENT_ORDER)


def ensure_exp07_scene() -> Path:
    """Experiment 7 reuses the existing cooking scene without adding new objects."""
    if not BASE_SCENE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BASE_SCENE_PATH}. Experiment 7 reuses the existing cooking scene."
        )
    return BASE_SCENE_PATH


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def append_jsonl(path: Optional[Path], record: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def all_names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> List[str]:
    names: List[str] = []
    for index in range(count):
        name = mujoco.mj_id2name(model, obj_type, index)
        if name is not None:
            names.append(name)
    return names


def find_end_effector_ref(model: mujoco.MjModel) -> Tuple[str, int, str]:
    site_names = all_names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    body_names = all_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)

    for wanted in ("pinch", "ee", "ee_site", "tcp", "gripper", "attachment_site", "hand"):
        for name in site_names:
            if name == wanted or wanted in name.lower():
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                print(f"Using end-effector site: {name}")
                return "site", site_id, name

    for wanted in ("hand", "panda_hand", "link7", "panda_link7"):
        for name in body_names:
            if name == wanted or wanted in name.lower():
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                print(f"Using end-effector body: {name}")
                return "body", body_id, name

    print("Available sites:", site_names)
    print("Available bodies:", body_names)
    raise RuntimeError("Could not find an end-effector site/body.")


def find_arm_joints(model: mujoco.MjModel) -> List[int]:
    preferred = [f"joint{i}" for i in range(1, 8)]
    joint_ids: List[int] = []

    for name in preferred:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            joint_ids.append(jid)

    if len(joint_ids) == 7:
        print("Using arm joints:", preferred)
        return joint_ids

    joint_ids = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None or "finger" in name.lower():
            continue
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            joint_ids.append(jid)
        if len(joint_ids) == 7:
            break

    if len(joint_ids) != 7:
        print("Available joints:", all_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt))
        raise RuntimeError("Could not find 7 Franka arm joints.")

    print(
        "Using fallback arm joints:",
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in joint_ids],
    )
    return joint_ids


def find_gripper_joints(model: mujoco.MjModel) -> List[int]:
    joints: List[int] = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is not None and "finger" in name.lower():
            joints.append(jid)

    if joints:
        print(
            "Using gripper joints:",
            [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in joints],
        )
    else:
        print("No gripper joints found. Gripper open/close will be virtual only.")
    return joints


# -----------------------------------------------------------------------------
# Scripted Franka executor
# -----------------------------------------------------------------------------


class FrankaCookingSimulator:
    """Scripted kinematic executor: add ingredient, stir, return home."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        viewer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.data = data
        self.viewer = viewer

        self.ee_kind, self.ee_id, self.ee_name = find_end_effector_ref(model)
        self.arm_joint_ids = find_arm_joints(model)
        self.gripper_joint_ids = find_gripper_joints(model)

        self.arm_qpos_addrs = np.array(
            [model.jnt_qposadr[jid] for jid in self.arm_joint_ids], dtype=int
        )
        self.arm_dof_addrs = np.array(
            [model.jnt_dofadr[jid] for jid in self.arm_joint_ids], dtype=int
        )
        self.gripper_qpos_addrs = np.array(
            [model.jnt_qposadr[jid] for jid in self.gripper_joint_ids], dtype=int
        )
        self.gripper_dof_addrs = np.array(
            [model.jnt_dofadr[jid] for jid in self.gripper_joint_ids], dtype=int
        )

        self.home_qpos = np.array(
            [0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8], dtype=float
        )

    def sync(self, sleep: float = 0.01) -> None:
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
        time.sleep(sleep)

    def ee_pos(self) -> np.ndarray:
        if self.ee_kind == "site":
            return self.data.site_xpos[self.ee_id].copy()
        return self.data.xpos[self.ee_id].copy()

    def ee_jacobian(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        if self.ee_kind == "site":
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        return jacp[:, self.arm_dof_addrs]

    def set_gripper(self, opening: float, steps: int = 60) -> None:
        if len(self.gripper_joint_ids) == 0:
            return
        start = self.data.qpos[self.gripper_qpos_addrs].copy()
        target = np.ones_like(start) * opening
        for alpha in np.linspace(0.0, 1.0, steps):
            self.data.qpos[self.gripper_qpos_addrs] = (1 - alpha) * start + alpha * target
            self.data.qvel[self.gripper_dof_addrs] = 0.0
            self.sync()

    def open_gripper(self) -> None:
        print("Opening gripper")
        self.set_gripper(0.04)

    def close_gripper(self) -> None:
        print("Closing gripper")
        self.set_gripper(0.0)

    def update_held_object(self, held_item: Optional[str]) -> None:
        if held_item is None:
            return
        ee = self.ee_pos()
        half_height = float(INGREDIENTS[held_item]["half_height"])
        object_pos = ee + np.array([0.0, 0.0, -(half_height + 0.055)])
        set_body_pose(self.model, self.data, held_item, object_pos)

    def move_ee_to(
        self,
        target_pos: Sequence[float],
        held_item: Optional[str] = None,
        steps: int = 400,
        tolerance: float = 0.012,
    ) -> None:
        target = np.asarray(target_pos, dtype=float)
        for _ in range(steps):
            current = self.ee_pos()
            error = target - current
            if np.linalg.norm(error) < tolerance:
                break

            jacobian = self.ee_jacobian()
            damping = 1e-3
            dq = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(3), error
            )

            max_step = 0.035
            norm = np.linalg.norm(dq)
            if norm > max_step:
                dq = dq * (max_step / norm)

            self.data.qpos[self.arm_qpos_addrs] += dq
            self.data.qvel[self.arm_dof_addrs] = 0.0

            for idx, jid in enumerate(self.arm_joint_ids):
                if self.model.jnt_limited[jid]:
                    low, high = self.model.jnt_range[jid]
                    addr = self.arm_qpos_addrs[idx]
                    self.data.qpos[addr] = np.clip(self.data.qpos[addr], low, high)

            self.update_held_object(held_item)
            self.sync()

        self.update_held_object(held_item)
        self.sync()

    def return_home(self) -> None:
        print("\nReturning Franka to home pose")
        start = self.data.qpos[self.arm_qpos_addrs].copy()
        for alpha in np.linspace(0.0, 1.0, 160):
            self.data.qpos[self.arm_qpos_addrs] = (1 - alpha) * start + alpha * self.home_qpos
            self.data.qvel[self.arm_dof_addrs] = 0.0
            self.sync()

    def object_position(self, item: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, item)
        if body_id < 0:
            raise ValueError(f"Unknown ingredient body: {item}")
        return self.data.xpos[body_id].copy()

    def pick(self, item: str) -> None:
        print(f"\nPicking {item}")
        obj_pos = self.object_position(item)
        above = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.24])
        grasp = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.095])
        lift = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.28])

        self.open_gripper()
        self.move_ee_to(above)
        self.move_ee_to(grasp)
        self.close_gripper()
        self.update_held_object(item)
        self.move_ee_to(lift, held_item=item)

    def place_in_bowl(self, item: str) -> None:
        print(f"Placing {item} in the mixing bowl")
        place_pos = bowl_place_position(item)
        above = place_pos + np.array([0.0, 0.0, 0.24])
        lower = place_pos + np.array([0.0, 0.0, 0.095])

        self.move_ee_to(above, held_item=item)
        self.move_ee_to(lower, held_item=item)
        self.open_gripper()
        set_body_pose(self.model, self.data, item, place_pos)
        self.move_ee_to(above)

    def add_ingredient(self, item: str) -> None:
        self.pick(item)
        self.place_in_bowl(item)
        self.return_home()

    def stir(self, revolutions: int = 2, points_per_revolution: int = 28) -> None:
        """Scripted repeatable co-live action: circle the end-effector above the bowl."""
        print("\nStirring the mixing bowl")
        center = np.array([BOWL_CENTER[0], BOWL_CENTER[1], TABLE_TOP_Z + 0.25])
        radius = 0.10
        start = center + np.array([radius, 0.0, 0.0])
        self.move_ee_to(start)

        total_points = revolutions * points_per_revolution
        for angle in np.linspace(0.0, 2.0 * np.pi * revolutions, total_points):
            target = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
            self.move_ee_to(target, steps=45, tolerance=0.018)
        self.return_home()


# -----------------------------------------------------------------------------
# Image + planner action helpers
# -----------------------------------------------------------------------------


def render_camera_image(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    image_path: Path,
) -> Path:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if camera_id >= 0:
        renderer.update_scene(data, camera=CAMERA_NAME)
    else:
        renderer.update_scene(data)
    image = renderer.render()
    Image.fromarray(image).save(image_path)
    return image_path


def extract_json_object(text: str) -> Dict[str, Any]:
    if text is None:
        raise ValueError("Planner response was None.")
    raw = text.strip()
    if not raw:
        raise ValueError("Planner returned an empty response.")

    decoder = json.JSONDecoder()
    try:
        parsed, end_index = decoder.raw_decode(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Planner must return exactly one valid JSON object and nothing else. "
            f"JSON parsing failed at character {exc.pos}: {exc.msg}. Raw response:\n{raw}"
        ) from exc

    trailing = raw[end_index:].strip()
    if trailing:
        raise ValueError(
            "Planner must return exactly one JSON object and nothing else. "
            f"Unexpected trailing content:\n{trailing}"
        )
    if not isinstance(parsed, dict):
        raise ValueError(f"Planner JSON must be an object, got {type(parsed).__name__}.")
    return parsed


def validate_planner_action(action: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action, dict):
        raise ValueError("Planner action must be a JSON object.")

    action_name = action.get("action")
    if action_name == "done":
        action_name = "finish"
    if action_name not in VALID_ACTIONS:
        raise ValueError(f"Invalid action {action_name!r}. Expected {sorted(VALID_ACTIONS)}")

    if action_name == "add":
        item = action.get("item")
        if item not in VALID_ITEMS:
            raise ValueError(f"Unknown ingredient {item!r}. Expected {sorted(VALID_ITEMS)}")
        return {"action": "add", "item": item}

    if action_name == "stir":
        return {"action": "stir"}

    return {"action": "finish"}


def action_to_label(action: Dict[str, Any]) -> str:
    if action["action"] == "add":
        return f"add:{action['item']}"
    return str(action["action"])


def format_previous_commands(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "None."
    lines: List[str] = []
    for index, action in enumerate(history, start=1):
        if action["action"] == "add":
            text = f"Add {action['item']} to the mixing bowl."
        elif action["action"] == "stir":
            text = "Stir the mixing bowl."
        else:
            text = "Finish the recipe."
        lines.append(f"{index}. {text}")
    return "\n".join(lines)


def make_planner_prompt(
    history: List[Dict[str, Any]],
    guidance_text: str = "",
    rejection_text: str = "",
) -> str:
    previous_commands = format_previous_commands(history)
    shield_section = ""
    if rejection_text or guidance_text:
        shield_section = "\n\n" + "\n\n".join(
            part for part in (rejection_text, guidance_text) if part
        )

    return f"""
You control a Franka robot arm performing a symbolic cooking task in MuJoCo.
Use ONLY the current image plus the task rules below to choose the next high-level action.

VISUAL LEGEND:
- Purple sphere: onion.
- Off-white horizontal capsule: garlic.
- Red sphere: tomato.
- Yellow cylinder: turmeric.
- Green cylinder: coriander powder.
- White cylinder with blue cap: salt.
- Blue-and-white tall box: cream.
- Large blue circular region in the center: mixing bowl.
- Light-gray surface around it: table.
An ingredient counts as added only when it is visibly inside the blue mixing-bowl region.

RECIPE CONSTRAINTS:
Stage 1: add onion and garlic. Either order is allowed.
Stage 2: after BOTH onion and garlic are added, add tomato.
Stage 3: after tomato is added, add turmeric and coriander powder. These core spices may be added in either order.
Salt rule: add at least one small salt increment during Stage 3 or after cream, but before finish. Up to three salt increments may be added in total.
Stage 4: after all required core spices are added, add cream. Salt may be added before or after cream.
Stage 5: finish only after onion, garlic, tomato, all required core spices, cream, and at least one salt increment are complete.
The action `stir` may be used temporarily, but stirring alone does not satisfy any missing ingredient obligation.
Choose only one next action. Do not output a full plan.

Previous ACCEPTED high-level commands:
{previous_commands}{shield_section}

Return exactly one JSON object and nothing else. Include a concise `reasoning` field for experiment logging.

To add an ingredient:
{{"reasoning": "<one to three concise sentences>", "action": "add", "item": "<ingredient>"}}

To stir:
{{"reasoning": "<one to three concise sentences>", "action": "stir", "item": null}}

When the recipe is complete:
{{"reasoning": "<one to three concise sentences>", "action": "finish", "item": null}}
""".strip()


# -----------------------------------------------------------------------------
# Planner backends
# -----------------------------------------------------------------------------


class PlannerRunFailure(RuntimeError):
    """A model request or unusable planner output that terminates the run."""


class DecisionBudgetExhausted(PlannerRunFailure):
    """The configured global model-decision budget was consumed."""


class BasePlanner:
    def request_action(
        self,
        image_path: Path,
        prompt: str,
        trace_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class ManualPlanner(BasePlanner):
    def request_action(
        self,
        image_path: Path,
        prompt: str,
        trace_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        del trace_context
        print("\nPlanner image:", image_path)
        print("\nPROMPT SENT TO MANUAL PLANNER:\n")
        print(prompt)
        print("\nExamples:")
        print('{"action": "add", "item": "onion"}')
        print('{"action": "stir"}')
        print('{"action": "finish"}')
        raw = input("planner_json> ").strip()
        parsed = extract_json_object(raw)
        return validate_planner_action(parsed)


class QwenOllamaPlanner(BasePlanner):
    TRANSIENT_HTTP_CODES = {502, 503, 504}

    def __init__(
        self,
        base_url: str,
        model_id: str,
        timeout: int,
        http_retries: int,
        retry_backoff: float,
        num_predict: int,
        keep_alive: str,
        think: Optional[bool],
        trace_jsonl: Optional[str],
        run_id: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_url = self.base_url + "/api/chat"
        self.model_id = model_id
        self.timeout = timeout
        self.http_retries = max(0, http_retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.num_predict = max(64, num_predict)
        self.keep_alive = keep_alive
        self.think = False if think is None else bool(think)
        self.trace_jsonl = Path(trace_jsonl).resolve() if trace_jsonl else None
        self.run_id = run_id
        self.request_index = 0

    def _call_qwen(self, prompt: str, image_path: Path) -> Dict[str, Any]:
        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()

        action_schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "action": {"type": "string", "enum": ["add", "stir", "finish"]},
                "item": {"enum": [None, *sorted(VALID_ITEMS)]},
            },
            "required": ["reasoning", "action", "item"],
            "additionalProperties": False,
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "format": action_schema,
            "options": {
                "temperature": 0,
                "num_predict": self.num_predict,
            },
        }

        message_count = len(payload["messages"])
        image_count = sum(len(message.get("images", [])) for message in payload["messages"])
        if message_count != 1 or image_count != 1:
            raise RuntimeError(
                f"ONE-IMAGE INVARIANT VIOLATED: messages={message_count}, images={image_count}"
            )

        request_data = json.dumps(payload).encode("utf-8")
        print(
            "REQUEST DIAGNOSTICS: "
            f"messages={message_count}, images={image_count}, "
            f"prompt_chars={len(prompt)}, image_bytes={len(image_bytes)}, "
            f"payload_mb={len(request_data) / (1024 * 1024):.2f}",
            flush=True,
        )

        total_attempts = self.http_retries + 1
        started = time.perf_counter()
        response_body = ""

        for attempt_index in range(total_attempts):
            attempt_no = attempt_index + 1
            request = urllib.request.Request(
                self.api_url,
                data=request_data,
                headers={
                    "Content-Type": "application/json",
                    "X-User-Id": "sshukla82",
                    "Connection": "keep-alive",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_body = response.read().decode("utf-8")
                break

            except urllib.error.HTTPError as exc:
                try:
                    server_body = exc.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    server_body = ""
                retries_left = attempt_index < self.http_retries
                if exc.code in self.TRANSIENT_HTTP_CODES and retries_left:
                    delay = min(self.retry_backoff * (2 ** attempt_index), 120.0)
                    print(
                        f"\nMODEL REQUEST TRANSIENT FAILURE: HTTP {exc.code} "
                        f"(attempt {attempt_no}/{total_attempts})"
                    )
                    print(server_body or "<empty server response>")
                    print(f"Retrying exact request in {delay:.0f} seconds.")
                    time.sleep(delay)
                    continue
                print(f"\nMODEL REQUEST FAILED: HTTP {exc.code}")
                print(server_body or "<empty server response>")
                raise PlannerRunFailure(f"HTTP {exc.code}") from exc

            except (TimeoutError, socket.timeout) as exc:
                retries_left = attempt_index < self.http_retries
                if retries_left:
                    delay = min(self.retry_backoff * (2 ** attempt_index), 120.0)
                    print(
                        f"\nMODEL REQUEST TRANSIENT FAILURE: timeout "
                        f"(attempt {attempt_no}/{total_attempts})"
                    )
                    print(f"Retrying exact request in {delay:.0f} seconds.")
                    time.sleep(delay)
                    continue
                print("\nMODEL REQUEST FAILED: timeout")
                print(f"No response within {self.timeout} seconds.")
                raise PlannerRunFailure("timeout") from exc

            except urllib.error.URLError as exc:
                retries_left = attempt_index < self.http_retries
                if retries_left:
                    delay = min(self.retry_backoff * (2 ** attempt_index), 120.0)
                    print(
                        f"\nMODEL REQUEST TRANSIENT NETWORK FAILURE "
                        f"(attempt {attempt_no}/{total_attempts})"
                    )
                    print(f"Reason: {exc.reason}")
                    print(f"Retrying exact request in {delay:.0f} seconds.")
                    time.sleep(delay)
                    continue
                print("\nMODEL REQUEST FAILED: network/connection error")
                print(f"Reason: {exc.reason}")
                raise PlannerRunFailure(f"network/connection error: {exc.reason}") from exc

        latency_seconds = time.perf_counter() - started
        try:
            output = json.loads(response_body)
        except json.JSONDecodeError as exc:
            print("\nMODEL REQUEST FAILED: server returned invalid JSON")
            print(response_body)
            raise PlannerRunFailure("invalid JSON from Ollama server") from exc

        return {
            "ollama_response": output,
            "prompt": prompt,
            "image_path": str(image_path.resolve()),
            "image_sha256": image_sha256,
            "image_count": image_count,
            "message_count": message_count,
            "payload_bytes": len(request_data),
            "latency_seconds": latency_seconds,
        }

    def request_action(
        self,
        image_path: Path,
        prompt: str,
        trace_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.request_index += 1
        result = self._call_qwen(prompt, image_path)
        output = result["ollama_response"]
        message = output.get("message", {}) or {}
        content = (message.get("content") or "").strip()
        server_thinking = (message.get("thinking") or "").strip()
        done_reason = output.get("done_reason")

        print("\n" + "=" * 72)
        print("MODEL-RETURNED SERVER THINKING FIELD")
        print("=" * 72)
        print(server_thinking if server_thinking else "<no separate thinking field returned>")
        print("=" * 72)
        print("RAW MODEL CONTENT")
        print("=" * 72)
        print(content if content else "<empty content>")

        trace_record: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "request_index": self.request_index,
            "model": self.model_id,
            "image_path": result["image_path"],
            "image_sha256": result["image_sha256"],
            "image_count": result["image_count"],
            "message_count": result["message_count"],
            "payload_bytes": result["payload_bytes"],
            "prompt": prompt,
            "trace_context": trace_context,
            "server_thinking": server_thinking,
            "raw_model_content": content,
            "raw_ollama_response": output,
            "done_reason": done_reason,
            "prompt_eval_count": output.get("prompt_eval_count"),
            "eval_count": output.get("eval_count"),
            "latency_seconds": result["latency_seconds"],
            "error": None,
        }

        try:
            if done_reason == "length" and not content:
                raise ValueError(
                    f"generation budget ended before final JSON (num_predict={self.num_predict})"
                )

            parsed = extract_json_object(content)
            explicit_reasoning = parsed.get("reasoning")
            if not isinstance(explicit_reasoning, str) or not explicit_reasoning.strip():
                raise ValueError("missing or empty required 'reasoning' field")

            action = validate_planner_action(parsed)
            trace_record["explicit_reasoning"] = explicit_reasoning.strip()
            trace_record["parsed_action"] = action
            append_jsonl(self.trace_jsonl, trace_record)

            print("=" * 72)
            print("EXPLICIT MODEL-GENERATED RATIONALE")
            print("=" * 72)
            print(explicit_reasoning.strip())
            print("=" * 72)
            print("PARSED MODEL ACTION")
            print("=" * 72)
            print(json.dumps(action, indent=2))
            print("=" * 72)
            print(f"TRACE JSONL: {self.trace_jsonl if self.trace_jsonl else '<disabled>'}")
            return action

        except Exception as exc:
            trace_record["error"] = f"invalid or unusable planner output: {exc}"
            append_jsonl(self.trace_jsonl, trace_record)
            print("\nMODEL OUTPUT FAILED: invalid or unusable planner action")
            print(f"Reason: {exc}")
            print("Stopping this run.")
            raise PlannerRunFailure(f"invalid model output: {exc}") from exc


# -----------------------------------------------------------------------------
# Shield logging + prompt loop
# -----------------------------------------------------------------------------


def print_shield_snapshot(snapshot: TemplateSnapshot) -> None:
    print("\n" + "=" * 72)
    print("AUTOMATON-BACKED STRATEGY-TEMPLATE SHIELD STATE")
    print("=" * 72)
    print("Automaton state ID:", snapshot.automaton_state_id)
    print("Automaton state key:", snapshot.state_key)
    print("Cooking stage:", snapshot.stage_name)
    print("Unsafe S:", list(snapshot.unsafe_actions))
    print("Co-live D:", list(snapshot.colive_actions))
    print("Unconstrained U:", list(snapshot.unconstrained_actions))
    print("Live H_l:", list(snapshot.live_actions))
    print("Salt additions:", snapshot.salt_add_count)
    print("Live-neglect counter:", snapshot.live_counter)
    print("Co-live counters:", snapshot.colive_counters)
    print(
        "Repeated state-preserving unconstrained counters:",
        snapshot.unconstrained_self_loop_counters,
    )
    print("Base distribution after unsafe=0 + normalization:")
    print(json.dumps(snapshot.base_distribution, indent=2))
    print("Pre-prune distribution after liveness/co-liveness/self-loop shaping:")
    print(json.dumps(snapshot.pre_prune_distribution, indent=2))
    print(
        "Theta prune (co-live + repeated state-preserving unconstrained):",
        snapshot.theta_prune,
    )
    print("Actions pruned to probability 0:", list(snapshot.pruned_actions))
    print("Final shield-side distribution:")
    print(json.dumps(snapshot.modified_distribution, indent=2))
    print("Total live-set probability mass:", f"{snapshot.live_group_probability_mass:.6f}")
    print("Theta guidance:", snapshot.guidance_threshold)
    print("Non-answer-revealing progress directive active:", snapshot.directive_active)
    print("=" * 72)


def get_model_action_with_template_guidance(
    args: argparse.Namespace,
    planner: BasePlanner,
    guide: Optional[ManualCookingStrategyTemplateGuide],
    automaton: CookingAutomaton,
    image_path: Path,
    task_state: Dict[str, Dict[str, Any]],
    history: List[Dict[str, Any]],
    model_decision_history: List[Dict[str, Any]],
    decision_budget: Dict[str, int],
    shield_trace_path: Optional[Path],
    executed_step_index: int,
) -> Tuple[Dict[str, Any], bool, int]:
    """Return one accepted model action without selecting an action for the VLM.

    Every parsed model response consumes the global decision budget. In shielded
    mode an unsafe or theta-pruned proposal is rejected and reprompted, up to
    the configured consecutive reprompt limit. Threshold guidance is a firm
    progress directive and never lists the correct live action.
    """
    shield_mode = args.shield_mode
    rejection_text = ""
    guidance_text = guide.guidance_text(guide.snapshot(task_state)) if guide is not None else ""
    attempts = 1 if shield_mode == "off" else args.shield_reprompt_limit + 1

    for attempt_index in range(attempts):
        if decision_budget["count"] >= decision_budget["limit"]:
            raise DecisionBudgetExhausted(
                f"global model-decision budget of {decision_budget['limit']} was exhausted"
            )

        prompt = make_planner_prompt(
            history=history,
            guidance_text=guidance_text,
            rejection_text=rejection_text,
        )

        trace_context: Dict[str, Any] = {
            "accepted_step_index": executed_step_index,
            "attempt_index_within_step": attempt_index,
            "shield_mode": shield_mode,
            "ground_truth_task_state_for_analysis_only": task_state,
            "automaton_snapshot_before_model_request": automaton.snapshot_jsonable(task_state),
            "shield_snapshot_before_model_request": (
                snapshot_to_jsonable(guide.snapshot(task_state)) if guide is not None else None
            ),
            "rejection_text_present": bool(rejection_text),
            "guidance_text_present": bool(guidance_text),
        }

        action = planner.request_action(
            image_path=image_path,
            prompt=prompt,
            trace_context=trace_context,
        )
        decision_budget["count"] += 1
        decision_number = decision_budget["count"]

        label = action_to_label(action)
        transition = automaton.transition(task_state, label)
        classification = transition.classification
        decision_record: Dict[str, Any] = {
            "decision_step": decision_number,
            "accepted_step_slot": executed_step_index + 1,
            "attempt_index_within_step": attempt_index,
            "action": dict(action),
            "action_label": label,
            "classification": classification,
            "automaton_state_id": automaton.node(task_state).state_id,
            "target_state_id": transition.target_state_id,
            "physical_effect": transition.physical_effect,
            "template_compliant": classification != UNSAFE,
            "shield_allowed": None,
            "executed_in_simulation": False,
            "state_changed": False,
            "violation": False,
            "violation_reason": None,
            "skip_reason": None,
        }
        model_decision_history.append(decision_record)

        if shield_mode == "off" or guide is None:
            return action, classification != UNSAFE, decision_number

        check = guide.check_action(task_state, label)
        decision_record["shield_allowed"] = check.allowed
        decision_record["shield_reason"] = check.reason

        print("\n" + "=" * 72)
        print("AUTOMATON/STRATEGY-TEMPLATE CHECK OF MODEL PROPOSAL")
        print("=" * 72)
        print("Model decision number:", decision_number)
        print("Model action:", json.dumps(action))
        print("Action label:", label)
        print("Classification:", check.classification)
        print("Physical effect:", check.physical_effect)
        print("Allowed:", check.allowed)
        print("Reason:", check.reason)
        print("=" * 72)

        shield_record: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "model_decision_number": decision_number,
            "accepted_step_index": executed_step_index,
            "attempt_index_within_step": attempt_index,
            "model_action": action,
            "action_label": label,
            "classification": check.classification,
            "physical_effect": check.physical_effect,
            "target_state_id": check.target_state_id,
            "allowed": check.allowed,
            "reason": check.reason,
            "snapshot_before_update": snapshot_to_jsonable(check.snapshot_before_update),
        }

        if check.allowed:
            decision_record["outcome"] = "model_action_allowed_unchanged"
            shield_record["outcome"] = "model_action_allowed_unchanged"
            append_jsonl(shield_trace_path, shield_record)
            return action, True, decision_number

        decision_record["outcome"] = "rejected_and_reprompted"
        print("\n*** MODEL ACTION REJECTED BY AUTOMATON-BACKED SHIELD ***")
        print("Robot action executed: NONE")
        print("The shield is not replacing the model action.")

        snapshot_after = guide.observe_rejected_attempt(task_state, label)
        print_shield_snapshot(snapshot_after)
        rejection_text = guide.rejection_text(check)
        # The answer-revealing rejection reprompt replaces the generic threshold
        # directive for this retry, avoiding contradictory prompt instructions.
        guidance_text = ""

        shield_record["outcome"] = "rejected_and_reprompted"
        shield_record["snapshot_after_update"] = snapshot_to_jsonable(snapshot_after)
        shield_record["progress_directive_active"] = bool(guidance_text)
        append_jsonl(shield_trace_path, shield_record)

        if guidance_text:
            print("\n*** NON-ANSWER-REVEALING PROGRESS DIRECTIVE ACTIVE ***")
            print("The model is required to infer a progress action from the image and recipe.")
        else:
            print("\nGuidance threshold not yet crossed. Reprompting after rejection.")

    raise PlannerRunFailure(
        "model failed to return an automaton-compliant action within "
        f"{args.shield_reprompt_limit} shield reprompts"
    )


# -----------------------------------------------------------------------------
# Controlled disturbance
# -----------------------------------------------------------------------------


def maybe_apply_disturbance(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sim: FrankaCookingSimulator,
    executed_action_count: int,
    already_applied: bool,
) -> bool:
    """After the first executed model action, have the environment complete garlic."""
    del args
    if already_applied or executed_action_count < 1:
        return already_applied

    current_state = get_task_state(model, data)
    print("\n*** COOKING EXPERIMENT 7 DISTURBANCE APPLIED ***")
    print("Mode: environment_completes_garlic")
    print("Trigger: after executed action 1")
    print("Environment/human action: move garlic into the mixing bowl")
    print("Previous-command history changed: False")
    print("Shield state source of truth: current MuJoCo scene")

    if bool(current_state[ENV_COMPLETED_ITEM]["added"]):
        print("Garlic was already in the bowl before disturbance; no physical move needed.")
        print("This can happen if Qwen chose garlic as the first accepted action.")
        sim.return_home()
        return True

    target_pos = bowl_place_position(ENV_COMPLETED_ITEM)
    print("Moving garlic into bowl position:", target_pos.tolist())
    set_body_pose(model, data, ENV_COMPLETED_ITEM, target_pos)
    sim.return_home()
    return True


# -----------------------------------------------------------------------------
# Episode
# -----------------------------------------------------------------------------


def build_planner(args: argparse.Namespace) -> BasePlanner:
    if args.planner == "manual":
        return ManualPlanner()
    if args.planner == "qwen_ollama":
        return QwenOllamaPlanner(
            base_url=args.ollama_url,
            model_id=args.ollama_model,
            timeout=args.ollama_timeout,
            http_retries=args.ollama_http_retries,
            retry_backoff=args.ollama_retry_backoff,
            num_predict=args.ollama_num_predict,
            keep_alive=args.ollama_keep_alive,
            think=args.ollama_think,
            trace_jsonl=args.trace_jsonl,
            run_id=args.run_id,
        )
    raise ValueError(f"Unknown planner: {args.planner}")


def run_episode(args: argparse.Namespace) -> bool:
    ensure_exp07_scene()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    reset_ingredients(model, data, seed=args.seed, jitter_xy=args.reset_jitter_xy)
    mujoco.mj_forward(model, data)

    image_dir = Path(args.image_dir).resolve()
    image_dir.mkdir(parents=True, exist_ok=True)
    shield_trace_path = Path(args.shield_trace_jsonl).resolve() if args.shield_trace_jsonl else None

    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    planner = build_planner(args)

    action_labels = ACTION_LABELS
    automaton = CookingAutomaton(
        action_labels=action_labels,
        always_unsafe_actions=(),
        max_salt_additions=args.salt_max_additions,
    )

    guide = None
    if args.shield_mode == "template_guidance":
        guide = ManualCookingStrategyTemplateGuide(
            gamma=args.shield_gamma,
            theta_prune=args.shield_theta_prune,
            guidance_threshold=args.shield_guidance_threshold,
            action_labels=action_labels,
            automaton=automaton,
            max_salt_additions=args.salt_max_additions,
        )

    # `history` contains accepted high-level commands that affected the physical
    # trajectory or the symbolic salt quantity, plus a final valid finish. It is
    # shown back to the model as "Previous ACCEPTED high-level commands".
    history: List[Dict[str, Any]] = []
    rejected_actions: List[Dict[str, Any]] = []

    # Unshielded observation records are deliberately separate from `history`.
    # Duplicate adds and premature finish recommendations are logged here but are
    # not falsely presented to the model as executed commands on the next step.
    model_decision_history: List[Dict[str, Any]] = []
    violation_history: List[Dict[str, Any]] = []
    skipped_action_history: List[Dict[str, Any]] = []

    model_failure_reason: Optional[str] = None
    termination_reason: Optional[str] = None
    hard_violation_observed = False
    hard_violation_executed = False
    finished = False
    executed_simulation_action_count = 0
    episode_step_limit = (
        args.unshielded_max_steps if args.shield_mode == "off" else args.max_steps
    )
    decision_budget: Dict[str, int] = {"count": 0, "limit": episode_step_limit}
    automaton_nonprogress_history: List[Dict[str, Any]] = []
    disturbance_applied = False

    viewer_cm = mujoco.viewer.launch_passive(model, data) if args.viewer else None

    try:
        viewer = viewer_cm.__enter__() if viewer_cm is not None else None
        sim = FrankaCookingSimulator(model, data, viewer)

        print("\n" + "=" * 72)
        print("COOKING EXPERIMENT 7: HUMAN/ENVIRONMENT COMPLETES GARLIC")
        print("After the first executed robot action, garlic is moved into the bowl by the environment.")
        print("The previous-command history is not updated for this environment action.")
        print("=" * 72)
        print("\nInitial cooking task state:")
        print_task_state(model, data)
        sim.return_home()

        for step_index in range(episode_step_limit):
            if viewer is not None and not viewer.is_running():
                model_failure_reason = "viewer closed before episode completed"
                termination_reason = "viewer_closed"
                break

            mujoco.mj_forward(model, data)
            task_state = get_task_state(model, data)
            automaton_node = automaton.node(task_state)
            stage_name = automaton_node.stage_name
            incomplete = automaton_node.live_actions

            step_kind = "Accepted-decision slot"
            print(f"\n=== {step_kind} step {step_index + 1}/{episode_step_limit} ===")
            print("Current automaton state:", automaton_node.state_id)
            print("Current cooking stage:", stage_name)
            print("Current live actions:", list(incomplete))
            print("Current unconstrained actions:", list(automaton_node.unconstrained_actions))

            if guide is not None:
                print_shield_snapshot(guide.snapshot(task_state))

            image_path = image_dir / f"planner_step_{step_index:03d}.png"
            render_camera_image(model, data, renderer, image_path)
            print("Rendered image:", image_path)

            try:
                action, template_compliant, decision_number = get_model_action_with_template_guidance(
                    args=args,
                    planner=planner,
                    guide=guide,
                    automaton=automaton,
                    image_path=image_path,
                    task_state=task_state,
                    history=history,
                    model_decision_history=model_decision_history,
                    decision_budget=decision_budget,
                    shield_trace_path=shield_trace_path,
                    executed_step_index=step_index,
                )
            except DecisionBudgetExhausted as exc:
                termination_reason = "decision_budget_exhausted"
                print("\nExperiment result: decision budget exhausted")
                print("Reason:", exc)
                break
            except PlannerRunFailure as exc:
                model_failure_reason = str(exc)
                termination_reason = "model_or_output_failure"
                print("\nExperiment result: FAILED")
                print("Failure reason:", model_failure_reason)
                print("Stopping this run.")
                break

            label = action_to_label(action)
            transition = automaton.transition(task_state, label)
            classification = transition.classification
            decision_record = model_decision_history[-1]
            decision_record.update(
                {
                    "classification": classification,
                    "template_compliant": bool(template_compliant),
                    "automaton_state_id": automaton.node(task_state).state_id,
                    "target_state_id": transition.target_state_id,
                    "physical_effect": transition.physical_effect,
                }
            )

            print("\nMODEL DECISION ACCEPTED FOR DISPOSITION:")
            print(json.dumps(action, indent=2))
            print("Automaton classification:", classification)
            print("Physical disposition:", transition.physical_effect)

            if args.shield_mode == "off":
                disposition = assess_unshielded_action(
                    action=action,
                    task_state=task_state,
                    classification=classification,
                    task_complete=automaton.recipe_complete(task_state),
                    previously_executed_add_items={
                        str(previous["item"])
                        for previous in history
                        if previous.get("action") == "add" and "item" in previous
                    },
                )
                decision_record.update(
                    {
                        "violation": disposition.violation,
                        "violation_reason": disposition.violation_reason,
                        "skip_reason": disposition.skip_reason,
                    }
                )

                if disposition.violation:
                    hard_violation_observed = True
                    violation_record = {
                        "decision_step": decision_number,
                        "action": dict(action),
                        "classification": classification,
                        "reason": disposition.violation_reason,
                        "executed_in_simulation": disposition.execute_in_simulation,
                        "automaton_state_before_action": automaton.snapshot_jsonable(task_state),
                    }
                    violation_history.append(violation_record)
                    print("\n*** UNSHIELDED RECIPE VIOLATION OBSERVED ***")
                    print("Violation reason:", disposition.violation_reason)

                if action["action"] == "finish":
                    if disposition.valid_finish:
                        history.append(action)
                        decision_record["outcome"] = "valid_finish"
                        finished = True
                        termination_reason = "valid_finish"
                        print("Model returned finish after all required ingredients were complete.")
                        break

                    decision_record["outcome"] = "premature_finish_no_motion"
                    skipped_record = {
                        "decision_step": decision_number,
                        "action": dict(action),
                        "reason": disposition.skip_reason,
                    }
                    skipped_action_history.append(skipped_record)
                    print("Premature finish recorded; no robot action executed.")
                    print("Continuing unshielded observation.")
                    continue

                if not disposition.execute_in_simulation:
                    skipped_action_history.append(
                        {
                            "decision_step": decision_number,
                            "action": dict(action),
                            "reason": disposition.skip_reason,
                        }
                    )
                    if classification == UNCONSTRAINED:
                        decision_record["outcome"] = "allowed_no_motion_nonprogress_transition"
                        decision_record["state_changed"] = False
                        decision_record["automaton_state_changed"] = (
                            decision_record["automaton_state_id"]
                            != decision_record["target_state_id"]
                        )
                        automaton.observe_accepted_action(
                            task_state_before_action=task_state,
                            action_label=label,
                            physically_executed=False,
                        )
                        nonprogress_record = {
                            "decision_step": decision_number,
                            "action": dict(action),
                            "classification": classification,
                            "source_state_id": decision_record["automaton_state_id"],
                            "target_state_id": decision_record["target_state_id"],
                            "state_preserving": (
                                decision_record["automaton_state_id"]
                                == decision_record["target_state_id"]
                            ),
                            "reason": disposition.skip_reason,
                        }
                        automaton_nonprogress_history.append(nonprogress_record)
                        if label == "add:salt":
                            history.append(action)
                        print("Allowed non-progress model decision; no MuJoCo motion executed.")
                    else:
                        decision_record["outcome"] = "invalid_recommendation_skipped"
                        print("Invalid recommendation recorded; no MuJoCo motion executed.")
                    print("Reason:", disposition.skip_reason)
                    continue

                if disposition.violation:
                    hard_violation_executed = True
                    print("Unsafe/out-of-window action will execute in unshielded mode.")

            elif classification == FINISH:
                history.append(action)
                decision_record["outcome"] = "valid_finish"
                if automaton.recipe_complete(task_state):
                    finished = True
                    termination_reason = "valid_finish"
                    print("Model returned finish after all required ingredients were complete.")
                else:
                    hard_violation_observed = True
                    model_failure_reason = "premature finish while recipe was incomplete"
                    termination_reason = "premature_finish_in_shielded_mode"
                    print("\n*** PREMATURE FINISH DETECTED ***")
                break

            elif transition.physical_effect == NO_MOTION:
                if guide is not None:
                    guide.observe_accepted_action(task_state, label)
                automaton.observe_accepted_action(
                    task_state_before_action=task_state,
                    action_label=label,
                    physically_executed=False,
                )
                decision_record["outcome"] = "allowed_no_motion_nonprogress_transition"
                decision_record["state_changed"] = False
                decision_record["automaton_state_changed"] = (
                    decision_record["automaton_state_id"]
                    != decision_record["target_state_id"]
                )
                nonprogress_record = {
                    "decision_step": decision_number,
                    "action": dict(action),
                    "classification": classification,
                    "source_state_id": decision_record["automaton_state_id"],
                    "target_state_id": decision_record["target_state_id"],
                    "state_preserving": (
                        decision_record["automaton_state_id"]
                        == decision_record["target_state_id"]
                    ),
                    "reason": transition.reason,
                }
                automaton_nonprogress_history.append(nonprogress_record)
                if label == "add:salt":
                    history.append(action)
                print("Allowed unconstrained automaton transition; no MuJoCo motion executed.")
                continue

            state_before_execution = task_state

            if action["action"] == "add":
                item = action["item"]
                print(f"Executing scripted skill: add_ingredient({item})")
                sim.add_ingredient(item)
            elif action["action"] == "stir":
                print("Executing scripted skill: stir()")
                sim.stir()
            else:
                raise RuntimeError(f"Unhandled action: {action}")

            executed_simulation_action_count += 1
            decision_record["executed_in_simulation"] = True
            decision_record["outcome"] = "executed_in_simulation"
            decision_record["state_changed"] = classification == "live_progress"

            if guide is not None:
                guide.observe_accepted_action(state_before_execution, label)
            automaton.observe_accepted_action(
                task_state_before_action=state_before_execution,
                action_label=label,
                physically_executed=True,
            )

            history.append(action)

            executed_count = executed_simulation_action_count
            disturbance_applied = maybe_apply_disturbance(
                args=args,
                model=model,
                data=data,
                sim=sim,
                executed_action_count=executed_count,
                already_applied=disturbance_applied,
            )

            print("\nTask state after execution:")
            print_task_state(model, data)

        if termination_reason is None:
            termination_reason = "decision_budget_exhausted"

        final_state = get_task_state(model, data)
        complete = automaton.recipe_complete(final_state)
        success = bool(
            complete
            and finished
            and not hard_violation_observed
            and model_failure_reason is None
        )

        if termination_reason == "decision_budget_exhausted":
            success = False

        if success:
            outcome_reason: Optional[str] = None
        elif model_failure_reason is not None:
            outcome_reason = model_failure_reason
        elif hard_violation_observed:
            outcome_reason = "one or more unshielded hard recipe violations were observed"
        elif termination_reason == "decision_budget_exhausted":
            outcome_reason = (
                f"{decision_budget['limit']}-decision budget exhausted without a valid finish"
            )
        elif not finished:
            outcome_reason = "episode ended without a valid finish"
        else:
            outcome_reason = "episode did not satisfy the success criteria"

        print("\n" + "=" * 72)
        print("FINAL EXPERIMENT RESULT")
        print("=" * 72)
        print_task_state(model, data)
        print("Recipe ingredients complete:", complete)
        print("Valid finish received:", finished)
        print("Hard recipe violation observed:", hard_violation_observed)
        print("Hard recipe violation executed in simulation:", hard_violation_executed)
        print("Final success:", success)
        print("Termination reason:", termination_reason)
        print("Failure reason:", outcome_reason)
        print("Model decision count:", decision_budget["count"])
        print("Simulation action count:", executed_simulation_action_count)
        print("Violation count:", len(violation_history))
        print("Accepted non-progress transition count:", len(automaton_nonprogress_history))
        print("Environment completed garlic disturbance applied:", disturbance_applied)
        print("Executed/accepted high-level command history:")
        print(json.dumps(history, indent=2))
        print("Complete model-decision history:")
        print(json.dumps(model_decision_history, indent=2))
        print("Unshielded violation history:")
        print(json.dumps(violation_history, indent=2))
        print("Model recommendations skipped in simulation:")
        print(json.dumps(skipped_action_history, indent=2))
        print("Accepted non-progress transition history:")
        print(json.dumps(automaton_nonprogress_history, indent=2))
        print("=" * 72)

        if viewer is not None:
            print("\nClose viewer when done.")
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.01)

        return success

    finally:
        renderer.close()
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cooking Experiment 7: human/environment completes garlic with manual STARs-style strategy-template guidance."
    )
    parser.add_argument(
        "--planner",
        choices=["manual", "qwen_ollama"],
        default="qwen_ollama",
    )
    parser.add_argument(
        "--shield-mode",
        choices=["off", "template_guidance"],
        default="template_guidance",
        help="off = raw Qwen execution; template_guidance = reject unsafe proposals and threshold-guide Qwen.",
    )

    parser.add_argument("--ollama-url", default="http://volta13:11434")
    parser.add_argument("--ollama-model", default="qwen2.5vl:3b")
    parser.add_argument("--ollama-timeout", type=int, default=1800)
    parser.add_argument(
        "--ollama-http-retries",
        type=int,
        default=0,
        help="Default 0: no model-request fallback/retry beyond the requested call.",
    )
    parser.add_argument("--ollama-retry-backoff", type=float, default=20.0)
    parser.add_argument("--ollama-num-predict", type=int, default=1024)
    parser.add_argument("--ollama-keep-alive", default="2h")
    parser.add_argument(
        "--ollama-think",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--shield-gamma",
        type=float,
        default=0.20,
        help=(
            "State-local liveness mass-transfer rate. After a non-progress decision, "
            "gamma of the remaining non-live mass moves toward the live set."
        ),
    )
    parser.add_argument(
        "--shield-theta-prune",
        type=float,
        default=0.10,
        help=(
            "Low-probability pruning threshold applied to co-live actions and "
            "repeated state-preserving unconstrained actions. State-changing "
            "unconstrained transitions remain exempt."
        ),
    )
    parser.add_argument(
        "--shield-guidance-threshold",
        "--shield-theta-guide",
        dest="shield_guidance_threshold",
        type=float,
        default=0.95,
        help=(
            "Activate the firm non-answer-revealing progress directive when total "
            "live-set shield probability reaches this value."
        ),
    )
    parser.add_argument(
        "--shield-reprompt-limit",
        type=int,
        default=12,
        help=(
            "Maximum consecutive shield reprompts after rejected proposals. The initial "
            "request plus these reprompts are all counted in the global decision budget."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help=(
            "Shielded global model-decision budget. Every parsed model response counts, "
            "including rejected proposals and accepted no-motion non-progress transitions."
        ),
    )
    parser.add_argument(
        "--unshielded-max-steps",
        type=int,
        default=50,
        help=(
            "Unshielded model-decision observation budget. Duplicate adds and "
            "premature finish recommendations count toward this budget even though "
            "they are logged rather than executed in simulation."
        ),
    )
    parser.add_argument(
        "--salt-max-additions",
        type=int,
        default=3,
        help=(
            "Maximum number of small salt increments. The first satisfies the recipe; "
            "the second and third are unconstrained seasoning transitions; later salt is unsafe."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-jitter-xy", type=float, default=0.0)
    parser.add_argument("--viewer", action="store_true")

    parser.add_argument(
        "--run-id",
        default="qwen25_cooking_exp07_human_completed_garlic",
    )
    parser.add_argument(
        "--trace-jsonl",
        default="logs_model_reasoning/qwen25_cooking_exp07_human_completed_garlic.jsonl",
    )
    parser.add_argument(
        "--shield-trace-jsonl",
        default="logs_model_reasoning/qwen25_cooking_exp07_human_completed_garlic_shield.jsonl",
    )
    parser.add_argument(
        "--image-dir",
        default="planner_images_qwen25_cooking_exp07_human_completed_garlic",
    )

    args = parser.parse_args()

    try:
        success = run_episode(args)
    except Exception as exc:
        print("\nEXPERIMENT FAILED WITH UNHANDLED ERROR")
        print(type(exc).__name__ + ":", exc)
        raise

    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
