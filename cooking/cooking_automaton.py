"""Explicit finite-state cooking automaton used by the runtime shield.

The automaton is observation-synchronised: each MuJoCo observation is mapped to
an explicit node, and the model's one returned high-level action is looked up in
that node's transition table. Environment interventions can therefore move the
automaton independently of the previously executed command history.

The action partition follows the mentor discussion:
  * live_progress: advances the recipe state;
  * co_live: harmless state-preserving action (`stir`) that may not dominate forever;
  * unconstrained: currently sensible non-progress action that remains allowed;
  * unsafe: violates the current cooking window or quantity constraint;
  * finish: accepting transition once all required obligations are complete.

The automaton never asks the VLM for token/action probabilities. Qwen still
returns exactly one JSON action per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple


LIVE_PROGRESS = "live_progress"
CO_LIVE = "co_live"
UNCONSTRAINED = "unconstrained"
UNSAFE = "unsafe"
FINISH = "finish"

EXECUTE_ADD = "execute_add"
EXECUTE_STIR = "execute_stir"
NO_MOTION = "no_motion"
REJECT = "reject"
TERMINATE = "terminate"

BASE_INGREDIENT_ORDER: Tuple[str, ...] = (
    "onion",
    "garlic",
    "tomato",
    "turmeric",
    "coriander",
    "salt",
    "cream",
)


@dataclass(frozen=True)
class AutomatonTransition:
    action_label: str
    classification: str
    target_state_id: str
    physical_effect: str
    reason: str


@dataclass(frozen=True)
class AutomatonNode:
    state_id: str
    state_key: Tuple[Any, ...]
    stage_name: str
    live_actions: Tuple[str, ...]
    colive_actions: Tuple[str, ...]
    unconstrained_actions: Tuple[str, ...]
    unsafe_actions: Tuple[str, ...]
    transitions: Mapping[str, AutomatonTransition]
    salt_add_count: int
    cumin_required: bool

    def transition(self, action_label: str) -> AutomatonTransition:
        try:
            return self.transitions[action_label]
        except KeyError as exc:
            raise ValueError(f"Action {action_label!r} is not in this automaton's action space.") from exc


class CookingAutomaton:
    """Observation-synchronised finite-state automaton for the cooking benchmark."""

    def __init__(
        self,
        *,
        action_labels: Sequence[str],
        always_unsafe_actions: Iterable[str] = (),
        max_salt_additions: int = 3,
    ) -> None:
        if max_salt_additions < 1:
            raise ValueError("max_salt_additions must be at least 1")

        labels = tuple(dict.fromkeys(str(label) for label in action_labels))
        if "stir" not in labels or "finish" not in labels:
            raise ValueError("action_labels must include both 'stir' and 'finish'")

        self.action_labels = labels
        self.always_unsafe_actions = frozenset(str(a) for a in always_unsafe_actions)
        self.max_salt_additions = int(max_salt_additions)
        self.salt_add_count = 0
        self.aromatics_window_closed = False
        self.tomato_window_closed = False
        self.core_spice_window_closed = False
        self._node_cache: Dict[Tuple[Any, ...], AutomatonNode] = {}

    def reset(self) -> None:
        self.salt_add_count = 0
        self.aromatics_window_closed = False
        self.tomato_window_closed = False
        self.core_spice_window_closed = False
        self._node_cache.clear()

    @staticmethod
    def _added(task_state: Mapping[str, Mapping[str, Any]], item: str) -> bool:
        return bool(task_state.get(item, {}).get("added", False))

    @staticmethod
    def _cumin_required(task_state: Mapping[str, Mapping[str, Any]]) -> bool:
        return bool(task_state.get("cumin", {}).get("required", False))

    def _sync_observed_salt(self, task_state: Mapping[str, Mapping[str, Any]]) -> None:
        # If salt arrived through an environment intervention or an unshielded
        # out-of-order physical action, the observation establishes that at least one
        # salt increment has occurred.
        if self._added(task_state, "salt") and self.salt_add_count == 0:
            self.salt_add_count = 1

    def _sync_cooking_windows(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Close timing windows monotonically from the observed physical state.

        This preserves common-sense cooking semantics under disturbances: an early
        ingredient may be recovered while its cooking window is still open, but a
        missing onion after tomato has already been added does not rewind the recipe
        and make a late raw-onion addition valid again.
        """
        before = (
            self.aromatics_window_closed,
            self.tomato_window_closed,
            self.core_spice_window_closed,
        )

        if self._added(task_state, "tomato"):
            self.aromatics_window_closed = True

        if any(
            self._added(task_state, item)
            for item in ("turmeric", "coriander", "cumin")
        ):
            self.tomato_window_closed = True
            self.aromatics_window_closed = True

        if self._added(task_state, "cream"):
            self.core_spice_window_closed = True
            self.tomato_window_closed = True
            self.aromatics_window_closed = True

        after = (
            self.aromatics_window_closed,
            self.tomato_window_closed,
            self.core_spice_window_closed,
        )
        if after != before:
            self._node_cache.clear()

    def required_core_spices(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
    ) -> Tuple[str, ...]:
        spices = ["turmeric", "coriander"]
        if self._cumin_required(task_state):
            spices.append("cumin")
        return tuple(spices)

    def stage_name(self, task_state: Mapping[str, Mapping[str, Any]]) -> str:
        aromatics_complete = (
            self._added(task_state, "onion") and self._added(task_state, "garlic")
        )
        if not aromatics_complete:
            if self.aromatics_window_closed:
                return "irrecoverable_order_violation"
            return "aromatics"

        if not self._added(task_state, "tomato"):
            if self.tomato_window_closed:
                return "irrecoverable_order_violation"
            return "tomato"

        if any(
            not self._added(task_state, item)
            for item in self.required_core_spices(task_state)
        ):
            if self.core_spice_window_closed:
                return "irrecoverable_order_violation"
            return "spices"

        if not self._added(task_state, "cream") or self.salt_add_count < 1:
            return "finishing"
        return "ready_to_finish"

    def recipe_complete(self, task_state: Mapping[str, Mapping[str, Any]]) -> bool:
        return self.stage_name(task_state) == "ready_to_finish"

    def _state_key(self, task_state: Mapping[str, Mapping[str, Any]]) -> Tuple[Any, ...]:
        action_items = tuple(
            label.split(":", 1)[1]
            for label in self.action_labels
            if label.startswith("add:")
        )
        observed = tuple((item, self._added(task_state, item)) for item in action_items)
        return (
            self.stage_name(task_state),
            observed,
            self._cumin_required(task_state),
            int(self.salt_add_count),
            bool(self.aromatics_window_closed),
            bool(self.tomato_window_closed),
            bool(self.core_spice_window_closed),
        )

    @staticmethod
    def _state_id_from_key(state_key: Tuple[Any, ...]) -> str:
        (
            stage,
            observed,
            cumin_required,
            salt_count,
            aromatics_closed,
            tomato_closed,
            core_spice_closed,
        ) = state_key
        bits = "".join("1" if added else "0" for _item, added in observed)
        return (
            f"{stage}|bits={bits}|cumin_required={int(bool(cumin_required))}"
            f"|salt={salt_count}|windows={int(aromatics_closed)}"
            f"{int(tomato_closed)}{int(core_spice_closed)}"
        )

    def _target_state_id_after_add(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
        item: str,
    ) -> str:
        shadow: Dict[str, Dict[str, Any]] = {
            name: dict(values) for name, values in task_state.items()
        }
        shadow.setdefault(item, {})["added"] = True
        previous = (
            self.salt_add_count,
            self.aromatics_window_closed,
            self.tomato_window_closed,
            self.core_spice_window_closed,
        )
        try:
            if item == "salt":
                self.salt_add_count = min(
                    self.max_salt_additions,
                    max(1, self.salt_add_count + 1),
                )
            elif item == "tomato":
                self.aromatics_window_closed = True
            elif item in {"turmeric", "coriander", "cumin"}:
                self.aromatics_window_closed = True
                self.tomato_window_closed = True
            elif item == "cream":
                self.aromatics_window_closed = True
                self.tomato_window_closed = True
                self.core_spice_window_closed = True
            key = self._state_key(shadow)
            return self._state_id_from_key(key)
        finally:
            (
                self.salt_add_count,
                self.aromatics_window_closed,
                self.tomato_window_closed,
                self.core_spice_window_closed,
            ) = previous

    def _transition_for_add(
        self,
        *,
        task_state: Mapping[str, Mapping[str, Any]],
        stage: str,
        item: str,
        current_state_id: str,
    ) -> AutomatonTransition:
        label = f"add:{item}"
        added = self._added(task_state, item)

        if stage == "irrecoverable_order_violation":
            return AutomatonTransition(
                label,
                UNSAFE,
                current_state_id,
                REJECT,
                "the observed recipe state cannot be repaired within the remaining cooking windows",
            )

        if label in self.always_unsafe_actions:
            return AutomatonTransition(
                label,
                UNSAFE,
                current_state_id,
                REJECT,
                "action is permanently outside the recipe specification",
            )

        if item == "cumin" and not self._cumin_required(task_state):
            return AutomatonTransition(
                label,
                UNSAFE,
                current_state_id,
                REJECT,
                "cumin is not currently a required visible recipe ingredient",
            )

        if item == "salt":
            if stage in {"aromatics", "tomato"}:
                return AutomatonTransition(
                    label,
                    UNSAFE,
                    current_state_id,
                    REJECT,
                    "salt is outside its sensible cooking window",
                )
            if self.salt_add_count >= self.max_salt_additions:
                return AutomatonTransition(
                    label,
                    UNSAFE,
                    current_state_id,
                    REJECT,
                    f"salt limit of {self.max_salt_additions} increments has been reached",
                )
            if self.salt_add_count == 0:
                return AutomatonTransition(
                    label,
                    LIVE_PROGRESS,
                    self._target_state_id_after_add(task_state, item),
                    EXECUTE_ADD,
                    "first salt increment satisfies a required recipe obligation",
                )
            return AutomatonTransition(
                label,
                UNCONSTRAINED,
                self._target_state_id_after_add(task_state, item),
                NO_MOTION,
                "an additional salt increment is still sensible but does not advance the required recipe stage",
            )

        if item in {"onion", "garlic"}:
            if stage == "aromatics":
                if added:
                    return AutomatonTransition(
                        label,
                        UNCONSTRAINED,
                        current_state_id,
                        NO_MOTION,
                        "already-satisfied aromatic command is a permitted state-preserving self-loop",
                    )
                return AutomatonTransition(
                    label,
                    LIVE_PROGRESS,
                    self._target_state_id_after_add(task_state, item),
                    EXECUTE_ADD,
                    "missing aromatic advances Stage 1",
                )
            if stage == "tomato" and added:
                return AutomatonTransition(
                    label,
                    UNCONSTRAINED,
                    current_state_id,
                    NO_MOTION,
                    "completed aromatic remains a permitted self-loop before tomato is added",
                )
            return AutomatonTransition(
                label,
                UNSAFE,
                current_state_id,
                REJECT,
                "adding onion or garlic after the tomato window begins would not cook it properly",
            )

        if item == "tomato":
            if stage == "tomato" and not added:
                return AutomatonTransition(
                    label,
                    LIVE_PROGRESS,
                    self._target_state_id_after_add(task_state, item),
                    EXECUTE_ADD,
                    "tomato advances Stage 2 after both aromatics",
                )
            return AutomatonTransition(
                label,
                UNSAFE,
                current_state_id,
                REJECT,
                "tomato is either premature, already complete, or too late for its cooking window",
            )

        core_spices = set(self.required_core_spices(task_state))
        if item in {"turmeric", "coriander", "cumin"}:
            if item not in core_spices:
                return AutomatonTransition(
                    label,
                    UNSAFE,
                    current_state_id,
                    REJECT,
                    "ingredient is not in the current required core-spice set",
                )
            if stage == "spices":
                if added:
                    return AutomatonTransition(
                        label,
                        UNCONSTRAINED,
                        current_state_id,
                        NO_MOTION,
                        "already-satisfied core-spice command is a permitted self-loop during the spice stage",
                    )
                return AutomatonTransition(
                    label,
                    LIVE_PROGRESS,
                    self._target_state_id_after_add(task_state, item),
                    EXECUTE_ADD,
                    "missing core spice advances Stage 3",
                )
            return AutomatonTransition(
                label,
                UNSAFE,
                current_state_id,
                REJECT,
                "core spice is outside its sensible cooking window",
            )

        if item == "cream":
            if stage == "finishing" and not added:
                return AutomatonTransition(
                    label,
                    LIVE_PROGRESS,
                    self._target_state_id_after_add(task_state, item),
                    EXECUTE_ADD,
                    "cream advances the finishing stage after core spices are complete",
                )
            return AutomatonTransition(
                label,
                UNSAFE,
                current_state_id,
                REJECT,
                "cream is premature or has already been added",
            )

        return AutomatonTransition(
            label,
            UNSAFE,
            current_state_id,
            REJECT,
            "unknown or unsupported recipe ingredient",
        )

    def _build_node(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
    ) -> AutomatonNode:
        stage = self.stage_name(task_state)
        state_key = self._state_key(task_state)
        state_id = self._state_id_from_key(state_key)
        transitions: MutableMapping[str, AutomatonTransition] = {}

        for label in self.action_labels:
            if label in self.always_unsafe_actions:
                transitions[label] = AutomatonTransition(
                    label,
                    UNSAFE,
                    state_id,
                    REJECT,
                    "action is permanently outside the recipe specification",
                )
            elif label.startswith("add:"):
                transitions[label] = self._transition_for_add(
                    task_state=task_state,
                    stage=stage,
                    item=label.split(":", 1)[1],
                    current_state_id=state_id,
                )
            elif label == "stir":
                transitions[label] = AutomatonTransition(
                    label,
                    CO_LIVE,
                    state_id,
                    EXECUTE_STIR,
                    "stir is temporarily permitted but does not advance the recipe",
                )
            elif label == "finish":
                if stage == "ready_to_finish":
                    transitions[label] = AutomatonTransition(
                        label,
                        FINISH,
                        "terminal_success",
                        TERMINATE,
                        "all required cooking obligations are complete",
                    )
                else:
                    transitions[label] = AutomatonTransition(
                        label,
                        UNSAFE,
                        state_id,
                        REJECT,
                        "finish is premature while recipe obligations remain",
                    )
            else:
                transitions[label] = AutomatonTransition(
                    label,
                    UNSAFE,
                    state_id,
                    REJECT,
                    "unknown action label",
                )

        live = tuple(
            label
            for label, transition in transitions.items()
            if transition.classification in {LIVE_PROGRESS, FINISH}
        )
        colive = tuple(
            label for label, transition in transitions.items() if transition.classification == CO_LIVE
        )
        unconstrained = tuple(
            label
            for label, transition in transitions.items()
            if transition.classification == UNCONSTRAINED
        )
        unsafe = tuple(
            label for label, transition in transitions.items() if transition.classification == UNSAFE
        )

        return AutomatonNode(
            state_id=state_id,
            state_key=state_key,
            stage_name=stage,
            live_actions=live,
            colive_actions=colive,
            unconstrained_actions=unconstrained,
            unsafe_actions=unsafe,
            transitions=MappingProxyType(dict(transitions)),
            salt_add_count=int(self.salt_add_count),
            cumin_required=self._cumin_required(task_state),
        )

    def node(self, task_state: Mapping[str, Mapping[str, Any]]) -> AutomatonNode:
        self._sync_observed_salt(task_state)
        self._sync_cooking_windows(task_state)
        key = self._state_key(task_state)
        node = self._node_cache.get(key)
        if node is None:
            node = self._build_node(task_state)
            self._node_cache[key] = node
        return node

    def transition(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
        action_label: str,
    ) -> AutomatonTransition:
        return self.node(task_state).transition(action_label)

    def classify(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
        action_label: str,
    ) -> str:
        return self.transition(task_state, action_label).classification

    def observe_accepted_action(
        self,
        *,
        task_state_before_action: Mapping[str, Mapping[str, Any]],
        action_label: str,
        physically_executed: bool,
    ) -> None:
        transition = self.transition(task_state_before_action, action_label)

        if action_label == "add:tomato" and physically_executed:
            self.aromatics_window_closed = True
            self._node_cache.clear()
        elif action_label in {"add:turmeric", "add:coriander", "add:cumin"} and physically_executed:
            self.tomato_window_closed = True
            self.aromatics_window_closed = True
            self._node_cache.clear()
        elif action_label == "add:cream" and physically_executed:
            self.core_spice_window_closed = True
            self.tomato_window_closed = True
            self.aromatics_window_closed = True
            self._node_cache.clear()

        if action_label == "add:salt" and transition.classification in {
            LIVE_PROGRESS,
            UNCONSTRAINED,
        }:
            self.salt_add_count = min(
                self.max_salt_additions,
                self.salt_add_count + 1,
            )
            self._node_cache.clear()
        elif action_label == "add:salt" and physically_executed:
            # Unshielded execution of an unsafe early salt proposal still changes
            # the physical recipe quantity and must be represented.
            self.salt_add_count = min(
                self.max_salt_additions,
                max(1, self.salt_add_count + 1),
            )
            self._node_cache.clear()

    def snapshot_jsonable(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        node = self.node(task_state)
        return {
            "state_id": node.state_id,
            "state_key": list(node.state_key),
            "stage_name": node.stage_name,
            "salt_add_count": node.salt_add_count,
            "max_salt_additions": self.max_salt_additions,
            "cumin_required": node.cumin_required,
            "cooking_windows": {
                "aromatics_closed": self.aromatics_window_closed,
                "tomato_closed": self.tomato_window_closed,
                "core_spice_closed": self.core_spice_window_closed,
            },
            "live_actions": list(node.live_actions),
            "colive_actions": list(node.colive_actions),
            "unconstrained_actions": list(node.unconstrained_actions),
            "unsafe_actions": list(node.unsafe_actions),
            "transitions": {
                label: {
                    "classification": transition.classification,
                    "target_state_id": transition.target_state_id,
                    "physical_effect": transition.physical_effect,
                    "reason": transition.reason,
                }
                for label, transition in node.transitions.items()
            },
        }
