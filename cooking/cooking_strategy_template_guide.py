"""Runtime strategy-template guide backed by an explicit cooking automaton.

This keeps the one-action-per-request VLM interface from the previous experiment.
The guide maintains its own synthetic distribution; it does not request or depend
on token-derived action probabilities from the model.

For each automaton node:
  * unsafe transitions receive probability 0 and are rejected;
  * live transitions gain shield-side mass when progress is neglected;
  * `stir` is co-live and can be pruned after repeated use;
  * repeated state-preserving unconstrained transitions receive an action-specific
    penalty and can be pruned once their synthetic probability crosses theta_prune;
  * unconstrained transitions that change the automaton state remain unaffected;
  * when total live-set mass reaches theta_guide, the prompt receives a firm,
    non-answer-revealing progress directive;
  * after a rejected proposal, the reprompt reveals one exact live/progress action
    from the current automaton state while leaving action execution to the VLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from cooking_automaton import (
    CO_LIVE,
    FINISH,
    LIVE_PROGRESS,
    UNCONSTRAINED,
    UNSAFE,
    AutomatonNode,
    CookingAutomaton,
)
ACTION_LABELS: Tuple[str, ...] = (
    "add:onion", "add:garlic", "add:tomato", "add:turmeric",
    "add:coriander", "add:salt", "add:cream", "stir", "finish",
)


@dataclass(frozen=True)
class TemplateSnapshot:
    state_key: Tuple[Any, ...]
    automaton_state_id: str
    stage_name: str
    unsafe_actions: Tuple[str, ...]
    colive_actions: Tuple[str, ...]
    unconstrained_actions: Tuple[str, ...]
    live_actions: Tuple[str, ...]
    live_counter: int
    colive_counters: Dict[str, int]
    unconstrained_self_loop_counters: Dict[str, int]
    base_distribution: Dict[str, float]
    pre_prune_distribution: Dict[str, float]
    modified_distribution: Dict[str, float]
    live_group_probability_mass: float
    theta_prune: float
    guidance_threshold: float
    threshold_crossed: bool
    directive_active: bool
    guidance_candidates: Tuple[str, ...]
    pruned_actions: Tuple[str, ...]
    salt_add_count: int


@dataclass(frozen=True)
class TemplateCheck:
    action_label: str
    classification: str
    allowed: bool
    reason: str
    physical_effect: str
    target_state_id: str
    snapshot_before_update: TemplateSnapshot


def _normalize_nonnegative(distribution: Mapping[str, float]) -> Dict[str, float]:
    clipped = {action: max(0.0, float(value)) for action, value in distribution.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        raise RuntimeError(
            "Strategy-template distribution has zero total mass after filtering/pruning."
        )
    return {action: value / total for action, value in clipped.items()}


class ManualCookingStrategyTemplateGuide:
    """Automaton-backed mentor-directed strategy-template adaptation."""

    def __init__(
        self,
        gamma: float = 0.20,
        theta_prune: float = 0.10,
        guidance_threshold: float = 0.95,
        action_labels: Sequence[str] = ACTION_LABELS,
        automaton: Optional[CookingAutomaton] = None,
        max_salt_additions: int = 3,
    ) -> None:
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not 0.0 <= theta_prune < 1.0:
            raise ValueError("theta_prune must be in [0, 1)")
        if not 0.0 < guidance_threshold < 1.0:
            raise ValueError("guidance_threshold must be between 0 and 1")

        self.gamma = float(gamma)
        self.theta_prune = float(theta_prune)
        self.guidance_threshold = float(guidance_threshold)
        self.action_labels = tuple(action_labels)
        self.automaton = automaton or CookingAutomaton(
            action_labels=self.action_labels,
            max_salt_additions=max_salt_additions,
        )

        self.live_counters: Dict[Tuple[Any, ...], int] = {}
        self.colive_counters: Dict[Tuple[Any, ...], int] = {}
        self.unconstrained_self_loop_counters: Dict[str, int] = {}
        self._unconstrained_counter_state_key: Optional[Tuple[Any, ...]] = None

    def reset(self) -> None:
        self.live_counters.clear()
        self.colive_counters.clear()
        self.unconstrained_self_loop_counters.clear()
        self._unconstrained_counter_state_key = None
        self.automaton.reset()

    def _node(self, task_state: Mapping[str, Mapping[str, Any]]) -> AutomatonNode:
        node = self.automaton.node(task_state)
        self._sync_unconstrained_counter_scope(node)
        return node

    def _sync_unconstrained_counter_scope(self, node: AutomatonNode) -> None:
        """Reset repeated-self-loop penalties whenever the cooking state changes."""
        state_key = tuple(node.state_key)
        if self._unconstrained_counter_state_key is None:
            self._unconstrained_counter_state_key = state_key
            return
        if state_key != self._unconstrained_counter_state_key:
            self.unconstrained_self_loop_counters.clear()
            self._unconstrained_counter_state_key = state_key

    @staticmethod
    def _is_state_preserving_unconstrained(
        node: AutomatonNode,
        action_label: str,
    ) -> bool:
        transition = node.transition(action_label)
        return (
            transition.classification == UNCONSTRAINED
            and transition.target_state_id == node.state_id
        )

    @staticmethod
    def _counter_state_scope(node: AutomatonNode) -> Tuple[Any, ...]:
        # state_key layout: stage, observed bits, cumin-required, salt-count, windows.
        # Exclude only salt-count so second/third salt increments do not reset the
        # still-active live obligation. All physical/phase changes remain state-scoped.
        return tuple(node.state_key[:3]) + tuple(node.state_key[4:])

    @classmethod
    def _live_counter_key_from_node(cls, node: AutomatonNode) -> Tuple[Any, ...]:
        return (cls._counter_state_scope(node), tuple(node.live_actions))

    @classmethod
    def _colive_counter_key_from_node(
        cls,
        node: AutomatonNode,
        action_label: str,
    ) -> Tuple[Any, ...]:
        return (cls._counter_state_scope(node), action_label)

    def _live_counter_key(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
    ) -> Tuple[Any, ...]:
        return self._live_counter_key_from_node(self._node(task_state))

    def _colive_counter_key(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
        action_label: str,
    ) -> Tuple[Any, ...]:
        return self._colive_counter_key_from_node(self._node(task_state), action_label)

    def _current_colive_counters(
        self,
        node: AutomatonNode,
    ) -> Dict[str, int]:
        return {
            action: self.colive_counters.get(
                self._colive_counter_key_from_node(node, action),
                0,
            )
            for action in node.colive_actions
        }

    def _current_unconstrained_self_loop_counters(
        self,
        node: AutomatonNode,
    ) -> Dict[str, int]:
        return {
            action: self.unconstrained_self_loop_counters.get(action, 0)
            for action in node.unconstrained_actions
            if self._is_state_preserving_unconstrained(node, action)
        }

    def _template_sets(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
    ) -> Tuple[
        str,
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
    ]:
        node = self._node(task_state)
        return (
            node.stage_name,
            node.unsafe_actions,
            node.colive_actions,
            node.unconstrained_actions,
            node.live_actions,
        )

    def base_distribution(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, float]:
        """Uniform prior over all transitions that are not unsafe."""
        node = self._node(task_state)
        unsafe = set(node.unsafe_actions)
        raw = {
            action: (0.0 if action in unsafe else 1.0)
            for action in self.action_labels
        }
        return _normalize_nonnegative(raw)

    def pre_prune_distribution(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, float]:
        """Transfer mass toward the live set after state-preserving decisions.

        If n is the state-local live-neglect counter, the non-live mass retained is
        `(1 - gamma) ** n` times its initial amount. Relative probabilities inside
        the live set and inside the non-live set are preserved. Repeated co-live
        actions receive an additional action-specific multiplicative penalty.
        """
        node = self._node(task_state)
        distribution = self.base_distribution(task_state)
        live = tuple(node.live_actions)
        non_live = tuple(
            action
            for action in self.action_labels
            if action not in set(node.unsafe_actions)
            and action not in set(live)
        )

        live_key = self._live_counter_key_from_node(node)
        neglect_count = self.live_counters.get(live_key, 0)

        base_live_mass = sum(distribution.get(action, 0.0) for action in live)
        base_non_live_mass = sum(distribution.get(action, 0.0) for action in non_live)

        if live and base_live_mass > 0.0 and base_non_live_mass > 0.0:
            retained_non_live_mass = base_non_live_mass * ((1.0 - self.gamma) ** neglect_count)
            desired_live_mass = 1.0 - retained_non_live_mass
            live_scale = desired_live_mass / base_live_mass
            non_live_scale = retained_non_live_mass / base_non_live_mass

            for action in live:
                distribution[action] *= live_scale
            for action in non_live:
                distribution[action] *= non_live_scale

        # Co-live actions retain their existing action-specific penalty.
        for action in node.colive_actions:
            count = self.colive_counters.get(
                self._colive_counter_key_from_node(node, action),
                0,
            )
            distribution[action] *= (1.0 - self.gamma) ** count

        # Repeated state-preserving unconstrained actions receive the same style
        # of action-specific decay. State-changing unconstrained transitions (for
        # example additional permitted salt increments) are deliberately excluded.
        for action in node.unconstrained_actions:
            if not self._is_state_preserving_unconstrained(node, action):
                continue
            count = self.unconstrained_self_loop_counters.get(action, 0)
            distribution[action] *= (1.0 - self.gamma) ** count

        for action in node.unsafe_actions:
            distribution[action] = 0.0

        return _normalize_nonnegative(distribution)

    def modified_distribution(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, float]:
        """Apply theta_prune to co-live and repeated self-loop actions.

        A state-preserving unconstrained action becomes prune-eligible only after
        it has actually been accepted in the current cooking state. Unchosen
        unconstrained actions and state-changing unconstrained transitions remain
        available even when liveness mass transfer makes their probability small.
        """
        node = self._node(task_state)
        pre_prune = self.pre_prune_distribution(task_state)
        pruned = dict(pre_prune)

        prune_eligible = set(node.colive_actions)
        prune_eligible.update(
            action
            for action in node.unconstrained_actions
            if self._is_state_preserving_unconstrained(node, action)
            and self.unconstrained_self_loop_counters.get(action, 0) > 0
        )

        for action in prune_eligible:
            if pre_prune.get(action, 0.0) <= self.theta_prune:
                pruned[action] = 0.0
        return _normalize_nonnegative(pruned)

    def snapshot(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
    ) -> TemplateSnapshot:
        node = self._node(task_state)
        base = self.base_distribution(task_state)
        pre_prune = self.pre_prune_distribution(task_state)
        modified = self.modified_distribution(task_state)
        live_mass = sum(modified.get(action, 0.0) for action in node.live_actions)
        live_key = self._live_counter_key_from_node(node)
        live_counter = self.live_counters.get(live_key, 0)

        prune_eligible = tuple(node.colive_actions) + tuple(
            action
            for action in node.unconstrained_actions
            if self._is_state_preserving_unconstrained(node, action)
            and self.unconstrained_self_loop_counters.get(action, 0) > 0
        )
        pruned_actions = tuple(
            action
            for action in prune_eligible
            if pre_prune.get(action, 0.0) > 0.0
            and modified.get(action, 0.0) <= 1e-12
        )

        directive_active = bool(
            live_counter > 0
            and node.live_actions
            and live_mass >= self.guidance_threshold
        )

        return TemplateSnapshot(
            state_key=node.state_key,
            automaton_state_id=node.state_id,
            stage_name=node.stage_name,
            unsafe_actions=node.unsafe_actions,
            colive_actions=node.colive_actions,
            unconstrained_actions=node.unconstrained_actions,
            live_actions=node.live_actions,
            live_counter=live_counter,
            colive_counters=self._current_colive_counters(node),
            unconstrained_self_loop_counters=(
                self._current_unconstrained_self_loop_counters(node)
            ),
            base_distribution=base,
            pre_prune_distribution=pre_prune,
            modified_distribution=modified,
            live_group_probability_mass=float(live_mass),
            theta_prune=self.theta_prune,
            guidance_threshold=self.guidance_threshold,
            threshold_crossed=directive_active,
            directive_active=directive_active,
            guidance_candidates=tuple(),
            pruned_actions=pruned_actions,
            salt_add_count=node.salt_add_count,
        )

    def check_action(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
        action_label: str,
    ) -> TemplateCheck:
        node = self._node(task_state)
        transition = node.transition(action_label)
        snapshot = self.snapshot(task_state)
        probability = snapshot.modified_distribution.get(action_label, 0.0)
        classification = transition.classification

        if classification == UNSAFE:
            allowed = False
            reason = transition.reason
        elif classification == CO_LIVE and probability <= 1e-12:
            allowed = False
            reason = "co-live action has been repeated enough to be pruned"
        elif (
            classification == UNCONSTRAINED
            and self._is_state_preserving_unconstrained(node, action_label)
            and probability <= 1e-12
        ):
            allowed = False
            reason = (
                "state-preserving unconstrained action has been repeated enough "
                "to be pruned"
            )
        elif classification == LIVE_PROGRESS:
            allowed = True
            reason = "model action follows a live transition to recipe progress"
        elif classification == FINISH:
            allowed = True
            reason = "model action follows the accepting finish transition"
        elif classification == CO_LIVE:
            allowed = True
            reason = "model action is co-live and temporarily permitted"
        elif classification == UNCONSTRAINED:
            allowed = True
            reason = "model action is an allowed unconstrained transition that does not advance the required recipe stage"
        else:
            allowed = False
            reason = "model action is not represented by a permitted automaton transition"

        return TemplateCheck(
            action_label=action_label,
            classification=classification,
            allowed=allowed,
            reason=reason,
            physical_effect=transition.physical_effect,
            target_state_id=transition.target_state_id,
            snapshot_before_update=snapshot,
        )

    def observe_rejected_attempt(
        self,
        task_state: Mapping[str, Mapping[str, Any]],
        proposed_action_label: str,
    ) -> TemplateSnapshot:
        del proposed_action_label
        node = self._node(task_state)
        live_key = self._live_counter_key_from_node(node)
        self.live_counters[live_key] = self.live_counters.get(live_key, 0) + 1
        return self.snapshot(task_state)

    def observe_accepted_action(
        self,
        task_state_before_action: Mapping[str, Mapping[str, Any]],
        action_label: str,
    ) -> None:
        node = self._node(task_state_before_action)
        transition = node.transition(action_label)
        live_key = self._live_counter_key_from_node(node)

        if transition.classification in {LIVE_PROGRESS, FINISH}:
            self.live_counters[live_key] = 0
            return

        if transition.classification == CO_LIVE:
            colive_key = self._colive_counter_key_from_node(node, action_label)
            self.colive_counters[colive_key] = self.colive_counters.get(colive_key, 0) + 1
            self.live_counters[live_key] = self.live_counters.get(live_key, 0) + 1
            return

        if transition.classification == UNCONSTRAINED:
            if transition.target_state_id == node.state_id:
                self.unconstrained_self_loop_counters[action_label] = (
                    self.unconstrained_self_loop_counters.get(action_label, 0) + 1
                )
            self.live_counters[live_key] = self.live_counters.get(live_key, 0) + 1

    # Backward-compatible name used by the previous experiment scripts. In the
    # automaton branch this method is called for every accepted action, including
    # no-motion unconstrained transitions.
    def observe_executed_action(
        self,
        task_state_before_execution: Mapping[str, Mapping[str, Any]],
        action_label: str,
    ) -> None:
        self.observe_accepted_action(task_state_before_execution, action_label)

    @staticmethod
    def action_label_to_text(action_label: str) -> str:
        if action_label.startswith("add:"):
            return f"Add {action_label.split(':', 1)[1]} to the mixing bowl."
        if action_label == "stir":
            return "Stir the contents of the mixing bowl."
        if action_label == "finish":
            return "Finish the recipe."
        return action_label

    def guidance_text(self, snapshot: TemplateSnapshot) -> str:
        if not snapshot.directive_active:
            return ""
        return (
            "RUNTIME STRATEGY-TEMPLATE DIRECTIVE:\n"
            "The task state has remained unchanged for too long.\n\n"
            "Your next proposal must be a valid progress action that changes the current "
            "recipe state. Infer the appropriate progress action from the current image "
            "and recipe constraints. The runtime shield is not providing the action for you.\n\n"
            "Return exactly one valid JSON action and no additional text outside the JSON object."
        )

    @staticmethod
    def action_label_to_json(action_label: str) -> str:
        """Return the exact planner JSON for one automaton action label."""
        reasoning = "The runtime shield identified the required progress action."
        if action_label.startswith("add:"):
            item = action_label.split(":", 1)[1]
            return (
                f'{{"reasoning":"{reasoning}",'
                f'"action":"add","item":"{item}"}}'
            )
        if action_label == "stir":
            return (
                f'{{"reasoning":"{reasoning}",'
                '"action":"stir","item":null}'
            )
        if action_label == "finish":
            return (
                f'{{"reasoning":"{reasoning}",'
                '"action":"finish","item":null}'
            )
        raise ValueError(f"Unsupported action label for JSON guidance: {action_label!r}")

    def rejection_text(self, check: TemplateCheck) -> str:
        live_actions = check.snapshot_before_update.live_actions
        revealed_action = live_actions[0] if live_actions else None
        revealed_json = (
            self.action_label_to_json(revealed_action)
            if revealed_action is not None
            else ""
        )

        if check.classification in {CO_LIVE, UNCONSTRAINED}:
            if revealed_action is not None:
                return (
                    "ACTION REJECTED.\n"
                    "The proposed action has been repeated without making any progress and "
                    "is now blocked in the current cooking state. You must not propose it again.\n\n"
                    "The runtime shield has identified the correct next action. Return this "
                    "exact JSON object now:\n"
                    f"{revealed_json}\n\n"
                    "Do not choose a different action and do not output any additional text."
                )
            return (
                "ACTION REJECTED.\n"
                "The proposed action has been repeated without making any progress and "
                "is now blocked in the current cooking state. You must not propose it again.\n\n"
                "No valid progress action is available in the current automaton state. "
                "Reinspect the current image and recipe constraints.\n\n"
                "Output exactly one valid JSON action and no additional text."
            )

        if revealed_action is not None:
            return (
                "ACTION REJECTED.\n"
                f"Your previous proposal, {check.action_label}, was rejected and was not executed.\n\n"
                "The runtime shield has identified the correct next action. Return this "
                "exact JSON object now:\n"
                f"{revealed_json}\n\n"
                "Do not choose a different action and do not output any additional text."
            )

        return (
            "RUNTIME STRATEGY-TEMPLATE FEEDBACK:\n"
            f"Your previous proposal, {check.action_label}, was rejected and was not executed.\n\n"
            "That action is not permitted in the current automaton state, and no valid "
            "progress action is currently available. Reinspect the current image and recipe "
            "constraints.\n\n"
            "Return exactly one valid JSON action and no additional text outside the JSON object."
        )


def snapshot_to_jsonable(snapshot: TemplateSnapshot) -> Dict[str, Any]:
    return {
        "state_key": list(snapshot.state_key),
        "automaton_state_id": snapshot.automaton_state_id,
        "stage_name": snapshot.stage_name,
        "unsafe_actions": list(snapshot.unsafe_actions),
        "colive_actions": list(snapshot.colive_actions),
        "unconstrained_actions": list(snapshot.unconstrained_actions),
        "live_actions": list(snapshot.live_actions),
        "live_counter": snapshot.live_counter,
        "colive_counters": dict(snapshot.colive_counters),
        "unconstrained_self_loop_counters": dict(
            snapshot.unconstrained_self_loop_counters
        ),
        "base_distribution": dict(snapshot.base_distribution),
        "pre_prune_distribution": dict(snapshot.pre_prune_distribution),
        "modified_distribution": dict(snapshot.modified_distribution),
        "live_group_probability_mass": snapshot.live_group_probability_mass,
        "theta_prune": snapshot.theta_prune,
        "guidance_threshold": snapshot.guidance_threshold,
        "threshold_crossed": snapshot.threshold_crossed,
        "directive_active": snapshot.directive_active,
        "guidance_candidates": list(snapshot.guidance_candidates),
        "pruned_actions": list(snapshot.pruned_actions),
        "salt_add_count": snapshot.salt_add_count,
    }
