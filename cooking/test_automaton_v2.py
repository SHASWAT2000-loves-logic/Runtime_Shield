"""Small deterministic smoke test for the automaton-liveness-v2 shield logic.

This test does not start MuJoCo or contact the model server. It verifies the
action partition, probability-driven pruning of repeated state-preserving
unconstrained actions, exact answer-revealing rejection feedback, state-change
reset, three-increment salt rule, and cooking-window memory.
"""

from cooking_automaton import LIVE_PROGRESS, UNCONSTRAINED, UNSAFE, CookingAutomaton
from cooking_strategy_template_guide import ManualCookingStrategyTemplateGuide

ACTION_LABELS = (
    "add:onion",
    "add:garlic",
    "add:tomato",
    "add:turmeric",
    "add:coriander",
    "add:salt",
    "add:cream",
    "stir",
    "finish",
)
ITEMS = ("onion", "garlic", "tomato", "turmeric", "coriander", "salt", "cream")


def make_state(**added: bool):
    return {item: {"added": bool(added.get(item, False))} for item in ITEMS}


def main() -> None:
    automaton = CookingAutomaton(action_labels=ACTION_LABELS, max_salt_additions=3)
    guide = ManualCookingStrategyTemplateGuide(
        automaton=automaton,
        action_labels=ACTION_LABELS,
        gamma=0.20,
        theta_prune=0.10,
        guidance_threshold=0.95,
    )

    # Tomato is missing while both aromatics are complete. Repeated garlic is a
    # state-preserving unconstrained self-loop. Its probability should decay
    # action-specifically and theta_prune should eventually reject it, without
    # pruning the unchosen onion self-loop.
    state = make_state(onion=True, garlic=True)
    node = automaton.node(state)
    assert node.live_actions == ("add:tomato",)
    assert node.unconstrained_actions == ("add:onion", "add:garlic")
    assert node.colive_actions == ("stir",)

    accepted_repetitions = 0
    while accepted_repetitions < 20:
        check = guide.check_action(state, "add:garlic")
        if not check.allowed:
            break
        guide.observe_accepted_action(state, "add:garlic")
        accepted_repetitions += 1
    else:
        raise AssertionError("repeated garlic never crossed theta_prune")

    assert accepted_repetitions > 0
    snapshot = guide.snapshot(state)
    assert "add:garlic" in snapshot.pruned_actions
    assert snapshot.modified_distribution["add:garlic"] == 0.0
    assert not guide.check_action(state, "add:garlic").allowed
    assert guide.check_action(state, "add:onion").allowed
    assert "add:onion" not in snapshot.pruned_actions

    rejection = guide.rejection_text(guide.check_action(state, "add:garlic"))
    assert rejection.startswith("ACTION REJECTED.")
    assert "You must not propose it again." in rejection
    assert "The runtime shield has identified the correct next action." in rejection
    assert (
        '{"reasoning":"The runtime shield identified the required progress action.",'
        '"action":"add","item":"tomato"}'
    ) in rejection
    assert "Do not choose a different action" in rejection

    unsafe_rejection = guide.rejection_text(guide.check_action(state, "add:cream"))
    assert unsafe_rejection.startswith("ACTION REJECTED.")
    assert '"action":"add","item":"tomato"' in unsafe_rejection

    # When more than one progress action is valid, reveal one deterministic exact
    # action using the automaton's existing action order rather than inventing a
    # new planner or changing the one-action-per-request interface.
    spice_automaton = CookingAutomaton(action_labels=ACTION_LABELS)
    spice_guide = ManualCookingStrategyTemplateGuide(
        automaton=spice_automaton,
        action_labels=ACTION_LABELS,
        gamma=0.20,
        theta_prune=0.10,
        guidance_threshold=0.95,
    )
    spice_state = make_state(onion=True, garlic=True, tomato=True)
    assert spice_automaton.node(spice_state).live_actions == (
        "add:turmeric",
        "add:coriander",
        "add:salt",
    )
    spice_rejection = spice_guide.rejection_text(
        spice_guide.check_action(spice_state, "add:cream")
    )
    assert '"action":"add","item":"turmeric"' in spice_rejection
    assert '"item":"coriander"' not in spice_rejection

    # A cooking-state change resets the repeated-self-loop penalty. Returning to
    # the earlier observable state starts the action-specific counter from zero.
    progressed_state = make_state(onion=True, garlic=True, tomato=True)
    guide.snapshot(progressed_state)
    returned_state = make_state(onion=True, garlic=True)
    returned_snapshot = guide.snapshot(returned_state)
    assert returned_snapshot.unconstrained_self_loop_counters["add:garlic"] == 0
    assert guide.check_action(returned_state, "add:garlic").allowed

    # First salt is live, second and third are unconstrained state-changing
    # transitions, and fourth is unsafe. They must not be treated as repeated
    # state-preserving self-loops.
    automaton = CookingAutomaton(action_labels=ACTION_LABELS, max_salt_additions=3)
    guide = ManualCookingStrategyTemplateGuide(
        automaton=automaton,
        action_labels=ACTION_LABELS,
        gamma=0.20,
        theta_prune=0.10,
        guidance_threshold=0.95,
    )
    state = make_state(
        onion=True,
        garlic=True,
        tomato=True,
        turmeric=True,
        coriander=True,
    )
    assert automaton.classify(state, "add:salt") == LIVE_PROGRESS
    automaton.observe_accepted_action(
        task_state_before_action=state,
        action_label="add:salt",
        physically_executed=True,
    )
    state["salt"]["added"] = True
    assert automaton.classify(state, "add:salt") == UNCONSTRAINED
    guide.observe_accepted_action(state, "add:salt")
    assert "add:salt" not in guide.snapshot(state).unconstrained_self_loop_counters
    automaton.observe_accepted_action(
        task_state_before_action=state,
        action_label="add:salt",
        physically_executed=False,
    )
    assert automaton.classify(state, "add:salt") == UNCONSTRAINED
    guide.observe_accepted_action(state, "add:salt")
    assert "add:salt" not in guide.snapshot(state).unconstrained_self_loop_counters
    automaton.observe_accepted_action(
        task_state_before_action=state,
        action_label="add:salt",
        physically_executed=False,
    )
    assert automaton.classify(state, "add:salt") == UNSAFE

    # Once tomato has closed the aromatics window, a rollback does not make a
    # late raw-onion addition legal again.
    automaton = CookingAutomaton(action_labels=ACTION_LABELS)
    state = make_state(onion=True, garlic=True, tomato=True)
    automaton.node(state)
    state["onion"]["added"] = False
    assert automaton.node(state).stage_name == "irrecoverable_order_violation"
    assert automaton.classify(state, "add:onion") == UNSAFE

    print("AUTOMATON V2 ANSWER-REVEALING REPROMPT SMOKE TEST PASSED")
    print("Accepted garlic repetitions before theta_prune:", accepted_repetitions)
    print("Final shield distribution:", snapshot.modified_distribution)


if __name__ == "__main__":
    main()
