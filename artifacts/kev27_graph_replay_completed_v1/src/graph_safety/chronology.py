"""Causal prefixes: proposed action is visible, its result is not."""

from dataclasses import dataclass
from .schema import ModelInput


@dataclass(frozen=True)
class Prefix:
    event_id: str
    observed_messages: tuple[dict, ...]
    proposed_action: dict
    candidate_ids: tuple[str, ...]


def prefixes(model_input: ModelInput):
    timeline, history = [], []
    last_agent = None
    for message in model_input.messages:
        role = message["role"]
        payload = {k: v for k, v in message.items() if k != "role"}
        if role == "user":
            timeline.append({"role": role, "payload": payload})
        elif role == "environment":
            if last_agent is None:
                event_id = f"E{len(history) + 1}"
                history.append(event_id)
            else:
                event_id = last_agent
            timeline.append({"role": role, "event_id": event_id, "payload": payload})
        else:
            event_id = f"E{len(history) + 1}"
            yield Prefix(event_id, tuple(timeline), payload, tuple(history))
            timeline.append({"role": role, "event_id": event_id, "payload": payload})
            history.append(event_id)
            last_agent = event_id
