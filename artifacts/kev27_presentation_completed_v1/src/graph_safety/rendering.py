"""Chronological XML-compatible input, with optional binary graph annotations."""

import json
from xml.sax.saxutils import escape


def render(profile, prefix, parents=None):
    if parents is not None and set(parents) != set(prefix.candidate_ids + (prefix.event_id,)):
        raise ValueError("Graph must cover exactly the causal prefix")
    seen = set()

    def annotation(eid):
        if parents is None or eid in seen:
            return ""
        seen.add(eid)
        return ' tentative_parents="' + " ".join(parents[eid]) + '"'

    parts = [f"<agent_profile>\n{escape(profile)}\n</agent_profile>"]
    for message in prefix.observed_messages:
        attributes = f'role="{message["role"]}"'
        if "event_id" in message:
            eid = message["event_id"]
            attributes += f' event_id="{eid}"' + annotation(eid)
        parts.append(f"<message {attributes}>\n" + escape(json.dumps(message["payload"], ensure_ascii=False)) + "\n</message>")
    parts.append(f'<proposed_event id="{prefix.event_id}" role="agent"{annotation(prefix.event_id)}>\n'
                 + escape(json.dumps(prefix.proposed_action, ensure_ascii=False)) + "\n</proposed_event>")
    return "\n\n".join(parts)
