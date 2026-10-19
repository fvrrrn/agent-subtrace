"""Separate model-visible observations from evaluation-only annotations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInput:
    profile: str
    messages: tuple[dict, ...]

    @classmethod
    def from_legacy(cls, case):
        messages = []
        for message in case["messages"]:
            if message["role"] not in ("user", "agent", "assistant", "environment"):
                raise ValueError(f"Unsupported role: {message['role']}")
            messages.append({k: v for k, v in message.items()
                             if k in ("role", "content", "thought", "action")})
        return cls(case["profile"], tuple(messages))


@dataclass(frozen=True)
class Annotation:
    case_id: str
    trajectory_label: int | None
    direct_parents: dict[str, tuple[str, ...]] | None = None
    # Labels are intentionally absent from ModelInput and causal rendering APIs.
