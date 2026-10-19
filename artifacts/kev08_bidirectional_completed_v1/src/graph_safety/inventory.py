"""Content-addressed index of historical report files, without rewriting history."""

from pathlib import Path

from .io import read_json, sha256, write_json


def inventory(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    entries = []
    names = {"summary.json", "results.json", "statistics.json", "plan.json", "protocol.json", "selection.json", "holdout_report.json", "transfer_report.json"}
    for path in sorted(source.rglob("*.json")):
        if path.name.lower() not in names:
            continue
        entry = {"path": str(path.relative_to(source)), "sha256": sha256(path), "bytes": path.stat().st_size}
        try:
            value = read_json(path)
            entry["kind"] = "object" if isinstance(value, dict) else "non_object"
            if isinstance(value, dict):
                entry["status"] = value.get("status")
                entry["top_level_fields"] = sorted(value)
            else:
                entry["audit_flag"] = "Report/configuration filename contains non-object data; inspect historical filename collisions."
        except (ValueError, UnicodeError) as error:
            entry["audit_flag"] = str(error)
        entries.append(entry)
    write_json(output, {"source_root": str(source.resolve()), "entries": entries,
                       "scope": "Metadata/report index, not validation of the underlying inference or scientific claims."})
    return {"files_indexed": len(entries), "flagged": sum("audit_flag" in e for e in entries), "output": str(output)}
