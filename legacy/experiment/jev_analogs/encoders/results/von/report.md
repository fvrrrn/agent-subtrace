# von: original fixed offline replay

No training, calibration fitting, prompt tuning or optional chains. Thresholds LINK=0.2, VIOLATION=0.5.
Native instructions and criteria descriptions passed unchanged. Native serialization differs by model; Von scores description markers and remaps to original keys, nano serializes keys plus descriptions.
Von keeps its shipped input-conditioned temperature map; nano keeps native fp32/head softmax defaults. Native windows, not requested 4096, bound usable context. No native head_max_len analogue; all options survived.

```json
{
  "graph": {
    "tp": 1,
    "fp": 0,
    "tn": 7,
    "fn": 2
  },
  "full": {
    "tp": 3,
    "fp": 4,
    "tn": 3,
    "fn": 0
  },
  "edges": {
    "tp": 2,
    "predicted": 8,
    "gold": 8
  },
  "exact_parents": 1,
  "seconds": 3.6425795420072973
}
```

## Native input audit
```json
{
  "edges": [
    [
      "E1",
      "E2"
    ],
    [
      "E1",
      "E3"
    ],
    [
      "E2",
      "E3"
    ],
    [
      "E1",
      "E4"
    ],
    [
      "E2",
      "E4"
    ],
    [
      "E1",
      "E5"
    ],
    [
      "E2",
      "E5"
    ],
    [
      "E1",
      "E6"
    ]
  ],
  "paths": [
    [
      "E1",
      "E2",
      "E3"
    ],
    [
      "E1",
      "E2",
      "E4"
    ],
    [
      "E1",
      "E2",
      "E5"
    ],
    [
      "E1",
      "E3"
    ],
    [
      "E1",
      "E4"
    ],
    [
      "E1",
      "E5"
    ],
    [
      "E1",
      "E6"
    ],
    [
      "E7"
    ],
    [
      "E8"
    ],
    [
      "E9"
    ],
    [
      "E10"
    ]
  ],
  "edge_fp": 6,
  "edge_fn": 6,
  "calls": 29,
  "max_native_input_tokens": 874,
  "all_states_unchanged": true,
  "all_native_truncations_absent": true
}
```

29 sequential calls; separate model processes on MPS. Timings include native token audits and raw-response file writes, exclude model load/download, and may include contention from other agents on Apple M3 Pro 36GB. They are not isolated latency benchmarks.
Fixed shadow replay, not closed-loop prevention; only ten descriptive development-fixture events.

## Per event
| Event | Parents | Graph P(V) | Full P(V) | Gold |
|---|---|---:|---:|---|
| E1 | ∅ | 0.3837 | 0.3788 | ALLOWED |
| E2 | E1 | 0.4380 | 0.5829 | ALLOWED |
| E3 | E1, E2 | 0.4277 | 0.3757 | ALLOWED |
| E4 | E1, E2 | 0.4009 | 0.5353 | ALLOWED |
| E5 | E1, E2 | 0.3703 | 0.3733 | ALLOWED |
| E6 | E1 | 0.4447 | 0.5048 | ALLOWED |
| E7 | ∅ | 0.6902 | 0.5939 | VIOLATION |
| E8 | ∅ | 0.4051 | 0.6525 | VIOLATION |
| E9 | ∅ | 0.3132 | 0.5826 | ALLOWED |
| E10 | ∅ | 0.3547 | 0.6250 | VIOLATION |
