# Pytest test tiers

Collection assigns every test exactly one tier marker by statically reading
its module source through `item.fspath`. It does not import any test module
beyond pytest's normal collection. Classification uses this precedence:

1. `reproduction_legacy`: `tests/test_gate2_candidate*.py`, the frozen
   historical reproduction ledger.
2. `oracle_policyengine`: modules that reference
   `POPULACE_DYNAMICS_PE_US_DIR` or the legacy default
   `~/PolicyEngine/policyengine-us` checkout.
3. `integration_psid`: modules that reference
   `POPULACE_DYNAMICS_PSID_DIR` or the default `~/PolicyEngine/psid-data`
   root.
4. `artifact`: modules that read committed `runs/*.json` evidence artifacts.
5. `unit`: all remaining tests.

The count manifest in `tier_counts.json` is enforced during a full-suite
collection. Update it deliberately whenever tests move between tiers or the
suite grows.

## `-m` selection smoke at HEAD

Run each tier independently from the repository root:

```sh
pytest --collect-only -q -m unit | tail -1
pytest --collect-only -q -m artifact | tail -1
pytest --collect-only -q -m integration_psid | tail -1
pytest --collect-only -q -m reproduction_legacy | tail -1
pytest --collect-only -q -m oracle_policyengine | tail -1
```

| Tier | Tests at HEAD |
|---|---:|
| `unit` | 187 |
| `artifact` | 664 |
| `integration_psid` | 780 |
| `reproduction_legacy` | 520 |
| `oracle_policyengine` | 148 |
| **Total** | **2,299** |

The gate-W1 transport floors (`tests/test_gate_w1_floors.py`,
`tests/test_gate_w1_derivations.py`) are `artifact` tier — they read the
committed `runs/gate_w1_floors_v1.json` and never load the certified frame at
collection; the certified-frame reproduction pin inside the floors module
skips unless `huggingface_hub` + `tables` are importable and the pinned h5 is
already in the local Hugging Face cache. The family-A moment unit tests
(`tests/data/test_deployment_frame.py`) run on synthetic frames only, so they
are `unit` tier.
