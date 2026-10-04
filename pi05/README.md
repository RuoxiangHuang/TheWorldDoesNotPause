# π₀.₅

Import name: `fasterpi`.

| Preset | Stack |
|---|---|
| `pace` | `chunk_residual_cache` + `step_cache` |
| `ours`, `fasterpi` | same as `pace` |
| `baseline` | eager Euler |

`c3` resolves to `chunk_residual_cache`. `d1` resolves to `step_cache`.

```bash
export PYTHONPATH="$PWD/pi05:$PYTHONPATH"
python -m pytest pi05/tests/test_unit.py -q
```
