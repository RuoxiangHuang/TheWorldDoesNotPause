# π₀.₅ PACE

π₀.₅ 的加速栈。导入名是 `fasterpi`。论文里的 PACE 对应 `pace`（别名 `ours`、`fasterpi`）：

`chunk_residual_cache + step_cache`

`chunk_residual_cache` 是跨次重规划的残差复用，`step_cache` 是 chunk 内速度保持。旧规格里的 `c3`、`d1` 仍会解析成这两个名字。`baseline` 恢复 eager Euler。

```bash
export PYTHONPATH="/DATA/YuanZhen/TheWorldDoesNotPause/pi05:$PYTHONPATH"
python -m pytest /DATA/YuanZhen/TheWorldDoesNotPause/pi05/tests/test_unit.py -q
```

权重和 openpi 仍在仓库外。
