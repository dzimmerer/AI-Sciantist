## Experimentation

Each experiment runs on a singe machine with M GPUs. The training script runs for a **fixed time budget of n Hours ** (wall clock training time, excluding startup/compilation).

**What you CAN do:**
- Modify training scripts — this is the only files you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Config parameters 

**What you CANNOT do:**
- Modify validation. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify the evaluation harness.
- Modify the wandb logging.

**The goal is simple: get the lowest METRIC** Since the time budget is fixed, you don't need to worry about training time — it's always n hours. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful metric gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 metric improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 metric improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.