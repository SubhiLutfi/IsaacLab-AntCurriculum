"""Wrap a training command and kill it when converged.

Convergence = episode_length >= MAX_EP_LEN for STABLE_ITERS consecutive iterations.
Stops only at a checkpoint boundary (iter divisible by CHECKPOINT_INTERVAL),
so the last saved model is always clean.
"""
import subprocess
import sys
import re
import os

STABLE_ITERS = 20
MAX_EP_LEN = 5760
CHECKPOINT_INTERVAL = 100  # RSL-RL saves every 100 iters by default

ep_len_history = []
converged = False
current_iter = None

env = os.environ.copy()
env["TERM"] = "xterm"

proc = subprocess.Popen(
    sys.argv[1:],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env,
)

print(f"[monitor] PID={proc.pid}, will stop at ep_len={MAX_EP_LEN} for {STABLE_ITERS} iters, on checkpoint boundary (every {CHECKPOINT_INTERVAL})", flush=True)

for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()

    # Track current iteration
    m_iter = re.search(r"Learning iteration\s+(\d+)/\d+", line)
    if m_iter:
        current_iter = int(m_iter.group(1))

    # Track episode length
    m_ep = re.search(r"Mean episode length:\s+([\d.]+)", line)
    if m_ep:
        ep_len = float(m_ep.group(1))
        ep_len_history.append(ep_len)

        if not converged:
            if len(ep_len_history) >= STABLE_ITERS:
                count = sum(1 for v in ep_len_history[-STABLE_ITERS:] if v >= MAX_EP_LEN)
                if count == STABLE_ITERS:
                    converged = True
                    print(f"\n[monitor] CONVERGED at iter {current_iter} — waiting for next checkpoint boundary...", flush=True)

        # Stop only at a clean checkpoint boundary after convergence
        if converged and current_iter is not None and current_iter % CHECKPOINT_INTERVAL == 0:
            print(f"\n[monitor] Stopping at checkpoint boundary iter={current_iter}. Last saved model: model_{current_iter}.pt", flush=True)
            proc.terminate()
            proc.wait()
            break

proc.wait()
print(f"[monitor] Training finished (exit code {proc.returncode})", flush=True)
