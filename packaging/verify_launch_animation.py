"""Exercise the real pythonw splash while its owning process does no GUI work."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def run():
    root = Path(__file__).resolve().parents[1]
    out = root / 'examples/launch-animation-verified'
    out.mkdir(parents=True, exist_ok=True)
    (out/'animation.json').unlink(missing_ok=True)
    env = os.environ.copy()
    env['MLSTUDIO_SPLASH_EVIDENCE'] = str(out)
    command = [str(root/'.venv/Scripts/pythonw.exe'), str(root/'src/mlstudio/launch_splash.py')]
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    process = subprocess.Popen(command, stdin=subprocess.PIPE, env=env, text=True, creationflags=flags)
    try:
        # Deliberately leave the owning thread blocked: the animation must live.
        time.sleep(2)
        assert process.poll() is None, 'Splash exited before the IDE was ready'
        report = json.loads((out/'animation.json').read_text())
        assert report['frames'] > 12, report
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        changed = digest(out/'launch-first.png') != digest(out/'launch-animated.png')
        assert changed or report['reduced_motion'], 'Two animation frames were identical'
        process.stdin.write('stage:Preparing your workspace\n')
        process.stdin.flush()
        time.sleep(.15)
        process.stdin.write('ready\n')
        process.stdin.flush()
        assert process.wait(timeout=5) == 0, 'Ready transition did not close the splash'
    finally:
        if process.poll() is None:process.kill();process.wait()
        process.stdin.close()
    # A crash/owner exit closes the private pipe; no orphan splash may remain.
    env.pop('MLSTUDIO_SPLASH_EVIDENCE')
    orphan = subprocess.Popen(command, stdin=subprocess.PIPE, env=env, text=True, creationflags=flags)
    try:
        time.sleep(.7)
        orphan.stdin.close()
        assert orphan.wait(timeout=5) == 0
    finally:
        if orphan.poll() is None:orphan.kill();orphan.wait()
    result = {'passed': True, 'frames_during_blocked_owner': report['frames'],
              'rendered_frames_differ': changed, 'reduced_motion': report['reduced_motion'],
              'ready_transition_closes_splash': True, 'owner_exit_closes_splash': True}
    (out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':run()
