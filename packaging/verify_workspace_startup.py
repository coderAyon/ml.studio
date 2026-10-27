"""Launch the actual stable EXE twice in an isolated diagnostic instance."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def run():
    root=Path(__file__).resolve().parents[1];out=root/'examples/startup-verified';out.mkdir(parents=True,exist_ok=True)
    for name in ['ready.json','activation.json']:(out/name).unlink(missing_ok=True)
    env=os.environ.copy();env['MLSTUDIO_TEST_INSTANCE']='-'+uuid.uuid4().hex[:10]
    env['MLSTUDIO_STARTUP_EVIDENCE']=str(out);env['MLSTUDIO_HOME']=str(out/'storage')
    started=time.perf_counter();pid=None
    try:
        subprocess.Popen([str(root/'MLStudio.exe')],cwd=root,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+45
        while not (out/'ready.json').exists() and time.monotonic()<deadline:time.sleep(.1)
        if not (out/'ready.json').exists():raise RuntimeError('The first EXE launch did not show its main window within 45 seconds.')
        ready=json.loads((out/'ready.json').read_text());pid=ready['pid'];elapsed=time.perf_counter()-started
        subprocess.Popen([str(root/'MLStudio.exe')],cwd=root,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+20
        while not (out/'activation.json').exists() and time.monotonic()<deadline:time.sleep(.1)
        activated=json.loads((out/'activation.json').read_text());assert activated['pid']==pid
        result={'passed':True,'first_exe_launch_seconds':round(elapsed,2),'seconds_after_splash':round(ready['seconds_after_splash'],2),'second_launch_activated_same_pid':True}
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:
        if pid:subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],capture_output=True)


if __name__=='__main__':run()
