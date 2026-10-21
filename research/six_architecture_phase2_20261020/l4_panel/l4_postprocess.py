import json,pathlib,time
r=pathlib.Path('/content/tam-l4-panel-20261020')
while True:
    if (r/'STATUS.json').exists():
        status=json.loads((r/'STATUS.json').read_text())
        if status.get('phase')=='error':raise RuntimeError('Controller error; inspect before postprocessing')
    if (r/'EVIDENCE.json').exists() and json.loads((r/'EVIDENCE.json').read_text()).get('complete'):break
    time.sleep(20)
import train_panel as p
p.ROOT=r;p.OUT=r/'results';p.DATA=r/'data'
import inference_benchmark as i
i.ROOT=r;i.main()
import language_benchmarks as b
b.ROOT=r;b.main()
