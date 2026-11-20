"""Count explicitly supported forward arithmetic at the actual input size.

Run with PyTorch 2.0.1 in the archived environment. No data or weights required.
The count is not a complete hardware instruction count or a latency estimate.
"""
import argparse,json,os,sys
from pathlib import Path

def run(repo,output):
    os.environ.setdefault('YOLO_AUTOINSTALL','false')
    sys.path.insert(0,str(repo))
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4); torch.manual_seed(42)
    configs=repo/'ultralytics/cfg/models/11'
    files={'yolo11n':configs/'yolo11n.yaml','ctpa':configs/'yolo11-cnn-former.yaml',
           'sf':configs/'Att_yaml/01/08_SharedFormer.yaml'}
    included=('aten::conv2d','aten::mm','aten::addmm','aten::bmm')
    rows=[]
    for key,path in files.items():
        model=YOLO(str(path),task='detect').model.float().eval()
        mha=[m for m in model.modules() if isinstance(m,torch.nn.MultiheadAttention)]
        for m in mha:
            assert m.dropout==0
            m.train(True) # explicit arithmetic path only; other layers remain in eval
        x=torch.zeros(1,3,640,640)
        with torch.no_grad():
            model(x)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                        record_shapes=True,with_flops=True) as prof: model(x)
        counts={e.key:int(e.flops) for e in prof.key_averages() if e.key in included}
        rows.append({'model':key,'parameters':sum(p.numel() for p in model.parameters()),
                     'counted_gflops':sum(counts.values())/1e9,'operator_flops':counts})
    data={'torch_version':torch.__version__,'input':[1,3,640,640],'precision':'FP32',
          'network':'unfused','multiply_accumulate_flops':2,'included':list(included),
          'excluded':'1D ECA convolution, normalization, activation, softmax, pooling, elementwise operations, data handling and NMS',
          'scope':'Supported-operator estimate at the full input; not a complete operation count or latency benchmark.',
          'mha':'Zero-dropout MHA uses the explicit arithmetic path; all other modules remain in evaluation mode.',
          'profiles':rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,default=Path('full_input_operations.json'))
    a=p.parse_args(); run(a.repo.resolve(),a.output.resolve())
