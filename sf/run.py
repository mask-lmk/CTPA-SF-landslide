"""Portable fixed-protocol current SF runner. SF is sf_lr_aggressive."""
import argparse,json,sys,os,shutil,hashlib
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'outputs/settings'));os.environ['YOLO_AUTOINSTALL']='false'
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds
import sf_lowrank
from sf_lowrank import compress_detector

def preserving_trainer():
 from ultralytics.models.yolo.detect import DetectionTrainer
 class PreserveCompressedTrainer(DetectionTrainer):
  def get_model(self,cfg=None,weights=None,verbose=True):
   assert weights is not None and self.data['nc']==1
   model=deepcopy(weights).float().cpu()
   if hasattr(model,'criterion'):delattr(model,'criterion')
   return model
 return PreserveCompressedTrainer

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluate(path,a,split,name):
 model=YOLO(str(path),task='detect');params=sum(p.numel() for p in model.model.parameters())
 m=model.val(data=str(a.data.resolve()),split=split,project=str(a.output.resolve()),name=name,device=a.device,batch=8,imgsz=640,workers=0,half=False,conf=.001,iou=.7,max_det=300,augment=False,plots=True,save_json=True)
 row=dict(checkpoint=str(Path(path).resolve()),checkpoint_sha256=sha(path),parameters=params,split=split,precision=float(m.box.mp),recall=float(m.box.mr),map50=float(m.box.map50),map5095=float(m.box.map))
 (Path(m.save_dir)/'metrics.json').write_text(json.dumps(row,indent=2),encoding='utf-8');return row

def main(a):
 torch.set_num_threads(4);init_seeds(a.seed,deterministic=True);a.output=a.output.resolve()
 if a.action=='evaluate':evaluate(a.checkpoint,a,a.split,'evaluation');return
 if a.action=='compress':
  model=YOLO(str(a.checkpoint),task='detect');model.model=compress_detector(model.model,a.variant);a.output.mkdir(parents=True,exist_ok=True);model.save(str(a.output/'initial.pt'));print(a.output/'initial.pt');return
 if a.action=='source':
  if a.model in ('yolo11n','ctpa'):
   cfg=ROOT/'ultralytics/cfg/models/11'/('yolo11n.yaml' if a.model=='yolo11n' else 'yolo11-cnn-former.yaml');model=YOLO(str(cfg),task='detect').load(str(a.pretrained))
  elif a.model in ('yolov8n','yolov10n'):model=YOLO(str(a.pretrained),task='detect')
  else:raise ValueError('SF is constructed from a trained CTPA checkpoint; use compress then recover')
  initial=None
 else:initial=a.checkpoint;model=YOLO(str(initial),task='detect')
 params=sum(p.numel() for p in model.model.parameters());variant=getattr(model.model,'compression_variant',None)
 if a.model=='sf':assert variant=='sf_lr_aggressive' or (a.action=='recover' and variant in sf_lowrank.VARIANTS)
 target=a.action=='adapt';kwargs=dict(data=str(a.data.resolve()),project=str(a.output.parent),name=a.output.name,epochs=200,patience=0,batch=8,imgsz=640,workers=0,device=a.device,optimizer='AdamW',lr0=.001 if a.action=='source' else .0001,lrf=.01,cos_lr=True,weight_decay=.0005,momentum=.937,warmup_epochs=3,nbs=64,seed=a.seed,deterministic=True,amp=True,degrees=0,translate=.1 if target else 0,scale=.3 if target else 0,shear=0,perspective=0,mosaic=.25 if target else 0,close_mosaic=20 if target else 10,mixup=0,flipud=0,fliplr=.5 if target else 0,hsv_h=.01 if target else 0,hsv_s=.2 if target else 0,hsv_v=.15 if target else 0,plots=True,save_period=25 if a.action=='source' else 50,resume=False)
 if a.action!='source':kwargs.update(trainer=preserving_trainer(),warmup_bias_lr=.0001,pretrained=str(initial),freeze=None)
 assert not a.output.exists(),a.output
 model.train(**kwargs);best=a.output/'weights/best.pt'
 if a.action!='source':
  restored=YOLO(str(best),task='detect');assert sum(p.numel() for p in restored.model.parameters())==params
  if variant:assert restored.model.compression_variant==variant
 # Only validation selects between initial and recovery; target adaptation selects trained best.
 selected=best
 if a.action=='recover':
  before=evaluate(initial,a,'val','initial_validation');after=evaluate(best,a,'val','recovery_validation')
  if .1*before['map50']+.9*before['map5095']>=.1*after['map50']+.9*after['map5095']:selected=initial
  (a.output/'selection.json').write_text(json.dumps(dict(selected=str(Path(selected).resolve()),sha256=sha(selected),selection='validation fitness; initial remains eligible'),indent=2),encoding='utf-8')
 print('Selected checkpoint:',selected)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=('source','compress','recover','adapt','evaluate'));p.add_argument('--model',choices=('yolo11n','ctpa','sf','yolov8n','yolov10n'),default='sf');p.add_argument('--data',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--pretrained',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,choices=(0,1,2),default=0);p.add_argument('--device',default='0');p.add_argument('--split',choices=('val','test'),default='test');p.add_argument('--variant',choices=tuple(sf_lowrank.VARIANTS),default='sf_lr_aggressive');main(p.parse_args())
