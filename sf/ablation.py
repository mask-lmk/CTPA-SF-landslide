"""Current SF component ablations, no automatic model replacement."""
import argparse
from copy import deepcopy
from pathlib import Path
import run as R
from sf_lowrank import compress_detector

def main(a):
 if a.action=='initialize':
  teacher=R.YOLO(str(a.checkpoint),task='detect');model=compress_detector(teacher.model,'sf_lr_aggressive')
  block=next(m for m in model.modules() if type(m).__name__=='CNNTransformerParallel');base=next(m for m in teacher.model.modules() if type(m).__name__=='CNNTransformerParallel')
  if a.restore=='attention':block.attn=deepcopy(base.attn)
  elif a.restore=='ffn':block.ffn[0]=deepcopy(base.ffn[0]);block.ffn[2]=deepcopy(base.ffn[2])
  else:block.fuse[0]=deepcopy(base.fuse[0])
  model.ablation_variant=a.restore+'_dense';model.compression_variant='ablation_'+model.ablation_variant
  block.np=sum(p.numel() for p in block.parameters());teacher.model=model;a.output.mkdir(parents=True,exist_ok=True);teacher.save(str(a.output/'initial.pt'));return
 a.action='recover';a.model='ctpa';R.main(a)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=('initialize','recover'));p.add_argument('--restore',choices=('attention','ffn','fusion'));p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,choices=(0,1,2),default=0);p.add_argument('--device',default='0');a=p.parse_args();main(a)
