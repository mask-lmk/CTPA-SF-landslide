"""Recreate frozen positive-only Bijie or registered CAS subsets; no split search."""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
def boxes(mask,width,height):
 n,_,stats,_=cv2.connectedComponentsWithStats((mask>0).astype('uint8'),connectivity=8)
 return [f'0 {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}' for x,y,w,h,area in stats[1:n] if area>=1]
def main(a):
 out=a.output.resolve();assert not out.exists(),out
 if a.kind=='bijie':
  rows=json.loads((ROOT/'splits/bijie_manifest.json').read_text(encoding='utf-8'))
  raw=a.raw/'landslide';index={p.stem:p for p in (raw/'image').iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff')}
  for r in rows:
   split=r['split'];sid=r['source_id'];stem=sid.removeprefix('landslide_');target=sid+('__flip' if r['transform']=='horizontal_flip' else '')
   im=Image.open(index[stem]).convert('RGB');w,h=im.size;mask=np.asarray(Image.open(raw/'mask'/(stem+'.png')).convert('L'));lines=boxes(mask,w,h)
   assert len(lines)==r['instances']
   if r['transform']=='horizontal_flip':
    im=ImageOps.mirror(im);lines=[f'{c} {1-float(x):.10f} {y} {bw} {bh}' for c,x,y,bw,bh in (s.split() for s in lines)]
   images=out/'images'/split;labels=out/'labels'/split;images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
   # OpenCV PNG encoding matches the original source preparation; RGB pixels preserved.
   cv2.imwrite(str(images/(target+'.png')),np.asarray(im)[:,:,::-1]);(labels/(target+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
  config='path: '+json.dumps(out.as_posix())+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [landslide]\n'
 else:
  rows=json.loads((ROOT/'splits'/a.domain/'manifest.json').read_text(encoding='utf-8'));index={p.stem:p for p in (a.raw/'img').iterdir() if p.is_file()};masks={p.stem:p for p in (a.raw/'mask').iterdir() if p.is_file()}
  lists={s:[] for s in ('train','val','test')}
  for r in rows:
   sid=r['id'];im=Image.open(index[sid]).convert('RGB');w,h=im.size;mask=np.asarray(Image.open(masks[sid]).convert('L'));lines=boxes(mask,w,h);assert len(lines)==r['instances']
   images=out/'images/all';labels=out/'labels/all';images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
   im.save(images/(sid+'.png'));(labels/(sid+'.txt')).write_text('\n'.join(lines),encoding='utf-8');lists[r['split']].append((images/(sid+'.png')).as_posix())
  for split,paths in lists.items():(out/(split+'.txt')).write_text('\n'.join(sorted(paths))+'\n',encoding='utf-8')
  config='path: '+json.dumps(out.as_posix())+'\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames: [landslide]\n'
 (out/'data.yaml').write_text(config,encoding='utf-8');print(out/'data.yaml')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('kind',choices=('bijie','cas'));p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--domain',choices=('jiuzhai_05','moxi_02','moxi_1','mengdong_05','longxi_sat_05'));main(p.parse_args())
