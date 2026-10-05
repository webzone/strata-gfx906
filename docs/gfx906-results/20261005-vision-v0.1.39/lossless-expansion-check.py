import hashlib,json,pathlib,sys,numpy as np
b=pathlib.Path('/data/strata-gfx906/vision-gfx906-20261005')
sys.path.insert(0,str(b))
from gfx906_vision_check import read_sve,sha256
rows=[]
for stem in ['square','wide','tall','large','photo']:
    row={'image':stem}
    for mode in ['cpu','hip']:
        ha,a=read_sve(b/'compare-f32-300-gpu0'/f'{mode}-300-gpu0-{stem}-0.sve')
        hb,v=read_sve(b/'accepted-300-gpu0'/f'{mode}-300-gpu0-{stem}-2.sve')
        assert ha==hb and np.array_equal(a,v)
        row[mode+'_bitwise_equal']=True
    rows.append(row)
d={'source_mmproj_sha256':sha256(b/'mmproj-Qwen3.8-Flash-Next-BF16.gguf'),'independent_expanded_gguf_sha256':sha256(b/'probe-mmproj-F32.gguf'),'method':'Independent Python GGUF rewrite expands each BF16 uint16 by uint32 << 16, then views float32. Both CPU and HIP final embeddings are bitwise identical to the in-memory loader expansion on all five inputs. Derivative is diagnostic only; runtime reads original BF16 GGUF.','results':rows}
(b/'lossless-expansion-check.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))
