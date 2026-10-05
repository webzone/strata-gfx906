import sys,pathlib,numpy as np
root=pathlib.Path('/home/chris/dev/strata-gfx906')
base=pathlib.Path('/data/strata-gfx906/vision-gfx906-20261005')
sys.path.insert(0,str(root/'third_party/llama.cpp/gguf-py'))
import gguf
src=base/'mmproj-Qwen3.8-Flash-Next-BF16.gguf'
dst=base/'probe-mmproj-F32.gguf'
assert not dst.exists()
r=gguf.GGUFReader(src)
w=gguf.GGUFWriter(dst,'clip')
for key,f in r.fields.items():
    if key.startswith('GGUF.') or key=='general.architecture': continue
    w.add_key_value(key,f.contents(),f.types[0],f.types[-1] if len(f.types)>1 else None)
for t in r.tensors:
    a=t.data
    typ=t.tensor_type
    if typ==gguf.GGMLQuantizationType.BF16:
        a=(a.view('<u2').astype('<u4')<<16).view('<f4').reshape(tuple(reversed(t.shape)))
        typ=gguf.GGMLQuantizationType.F32
    w.add_tensor(t.name,a,raw_dtype=typ)
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
print(dst,dst.stat().st_size,flush=True)
