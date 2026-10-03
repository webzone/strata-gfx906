"""CPU-only serialization/admission tests with mocked GPU copies; not hardware/model parity."""
import pathlib
import array
import math
from tools import gfx906_qsa_diagnostic as diagnostic
from tools.gfx906_prefill_validate import parse_arms
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAKE = r'''#pragma once
#include <cstring>
inline int gpu_calls = 0;
inline const char* test_arch = "gfx906:sramecc+";
using cudaError_t = int;
using cudaStream_t = void*;
constexpr int cudaSuccess = 0, cudaMemcpyDeviceToHost = 2;
inline const char* cudaGetErrorString(int) { return "mock GPU error"; }
inline int cudaGetDevice(int* x) { *x=0; ++gpu_calls; return 0; }
struct hipDeviceProp_t { const char* gcnArchName = test_arch; int warpSize = 64; };
inline int hipGetDeviceProperties(hipDeviceProp_t* p, int) { *p={}; ++gpu_calls; return 0; }
inline int cudaMemcpyAsync(void* dst, const void* src, size_t bytes, int, void*) {
    ++gpu_calls; std::memcpy(dst, src, bytes); return 0;
}
inline int cudaStreamSynchronize(void*) { ++gpu_calls; return 0; }
'''
DRIVER = r'''#include "strata/prefill/gfx906_diagnostic.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#define CHECK(x) do { if(!(x)) throw std::runtime_error(#x); } while(0)
int main(int argc,char** argv) {
  CHECK(argc==2); namespace fs=std::filesystem;
  using namespace strata::prefill::gfx906; using namespace strata::kernels;
  const fs::path root(argv[1]); fs::permissions(root,fs::perms::owner_all);
  AttentionCapture capture; std::string error; QsaShapes shape{}; QsaAttnPools pool{};
  CHECK(capture_attention(nullptr,pool,nullptr,nullptr,0,shape,0,0,0,nullptr,capture,error));
  CHECK(capture.directory.empty() && gpu_calls==0);
  CHECK(capture_logits(0,0,0,{},error)); CHECK(gpu_calls==0);
  setenv("STRATA_GFX906_QSA_CAPTURE","1",1);
#ifdef STRATA_USE_HIP
  CHECK(!capture_attention(nullptr,pool,nullptr,nullptr,0,shape,0,0,0,nullptr,capture,error));
  setenv("STRATA_GFX906_QSA_CAPTURE_POS","3",1);
  CHECK(capture_attention(nullptr,pool,nullptr,nullptr,0,shape,0,0,4,nullptr,capture,error));
  CHECK(gpu_calls==0);
  setenv("STRATA_GFX906_QSA_CAPTURE_POS","0",1);
  setenv("STRATA_GFX906_QSA_CAPTURE_DIR",root.c_str(),1);
  shape.n_head=24;shape.n_head_kv=2;shape.head_dim=256;shape.page_size=1;
  std::vector<float> q(5*24*256),attn(q.size());
  for(size_t i=0;i<q.size();++i) {q[i]=(float)i;attn[i]=(float)(i+7);}
  std::vector<int8_t> k(3*2*256),v(k.size());
  for(size_t i=0;i<k.size();++i) {k[i]=(int8_t)(i/512+11);v[i]=(int8_t)(i/512-3);}
  std::vector<uint16_t> ks(k.size()/64,0x3800),vs(ks.size(),0x3000);
  std::vector<int32_t> ids(5*2),steps(5*kStepCount),table={2,0,-1};
  for(int i=0;i<5;++i) {ids[i*2]=i==2 ? 2:0;ids[i*2+1]=1;steps[i*kStepCount+kStepWidth]=2;}
  pool.k_q=k.data();pool.v_q=v.data();pool.k_scale=ks.data();pool.v_scale=vs.data();pool.page_table=table.data();
  test_arch="gfx1010";
  CHECK(!capture_attention(q.data(),pool,ids.data(),steps.data(),2,shape,5,3,0,nullptr,capture,error));
  CHECK(!fs::exists(root/"device0-layer3-pos0"));test_arch="gfx906";
  CHECK(capture_attention(q.data(),pool,ids.data(),steps.data(),2,shape,5,3,0,nullptr,capture,error));
  CHECK((capture.rows==std::vector<int64_t>{0,2,4}));
  const fs::path path(capture.directory); CHECK(fs::file_size(path/"geometry.i64")==9*8);
  CHECK(fs::file_size(path/"q.f32")==3*24*256*4);CHECK(fs::file_size(path/"k.i8")==2*2*256);
  std::vector<int32_t> compact(3);std::ifstream f(path/"pages.i32",std::ios::binary);
  f.read((char*)compact.data(),12);CHECK((compact==std::vector<int32_t>{0,1,-1}));
  CHECK(finish_attention(capture,attn.data(),nullptr,error));
  CHECK(!finish_attention(capture,attn.data(),nullptr,error));
  CHECK(fs::file_size(path/"actual.f32")==3*24*256*4);
  CHECK((fs::status(path/"actual.f32").permissions()&fs::perms::all)==(fs::perms::owner_read|fs::perms::owner_write));
  CHECK(!capture_attention(q.data(),pool,ids.data(),steps.data(),2,shape,5,3,0,nullptr,capture,error));
#else
  CHECK(!capture_attention(nullptr,pool,nullptr,nullptr,0,shape,0,0,0,nullptr,capture,error));
  CHECK(gpu_calls==0);
#endif
  setenv("STRATA_GFX906_QSA_LOGITS","1",1);setenv("STRATA_GFX906_QSA_LOGITS_DIR",root.c_str(),1);
  CHECK(capture_logits(0,41,2,{1,2,3},error));
  CHECK(fs::file_size(root/"prediction0.i64")==5*8);CHECK(fs::file_size(root/"prediction0.f32")==3*4);
  CHECK(!capture_logits(0,41,2,{1,2,3},error));
  CHECK(!capture_logits(1,42,2,{std::numeric_limits<float>::infinity()},error));
  CHECK(!fs::exists(root/"prediction1.i64"));
  std::cout<<"CPU mocked-runtime capture serialization/admission passed; NOT hardware parity\n";
}
'''


class DiagnosticTests(unittest.TestCase):
    def test_pipe_capture_hook_precedes_commit_and_uses_last_device(self):
        # Static coverage regression only; real logits and model IDs still require hardware evidence.
        root = pathlib.Path(__file__).resolve().parents[1]
        source = (root/'src/program/generate.cpp').read_text()
        pipe = source[source.index('while (!cancelled && produced_n < max_new)'):]
        pipe = pipe[:pipe.index('const double decode_ms')]
        self.assertLess(pipe.index('::capture_logits('),pipe.index('ver.commit('))
        self.assertIn('produced_n + i < max_new',pipe)
        verify = (root/'src/core/verify.cpp').read_text().split('bool Verifier::copy_logits',1)[1]
        self.assertLess(verify.index('next_->copy_logits'),verify.index('OnDevice on_device(device_)'))
        self.assertIn('t >= last_t_',verify)

    def test_auto_arms_are_exact_and_duplicate_safe(self):
        self.assertEqual(parse_arms('2048:0:24,4096:1:24:auto'), [(2048,0,24,0),(4096,1,24,-1)])
        for value in ['4096:1:24:AUTO','4096:1:24:-1','4096:1:24:0,4096:1:24','8192:1:24:auto']:
            with self.assertRaises(ValueError):
                parse_arms(value)

    def test_logit_flip_tracks_only_common_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            dirs = [pathlib.Path(directory) / v for v in ('control','online')]
            for p in dirs:
                p.mkdir()
            for i in range(2):
                for p, values in zip(dirs, ([1.0,1.001], [1.002,1.001])):
                    token = max(range(2), key=lambda n: values[n])
                    (p / f'prediction{i}.i64').write_bytes(array.array('q',[1,i,99+i,token,2]).tobytes())
                    (p / f'prediction{i}.f32').write_bytes(array.array('f',values).tobytes())
            result = diagnostic.predictions(*dirs)
            self.assertEqual(result['first_divergence']['index'], 0)
            self.assertTrue(result['predictions'][0]['same_prior_ids'])
            self.assertFalse(result['predictions'][1]['same_prior_ids'])
            self.assertGreater(result['first_divergence']['flip_pair']['control_gap'], 0)
            self.assertLess(result['first_divergence']['flip_pair']['online_gap'], 0)
            (dirs[1] / 'prediction1.i64').unlink()
            with self.assertRaises(ValueError):
                diagnostic.predictions(*dirs)

    def test_numeric_analysis_rejects_nonfinite_and_short_files(self):
        with self.assertRaises(ValueError):
            diagnostic.metric([math.nan],[0.0])
        with tempfile.TemporaryDirectory() as directory:
            p = pathlib.Path(directory) / 'raw'
            p.write_bytes(b'123')
            with self.assertRaises(ValueError):
                diagnostic.raw(p,'f',1)

    def test_changed_operator_inputs_not_called_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            dirs = [pathlib.Path(directory) / v for v in ('control','online')]
            for root, value in zip(dirs, (0.0,1.0)):
                p = root / 'device0-layer3-pos0';p.mkdir(parents=True)
                (p/'geometry.i64').write_bytes(array.array('q',[1,1,1,1,1,1,3,0,1]).tobytes())
                (p/'q.f32').write_bytes(array.array('f',[value]*6144).tobytes())
                (p/'actual.f32').write_bytes(array.array('f',[value]*6144).tobytes())
                for name in ['query-rows.i64','k.i8','v.i8','k-scale.f16','v-scale.f16','ids.i32','steps.i32','pages.i32']:
                    (p/name).write_bytes(b'0')
            records = diagnostic.operators(*dirs)
            self.assertFalse(records[0]['isolated_operator_comparison'])
            self.assertFalse(records[0]['input_equal']['q.f32'])
            # Different compact-page coverage downstream is evidence of changed operands, not a malformed experiment.
            (dirs[1]/'device0-layer3-pos0'/'geometry.i64').write_bytes(array.array('q',[1,1,1,1,2,2,3,0,1]).tobytes())
            records = diagnostic.operators(*dirs)
            self.assertFalse(records[0]['isolated_operator_comparison'])
            self.assertNotEqual(records[0]['geometry'],records[0]['online_geometry'])

    def test_cpu_serialization_with_explicit_mock_runtime(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        self.assertIsNotNone(compiler, "C++17 compiler required, not a silent skip")
        for hip in (False, True):
            with self.subTest(hip=hip), tempfile.TemporaryDirectory() as directory:
                p = pathlib.Path(directory)
                (p / "cuda_runtime.h").write_text(FAKE)
                (p / "main.cpp").write_text(DRIVER)
                flags = ["-DSTRATA_USE_HIP=1", "-DSTRATA_EXPERIMENTAL_GFX906=1"] if hip else []
                result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", *flags,
                                         "-I" + str(p), "-I" + str(ROOT / "include"), str(p / "main.cpp"),
                                         str(ROOT / "src/prefill/gfx906_diagnostic.cpp"), "-o", str(p / "probe")],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                evidence = p / "evidence"
                evidence.mkdir(mode=0o700)
                result = subprocess.run([str(p / "probe"), str(evidence)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
