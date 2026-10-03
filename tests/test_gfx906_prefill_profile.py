"""CPU-only profile schema/experiment-plan tests; never launch a GPU or server."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.gfx906_prefill_profile import make_plan, read_records, replace_arg, summarize
from tools.hip import bench_prefill


def stage(device=0, begin=0, end=24, chunk=2048):
    return dict(schema=1, device=device, layer_begin=begin, layer_end=end, pos0=0, tokens=2048, chunk=chunk,
                ms_wall=100, ms_gpu_timeline=80, ms_group_wait=12, ms_group_cpu=2,
                phase_ms={"QSA": 20, "GDN": 30, "GU": 30}, mmq_layer_chunks=20,
                fp16_layer_chunks=4, fallback_layers=[begin + 1], experts_streamed=5)


def raw(kind, record):
    return f"strata prefill {kind}: " + json.dumps(record) + "\n"


class ProfileTests(unittest.TestCase):
    def test_devices_and_ranges_never_become_one_wall_time(self):
        log = "ordinary engine log\n" + raw("profile", stage()) + raw("profile", stage(1, 24, 48))
        report = summarize(read_records(log))
        self.assertEqual(len(report["stages"]), 2)
        self.assertNotIn("ms_wall", report)
        self.assertNotIn("model_tps", report)
        self.assertEqual([r["ms_stage_wall_sum"] for r in report["stages"]], [100, 100])
        self.assertEqual(report["stages"][0]["phase_percent_of_device_timeline"]["QSA"], 25)

    def test_repeated_calls_are_deltas(self):
        report = summarize(read_records(raw("profile", stage()) * 2))
        group = report["stages"][0]
        self.assertEqual(group["tokens"], 4096)
        self.assertEqual(group["counters"]["experts_streamed"], 10)
        self.assertEqual(group["host_ms"]["ms_group_cpu"], 4)
        self.assertEqual(group["fallback_layers"], [1])

    def test_chunks_remain_separate(self):
        report = summarize(read_records(raw("profile", stage()) + raw("profile", stage(chunk=4096))))
        self.assertEqual([s["chunk"] for s in report["stages"]], [2048, 4096])

    def test_no_records_does_not_invent_results(self):
        report = summarize(read_records("old unrelated engine log\n"))
        self.assertEqual(report["stages"], [])
        self.assertEqual(report["requests"], [])

    def test_nonfinite_negative_and_bad_schemas_are_rejected(self):
        for mutate in (
            lambda r: r.update(schema=2), lambda r: r.update(schema=True), lambda r: r.update(ms_wall=-1),
            lambda r: r.update(ms_wall=float("nan")), lambda r: r.update(device=True), lambda r: r.update(chunk=0),
            lambda r: r.update(layer_end=-1), lambda r: r.update(phase_ms={"QSA": float("inf")}),
            lambda r: r.pop("ms_gpu_timeline"),
        ):
            r = stage(); mutate(r)
            with self.subTest(record=r), self.assertRaises(ValueError):
                read_records(raw("profile", r))
        with self.assertRaisesRegex(ValueError, "line 1"):
            read_records("strata prefill profile: not JSON")

    def test_request_cold_reuse_and_cancel_markers(self):
        r = dict(schema=1, prompt_tokens=2049, reused=0, read_from=0, prefill_rows=2048,
                 cancelled=False, ms_wall=110, ms_refill=10, refilled_slots=80)
        records = []
        for reused, read_from, cancelled in ((0,0,False), (1024,1024,False), (0,0,True), (0,100,False)):
            records.append(raw("request", dict(r, reused=reused, read_from=read_from, cancelled=cancelled)))
        report = summarize(read_records("".join(records)))
        self.assertEqual([r["cold"] for r in report["requests"]], [True, False, False, False])
        self.assertNotIn("tok_per_s", report["requests"][0])

    def test_draft_batch_marker_is_preserved(self):
        r = dict(schema=1, device=1, pos0=2048, tokens=2048, batched=True, ms_wall=15)
        self.assertEqual(summarize(read_records(raw("draft", r)))["draft"], [r])

    def test_request_envelopes_keep_warmup_out_of_cold_stage_totals(self):
        request = dict(schema=1, prompt_tokens=2049, reused=0, read_from=0, prefill_rows=2048,
                       cancelled=False, ms_wall=100, ms_refill=1, refilled_slots=4)
        log = raw("profile", stage()) + raw("request", request)
        log += raw("profile", stage()) * 2 + raw("request", dict(request, prompt_tokens=4097, prefill_rows=4096))
        report = summarize(read_records(log))
        self.assertEqual([r["stages"][0]["calls"] for r in report["request_profiles"]], [1, 2])
        self.assertEqual([r["request"]["prefill_rows"] for r in report["request_profiles"]], [2048, 4096])
        self.assertEqual(report["stages"][0]["calls"], 3) # Aggregate is still available, explicitly separate.

    def test_layer_rows_and_format_specific_phases(self):
        r=dict(schema=1,device=0,layer=8,pos0=0,tokens=2048,chunk=2048,gu_type=29,down_type=42,
               row_calls=1,routed_rows=20480,max_rows=130,
               row_bin_upper=[0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,None],
               row_histogram=[0,0,0,0,0,0,100,400,11,1,0,0,0,0,0,0],phase_ms={'fp16 gate/up':10,'fp16 down':5})
        report=summarize(read_records(raw('layer',r)*2));group=report['layers'][0]
        self.assertEqual(group['row_calls'],2);self.assertEqual(sum(group['row_histogram']),1024)
        self.assertEqual(group['phase_ms']['fp16 down'],10);self.assertEqual(group['max_rows'],130)
        for mutate in [lambda x:x.update(row_histogram=[0]),lambda x:x.update(layer=True),
                       lambda x:x.update(phase_ms={'fp16 down':float('nan')}),lambda x:x.update(row_bin_upper=[0]*16)]:
            bad=copy.deepcopy(r);mutate(bad)
            with self.assertRaises(ValueError):read_records(raw('layer',bad))

    def test_product_counts_actual_group_rows_not_activation_stride(self):
        r=dict(schema=1,device=1,layer=37,pos0=0,type='q2_0',groups=32,total_rows=20480,group_rows=640,
               max_rows=80,weight_rows=2560,weight_cols=640,requested_j=32,selected_j=32,forced=True)
        request=dict(schema=1,prompt_tokens=2049,reused=0,read_from=0,prefill_rows=2048,cancelled=False,
                     ms_wall=100,ms_refill=0,refilled_slots=0)
        report=summarize(read_records(raw('mmq',r)+raw('request',request)+raw('mmq',dict(r,group_rows=80))+raw('request',request)))
        self.assertEqual(report['products'][0]['group_rows'],720)
        self.assertNotIn('total_rows',report['products'][0])
        self.assertEqual([p['products'][0]['group_rows'] for p in report['request_profiles']],[640,80])
        for key,value in [('group_rows',20481),('forced',1),('selected_j',-1)]:
            with self.assertRaises(ValueError):read_records(raw('mmq',dict(r,**{key:value})))

    def test_negative_counters_are_rejected(self):
        r = stage(); r["experts_streamed"] = -1
        with self.assertRaises(ValueError):
            summarize(read_records(raw("profile", r)))


class PlanTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        base = dict(backend="hip", experimental_gfx906=True, exe="deployed-engine", gpu=[0,1],
                    host="0.0.0.0", port=8082, api_key="test-secret-not-printed",
                    args=["--pack","/existing/pack","--prefill=2048","--prefill","1024",
                          "--prompt-cache","6","--layer-split","24","--int8-kv","--mtp","/existing/mtp"],
                    env={"HSA_ENABLE_SDMA":"0", "STRATA_GFX906_PREFILL_ATTN":"1"})
        config = root / "original.json"; config.write_text(json.dumps(base))
        engine = root / "candidate-engine"; engine.write_bytes(b"synthetic-not-executable")
        return config, engine, base

    def test_replace_arg_handles_duplicates_and_equals(self):
        self.assertEqual(replace_arg(["--prefill=8192","--pack","p","--prefill","2048"], "--prefill", 3072),
                         ["--pack","p","--prefill","3072"])
        for args in (["--prefill"], ["--prefill","--pack","p"]):
            with self.assertRaises(ValueError): replace_arg(args, "--prefill", 2048)

    def test_plan_is_private_cloned_and_never_alters_source(self):
        with tempfile.TemporaryDirectory() as directory:
            config, engine, base = self.fixture(directory); original = config.read_bytes()
            output = Path(directory) / "plan"
            manifest = make_plan(config, engine, output, [2048,3072,4096], "test-source")
            self.assertEqual(config.read_bytes(), original)
            self.assertEqual(len(manifest["arms"]), 12)
            self.assertNotIn(base["api_key"], json.dumps(manifest))
            control = json.loads((output / "2048-control.json").read_text())
            optimized = json.loads((output / "4096-both.json").read_text())
            self.assertEqual(control["gpu"], base["gpu"])
            self.assertEqual((control["host"], control["port"]), ("127.0.0.1", 8096))
            self.assertEqual(control["env"]["HSA_ENABLE_SDMA"], "0")
            self.assertEqual(control["env"]["STRATA_GFX906_PREFILL_ATTN"], "0")
            self.assertEqual(optimized["env"]["STRATA_GFX906_MTP_BATCH"], "1")
            self.assertEqual(control["args"].count("--prefill"), 1)
            self.assertEqual(control["args"][control["args"].index("--prompt-cache")+1], "0")
            self.assertEqual((output.stat().st_mode & 0o777), 0o700)
            self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in output.iterdir()))

    def test_no_overwrite_or_unsafe_big_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            config, engine, _ = self.fixture(directory); output = Path(directory) / "plan"
            make_plan(config, engine, output, [2048], "test")
            old = (output / "plan.json").read_bytes()
            with self.assertRaises(ValueError): make_plan(config, engine, output, [2048], "test")
            self.assertEqual((output / "plan.json").read_bytes(), old)
            for chunks in ([8192], [], [2048,2048]):
                with self.assertRaises(ValueError): make_plan(config, engine, Path(directory)/"invalid", chunks, "test")
                self.assertFalse((Path(directory)/"invalid").exists())

    def test_nonopted_in_config_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config, engine, base = self.fixture(directory)
            for mutate in (lambda b: b.update(experimental_gfx906=False), lambda b: b.update(backend="cuda"),
                           lambda b: b.update(args="not-list"), lambda b: b.update(env=[])):
                b = copy.deepcopy(base); mutate(b); config.write_text(json.dumps(b))
                with self.assertRaises(ValueError): make_plan(config, engine, Path(directory)/"invalid", [2048], "test")
                self.assertFalse((Path(directory)/"invalid").exists())


class BenchClientTests(unittest.TestCase):
    def run_client(self, directory, reused=0, duplicate=False):
        root = Path(directory); log = root / "engine.log"; log.touch()
        output = root / "results.json"; calls = []
        reply = b'{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":"mock CPU reply"}}],"usage":{"prompt_tokens":1000,"completion_tokens":8}}'
        def request(req, timeout):
            calls.append(req.data)
            current_reuse = reused if len(calls) > 1 else 0
            line = f'prompt 1000 tokens = {current_reuse} reused + {1000-current_reuse} read in 100 ms (10000.0 tok/s), 8 generated in 200 ms (40.0 tok/s)\n'
            with log.open("a") as target: target.write(line * (2 if duplicate else 1))
            return io.BytesIO(reply)
        argv = ["bench_prefill.py", "--model", "mock", "--engine-log", str(log), "--output", str(output),
                "--label", "cpu-mock", "--rows", "1,2", "--fresh-only", "--max-tokens", "8"]
        with patch.object(sys, "argv", argv), patch.object(bench_prefill.urllib.request, "urlopen", request), contextlib.redirect_stdout(io.StringIO()):
            bench_prefill.main()
        return output, calls, reply

    def test_matched_cold_rows_and_verbatim_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output, calls, reply = self.run_client(directory)
            result = json.loads(output.read_text())
            self.assertEqual([r["kind"] for r in result], ["warmup", "fresh", "fresh"])
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads(calls[1])["max_tokens"], 8)
            evidence = output.with_suffix(".json.raw")
            self.assertEqual((evidence / "001-fresh.request.json").read_bytes(), calls[1])
            self.assertEqual((evidence / "001-fresh.response.json").read_bytes(), reply)
            self.assertIn(b"0 reused", (evidence / "001-fresh.engine.log").read_bytes())

    def test_reused_cold_prompt_is_rejected_but_raw_evidence_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "reused cached"):
                self.run_client(directory, reused=500)
            self.assertTrue((Path(directory) / "results.json.raw/001-fresh.response.json").is_file())

    def test_parallel_completion_lines_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                self.run_client(directory, duplicate=True)
            self.assertTrue((Path(directory) / "results.json.raw/000-warmup.engine.log").is_file())

    def test_existing_output_is_not_overwritten_or_contacted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"; output.write_text("preserve")
            argv = ["bench_prefill.py", "--model", "mock", "--engine-log", "irrelevant", "--output", str(output), "--label", "cpu"]
            with patch.object(sys, "argv", argv), patch.object(bench_prefill.urllib.request, "urlopen") as request, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench_prefill.main()
            request.assert_not_called()
            self.assertEqual(output.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
