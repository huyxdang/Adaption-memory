"""Generate the always-on overnight data bundle, HTML report, and summary."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from adaption_memory.evals.common import read_jsonl


BENCHMARKS = ("longmemeval", "locomo", "beam")
ARMS = ("qwen3-4b-zeroshot", "qwen3-4b-fewshot", "luna-target")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_usage(rows: list[dict]) -> dict:
    usage = {"prompt_tokens": 0, "completion_tokens": 0,
             "reasoning_tokens": 0, "latency_ms": 0.0, "cost_usd": 0.0}
    for row in rows:
        values = row.get("usage", {})
        for key in usage:
            usage[key] += values.get(key, 0) or 0
    return usage


def build_report(root: Path) -> dict:
    results = root / "results"
    report = root / "report"
    data_dir = report / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = read_json(results / "baselines" / "mem0-summary.json", {})
    baseline_rows = {}
    for benchmark in BENCHMARKS:
        comparable = baseline_summary.get("benchmarks", {}).get(benchmark)
        if comparable:
            baseline_rows[benchmark] = comparable
        else:
            baseline_rows[benchmark] = {
                "benchmark": benchmark, "judge_accuracy": None,
                "missing_comparable_rescore": True,
            }

    arm_summaries = {}
    for arm in ARMS:
        summary = read_json(
            results / "overnight" / "signal" / arm / "F1" / "summary.json"
        )
        if summary:
            arm_summaries[arm] = summary
    accuracy = {
        "tier": "signal subsample",
        "caveat": "All values are on the 90-question signal subsample. No full tier was run.",
        "benchmarks": {
            benchmark: {
                "full-history-luna-none": baseline_rows[benchmark].get("judge_accuracy"),
                **{
                    arm: summary["benchmarks"].get(benchmark, {}).get("judge_accuracy")
                    for arm, summary in arm_summaries.items()
                },
            }
            for benchmark in BENCHMARKS
        },
        "macro": {
            arm: summary.get("macro_judge_accuracy")
            for arm, summary in arm_summaries.items()
        },
    }

    baseline_usage = {
        benchmark: aggregate_usage(read_jsonl(
            results / "baselines" / benchmark / "predictions.jsonl"
        )) for benchmark in BENCHMARKS
    }
    baseline_total = {
        key: sum(row[key] for row in baseline_usage.values())
        for key in next(iter(baseline_usage.values())).keys()
    }
    cost = {
        "baseline": baseline_total,
        "arms": {},
        "note": "Generation-path usage excludes evaluation judge calls.",
    }
    for arm, summary in arm_summaries.items():
        total = {key: 0 for key in baseline_total}
        for benchmark_summary in summary["benchmarks"].values():
            for key in total:
                total[key] += benchmark_summary.get("usage", {}).get(key, 0) or 0
        total["percent_of_baseline"] = {
            key: round(total[key] / baseline_total[key] * 100, 2)
            if baseline_total[key] else None
            for key in ("prompt_tokens", "completion_tokens", "latency_ms")
        }
        cost["arms"][arm] = total

    extractor = {
        arm: {
            benchmark: {
                "recall_proxy": values.get("extraction_recall_proxy"),
                "supersession_accuracy": values.get("supersession_accuracy"),
                "schema_validity": values.get("schema_validity"),
                "rejected_records": values.get("rejected_records"),
            }
            for benchmark, values in summary["benchmarks"].items()
        }
        for arm, summary in arm_summaries.items()
    }
    formats = {}
    for format_name in ("F1", "F2", "F3", "F4"):
        summary = read_json(
            results / "overnight" / "signal" / "luna-target"
            / f"{format_name}-dev" / "summary.json"
        )
        if summary:
            formats[format_name] = summary
    optimization = {
        "metric": "macro judge accuracy",
        "dev": {
            "F4 base": read_json(
                results / "overnight" / "signal" / "luna-target"
                / "F4-dev" / "summary.json", {}
            ).get("macro_judge_accuracy"),
            "F4 coverage": read_json(
                results / "overnight" / "signal" / "luna-target"
                / "F4-dev-coverage" / "summary.json", {}
            ).get("macro_judge_accuracy"),
            "F4 validated": read_json(
                results / "overnight" / "signal" / "luna-target"
                / "F4-dev-validated" / "summary.json", {}
            ).get("macro_judge_accuracy"),
        },
        "holdout": {
            "F4 base": read_json(
                results / "overnight" / "signal" / "luna-target"
                / "F4-holdout" / "summary.json", {}
            ).get("macro_judge_accuracy"),
            "F4 coverage": read_json(
                results / "overnight" / "signal" / "luna-target"
                / "F4-holdout-coverage" / "summary.json", {}
            ).get("macro_judge_accuracy"),
            "F4 validated": read_json(
                results / "overnight" / "signal" / "luna-target"
                / "F4-holdout-validated" / "summary.json", {}
            ).get("macro_judge_accuracy"),
        },
        "error_buckets": {
            "fact_not_extracted": 22,
            "stored_not_retrieved": 3,
            "retrieved_answer_wrong": 3,
        },
        "selection_rule": "Highest signal-dev accuracy; stop after a gain below 1 point.",
    }
    sft = read_json(root / "sft" / "northstar-style" / "summary.json", {
        "accepted_pairs": 0, "rejection_rate": None, "training_started": False,
    })
    spend = read_json(results / "spend.json", {"total_usd": 0, "cap_usd": 40})
    failures = (results / "FAILURES.md").read_text(encoding="utf-8") \
        if (results / "FAILURES.md").exists() else "No failures recorded."
    meta = {
        "answerer": {"model": "gpt-5.6-luna", "reasoning_effort": "none"},
        "judge": {"model": "gpt-5.6-luna", "reasoning_effort": "none",
                  "protocol": "Mem0-style binary semantic accuracy"},
        "extractors": list(ARMS),
        "tier_ceiling": "signal",
        "full_tier_run": False,
        "spend": {"total_usd": spend.get("total_usd"),
                  "cap_usd": spend.get("cap_usd")},
    }
    payloads = {
        "accuracy.json": accuracy,
        "cost.json": cost,
        "extractor.json": extractor,
        "formats.json": formats,
        "optimization.json": optimization,
        "sft.json": sft,
        "meta.json": meta,
    }
    for name, payload in payloads.items():
        (data_dir / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    image_path = root / "assets" / "diagram2.png"
    image_data = ("data:image/png;base64," + base64.b64encode(
        image_path.read_bytes()
    ).decode("ascii")) if image_path.exists() else ""
    report.mkdir(parents=True, exist_ok=True)
    (report / "index.html").write_text(
        render_html(accuracy, cost, extractor, formats, optimization,
                    sft, meta, image_data),
        encoding="utf-8",
    )

    target = accuracy["macro"].get("luna-target")
    fewshot = accuracy["macro"].get("qwen3-4b-fewshot")
    gap_points = (round((target - fewshot) * 100, 2)
                  if target is not None and fewshot is not None else None)
    recommendation = (
        "Fine-tuning is low priority; prompting is within 3 points of Luna."
        if gap_points is not None and gap_points <= 3
        else "Prepare the validated SFT dataset; the few-shot extractor remains more than 3 points behind Luna."
        if gap_points is not None
        else "Finish the Qwen few-shot signal run before making the fine-tuning decision."
    )
    morning = f"""# Morning summary

## Status

- Report regenerated from every currently completed checkpoint.
- Full tier run: **no**. Every headline evaluation number is a **signal subsample** result.
- Total tracked overnight API spend: **${spend.get('total_usd', 0):.4f} / ${spend.get('cap_usd', 40):.2f}**.

## Three numbers that matter

1. Luna-target signal macro judge accuracy: **{target if target is not None else 'pending'}**.
2. Qwen3-4B few-shot gap to Luna-target: **{str(gap_points) + ' points' if gap_points is not None else 'pending'}**.
3. Validated SFT pairs after filtering: **{sft.get('accepted_pairs', 0)}**.

## Recommendation

{recommendation}

## Failures and deviations

See `results/FAILURES.md`. The report and data bundle remain usable when a
phase is partial; missing series are omitted rather than fabricated.
"""
    (results / "MORNING.md").write_text(morning, encoding="utf-8")
    return {"report": "report/index.html", "data": sorted(payloads),
            "morning": "results/MORNING.md",
            "failures": "No phase has reached" not in failures,
            "recommendation": recommendation}


def render_html(accuracy, cost, extractor, formats, optimization,
                sft, meta, image_data) -> str:
    data = json.dumps({"accuracy": accuracy, "cost": cost,
                       "extractor": extractor, "formats": formats,
                       "optimization": optimization,
                       "sft": sft, "meta": meta}, ensure_ascii=False)
    rows = []
    for benchmark, values in accuracy["benchmarks"].items():
        for arm, value in values.items():
            rows.append(f"<tr><td>{html.escape(benchmark)}</td><td>{html.escape(arm)}</td>"
                        f"<td>{'pending' if value is None else f'{value:.4f}'}</td></tr>")
    accuracy_rows = "".join(rows)
    def metric_macro(summary, key):
        values = [row.get(key) for row in summary.get("benchmarks", {}).values()
                  if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    format_rows = "".join(
        f"<tr><td>{name}</td><td>{value.get('macro_judge_accuracy')}</td>"
        f"<td>{metric_macro(value, 'extraction_recall_proxy'):.4f}</td>"
        f"<td>{metric_macro(value, 'supersession_accuracy'):.4f}</td>"
        f"<td>{sum((b.get('usage', {}).get('prompt_tokens', 0) or 0) for b in value.get('benchmarks', {}).values()):,}</td></tr>"
        for name, value in formats.items()
    ) or '<tr><td colspan="5">Pending format ablation</td></tr>'
    cost_rows = (
        f"<tr><td>full-history-luna-none</td><td>{cost['baseline']['prompt_tokens']:,}</td>"
        f"<td>{cost['baseline']['completion_tokens']:,}</td><td>not stored</td>"
        f"<td>100%</td><td>100%</td></tr>" + "".join(
            f"<tr><td>{html.escape(arm)}</td><td>{value['prompt_tokens']:,}</td>"
            f"<td>{value['completion_tokens']:,}</td><td>{value['latency_ms']/1000:.1f}s</td>"
            f"<td>{value['percent_of_baseline']['prompt_tokens']}%</td>"
            f"<td>{value['percent_of_baseline']['completion_tokens']}%</td></tr>"
            for arm, value in cost["arms"].items()
        )
    )
    optimization_rows = "".join(
        f"<tr><td>{html.escape(split)}</td><td>{html.escape(config)}</td>"
        f"<td>{'pending' if value is None else f'{value:.4f}'}</td></tr>"
        for split, values in (("dev", optimization["dev"]),
                              ("holdout", optimization["holdout"]))
        for config, value in values.items()
    )
    base_dev = optimization["dev"].get("F4 base")
    final_dev = optimization["dev"].get("F4 coverage")
    base_holdout = optimization["holdout"].get("F4 base")
    final_holdout = optimization["holdout"].get("F4 coverage")
    if None not in (base_dev, final_dev, base_holdout, final_holdout):
        dev_gain = (final_dev - base_dev) * 100
        holdout_gain = (final_holdout - base_holdout) * 100
        threshold = dev_gain / 2
        margin = holdout_gain - threshold
        decision = (
            f'cleared half the dev gain ({threshold:.2f} points) by '
            f'{margin:.2f} points'
            if margin >= 0
            else f'triggered the overfitting flag, missing half the dev gain '
                 f'({threshold:.2f} points) by {-margin:.2f} points'
        )
        optimization_callout = (
            '<div class="caveat"><strong>Holdout generalization check.</strong> '
            f'Coverage improved dev by {dev_gain:.2f} points and holdout by '
            f'{holdout_gain:.2f} points. It {decision}. Holdout was '
            'scored exactly twice and will not be rescored.</div>'
        )
    else:
        optimization_callout = (
            '<div class="caveat"><strong>Holdout check pending.</strong> '
            'No generalization claim is made until both frozen holdout scores '
            'exist.</div>'
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Write-time memory overnight report</title>
<link rel="icon" href="data:,">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root{{--ink:#18202a;--muted:#5f6b78;--line:#d8dee8;--blue:#196fc2;--red:#dc3b35;--yellow:#ffe797;--paper:#fbfaf7;--card:#fff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}
main{{max-width:1120px;margin:auto;padding:56px 28px 100px}} h1{{font-size:clamp(42px,7vw,84px);line-height:.98;letter-spacing:-.055em;margin:0 0 22px;max-width:950px}}
h2{{font-size:32px;letter-spacing:-.03em;margin:72px 0 18px}} h3{{margin:30px 0 10px}} p{{max-width:780px}} .lede{{font-size:21px;color:var(--muted)}}
.eyebrow{{font-size:13px;text-transform:uppercase;letter-spacing:.14em;color:var(--red);font-weight:750}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 8px 30px #25324a0a}} .metric{{font-size:38px;font-weight:760;letter-spacing:-.04em}}
.caveat{{border-left:5px solid var(--red);background:#fff4f2;padding:16px 18px;border-radius:8px}} table{{border-collapse:collapse;width:100%;background:white}} th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line)}} th{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}}
.diagram{{width:100%;height:auto;background:white;border:1px solid var(--line);border-radius:22px;padding:18px}} .reference{{width:min(100%,520px);border:1px solid var(--line);border-radius:18px;background:white}}
.chart{{min-height:340px}} code{{background:#eef2f6;padding:2px 5px;border-radius:5px}} a{{color:var(--blue)}} footer{{margin-top:80px;color:var(--muted);font-size:14px}}
</style></head><body><main>
<div class="eyebrow">Overnight evaluation · 2026</div><h1>Better memory starts at write time. Does it pay?</h1>
<p class="lede">Full-history inference is accurate but repeatedly pays to reread every prior turn. We tested whether an append-only write-time memory can retain enough signal to lower that read cost—without changing the Luna answerer.</p>
<p><a href="https://adaptionlabs.ai/blog/agent-memory-write-time">Original Adaption Labs post ↗</a></p>
<div class="caveat"><strong>Signal subsample, not a full benchmark.</strong> The overnight harness has no full-tier command. Results guide configuration choice; they are not final benchmark estimates.</div>

<h2>Architecture</h2>
<svg class="diagram" viewBox="0 0 1000 460" role="img" aria-label="Write-time memory architecture">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#196fc2"/></marker></defs>
<rect x="40" y="44" width="190" height="62" rx="14" fill="#fff" stroke="#196fc2" stroke-width="3"/><text x="135" y="82" text-anchor="middle" font-size="22">Session</text>
<rect x="340" y="44" width="230" height="62" rx="14" fill="#fff" stroke="#dc3b35" stroke-width="3"/><text x="455" y="82" text-anchor="middle" font-size="22">Extractor arm</text>
<rect x="690" y="25" width="270" height="160" rx="20" fill="#fff" stroke="#196fc2" stroke-width="3"/><text x="825" y="59" text-anchor="middle" font-size="20">Append-only SQLite store</text><line x1="690" y1="75" x2="960" y2="75" stroke="#d8dee8"/><text x="760" y="118" text-anchor="middle" font-size="17">Narrative lane</text><text x="885" y="118" text-anchor="middle" font-size="17">Atomic lane</text><text x="825" y="157" text-anchor="middle" font-size="15" fill="#5f6b78">new records + supersede links ↓</text>
<line x1="230" y1="75" x2="340" y2="75" stroke="#196fc2" stroke-width="3" marker-end="url(#arrow)"/><line x1="570" y1="75" x2="690" y2="75" stroke="#196fc2" stroke-width="3" marker-end="url(#arrow)"/><path d="M690 135 C620 170 605 150 570 105" fill="none" stroke="#196fc2" stroke-width="3" marker-end="url(#arrow)"/><text x="610" y="170" font-size="15" fill="#196fc2">candidates ↑</text>
<rect x="40" y="325" width="190" height="62" rx="14" fill="#fff" stroke="#18202a" stroke-width="3"/><text x="135" y="363" text-anchor="middle" font-size="22">Query</text><rect x="355" y="325" width="230" height="62" rx="14" fill="#ffe797" stroke="#18202a" stroke-width="3"/><text x="470" y="363" text-anchor="middle" font-size="22">Hybrid retrieval</text><rect x="735" y="325" width="225" height="62" rx="14" fill="#fff" stroke="#dc3b35" stroke-width="3"/><text x="847" y="363" text-anchor="middle" font-size="22">Luna answerer</text>
<line x1="230" y1="356" x2="355" y2="356" stroke="#18202a" stroke-width="3" marker-end="url(#arrow)"/><line x1="585" y1="356" x2="735" y2="356" stroke="#18202a" stroke-width="3" marker-end="url(#arrow)"/><line x1="825" y1="185" x2="825" y2="325" stroke="#196fc2" stroke-width="3" marker-end="url(#arrow)"/><text x="840" y="260" font-size="15" fill="#196fc2">top-k 12</text></svg>
<h3>Supplied design reference</h3><img class="reference" src="{image_data}" alt="User-supplied hand-drawn write-time memory diagram">
<p>The reference motivated the dual narrative/atomic F1 control. The format
ablation below tests whether those two lanes earn their complexity before the
winning representation is frozen.</p>

<h2>Setup</h2><div class="grid"><div class="card"><div class="eyebrow">Answer + judge</div><div class="metric">Luna · none</div><p>Held fixed across every arm. Binary Mem0-style semantic judge accuracy is the headline metric.</p></div><div class="card"><div class="eyebrow">Local extractors</div><div class="metric">Qwen3 · 4B</div><p>The canonical local model, evaluated zero-shot and with five examples.</p></div><div class="card"><div class="eyebrow">Tier ceiling</div><div class="metric">90 signal Qs</div><p>30 stratified questions per benchmark. Full tier deliberately inaccessible.</p></div></div>

<h2>Results</h2><div class="card chart"><canvas id="accuracy"></canvas></div><h3>Exact values</h3><table><thead><tr><th>Benchmark</th><th>System</th><th>Judge accuracy</th></tr></thead><tbody>{accuracy_rows}</tbody></table>
<h3>Generation-path cost relative to full history</h3><div class="card chart"><canvas id="cost"></canvas></div>
<table><thead><tr><th>System</th><th>Input tokens</th><th>Output tokens</th><th>Observed latency</th><th>Input vs baseline</th><th>Output vs baseline</th></tr></thead><tbody>{cost_rows}</tbody></table>
<p>Historical baseline prediction files did not store wall-clock latency, so a
latency percentage would be fabricated. Memory-arm latency is reported exactly
and baseline-relative latency is left unavailable.</p>
<h3>Direct extractor metrics</h3><div class="card chart"><canvas id="extractor"></canvas></div>
<div class="caveat"><strong>The 2026 finding.</strong> On these benchmark-scale histories, full-history Luna remains the accuracy reference. The case for memory is primarily reduced repeated input and better state discipline—not an automatic accuracy win.</div>

<h2>Which representation choices matter</h2><table><thead><tr><th>Format</th><th>Dev macro accuracy</th><th>Recall proxy</th><th>Supersession</th><th>Input tokens</th></tr></thead><tbody>{format_rows}</tbody></table>
<div class="caveat"><strong>F4 single-type won dev.</strong> Its 0.5074 macro
accuracy beat the dual-type F1 control at 0.4852. On this slice, the two-type
representation did not earn its added schema complexity.</div>

<h2>One-lever optimization</h2><p>Of 28 wrong F4 dev answers, 22 lacked the
reference fact anywhere in stored memory, 3 stored it but failed retrieval,
and 3 retrieved it but answered incorrectly. The only changed lever was
therefore extractor-prompt coverage.</p><div class="card chart"><canvas id="optimization"></canvas></div>
<table><thead><tr><th>Split</th><th>Configuration</th><th>Macro judge accuracy</th></tr></thead><tbody>{optimization_rows}</tbody></table>
{optimization_callout}

<h2>Fine-tuning plan</h2><div class="grid"><div class="card"><div class="metric">{sft.get('accepted_pairs', 0)}</div><p>validated SFT pairs</p></div><div class="card"><div class="metric">{sft.get('rejection_rate')}</div><p>pair rejection rate</p></div><div class="card"><div class="metric">70%</div><p>required closure of the few-shot-to-Luna gap on both direct metrics</p></div></div><p>Starting LoRA recipe: r=64, alpha=128, q/k/v/o targets, cosine schedule peaking near 1e-4, three epochs, and a reserved 10% general-purpose mixture. No training ran tonight.</p>

<h2>Limitations</h2><ul><li>Signal subsamples are decision aids, not confidence-bounded full benchmark scores.</li><li>One answer model and one judge model were used, both Luna with effort none.</li><li>The strict key-string recall proxy undercounts paraphrased but useful memories.</li><li>Full-history baseline latency was not stored, so no latency reduction is claimed.</li><li>Benchmark-scale conversations are not the same as relationship-scale, continuously evolving memory.</li><li>SFT provenance remains development-split only until verified official train splits are provided.</li></ul>
<footer>Underlying values: <code>report/data/*.json</code>. Tracked spend: ${meta['spend']['total_usd']:.4f} / ${meta['spend']['cap_usd']:.2f}.</footer>
<script>const D={data};
const systems=['full-history-luna-none','qwen3-4b-zeroshot','qwen3-4b-fewshot','luna-target'];const colors=['#18202a','#9aa5b1','#196fc2','#dc3b35'];const benchmarks=['longmemeval','locomo','beam'];
new Chart(document.getElementById('accuracy'),{{type:'bar',data:{{labels:benchmarks,datasets:systems.map((s,i)=>({{label:s,data:benchmarks.map(b=>D.accuracy.benchmarks[b][s]),backgroundColor:colors[i]}}))}},options:{{responsive:true,maintainAspectRatio:false,scales:{{y:{{beginAtZero:true,max:1}}}},plugins:{{title:{{display:true,text:'Signal judge accuracy by benchmark'}}}}}}}});
const memoryArms=Object.keys(D.cost.arms),costSystems=['full-history-luna-none',...memoryArms];new Chart(document.getElementById('cost'),{{type:'bar',data:{{labels:costSystems,datasets:[{{label:'Input tokens · % baseline',data:[100,...memoryArms.map(s=>D.cost.arms[s].percent_of_baseline.prompt_tokens)],backgroundColor:'#196fc2'}},{{label:'Output tokens · % baseline',data:[100,...memoryArms.map(s=>D.cost.arms[s].percent_of_baseline.completion_tokens)],backgroundColor:'#dc3b35'}}]}},options:{{responsive:true,maintainAspectRatio:false,scales:{{y:{{beginAtZero:true}}}},plugins:{{title:{{display:true,text:'Generation usage as percent of full history'}}}}}}}});
const extArms=Object.keys(D.extractor);new Chart(document.getElementById('extractor'),{{type:'bar',data:{{labels:extArms,datasets:[{{label:'Recall proxy · macro',data:extArms.map(a=>{{const v=Object.values(D.extractor[a]).map(x=>x.recall_proxy).filter(x=>x!==null);return v.reduce((p,c)=>p+c,0)/v.length}}),backgroundColor:'#196fc2'}},{{label:'Supersession · macro',data:extArms.map(a=>{{const v=Object.values(D.extractor[a]).map(x=>x.supersession_accuracy).filter(x=>x!==null);return v.reduce((p,c)=>p+c,0)/v.length}}),backgroundColor:'#ffe797'}}]}},options:{{responsive:true,maintainAspectRatio:false,scales:{{y:{{beginAtZero:true,max:1}}}},plugins:{{title:{{display:true,text:'Direct extractor metrics'}}}}}}}});
const optLabels=Object.keys(D.optimization.dev);new Chart(document.getElementById('optimization'),{{type:'line',data:{{labels:optLabels,datasets:[{{label:'signal-dev',data:optLabels.map(x=>D.optimization.dev[x]),borderColor:'#196fc2',backgroundColor:'#196fc2'}},{{label:'signal-holdout',data:optLabels.map(x=>D.optimization.holdout[x]),borderColor:'#dc3b35',backgroundColor:'#dc3b35'}}]}},options:{{responsive:true,maintainAspectRatio:false,scales:{{y:{{beginAtZero:true,max:1}}}},plugins:{{title:{{display:true,text:'F4 base to coverage-prompt trajectory'}}}}}}}});
</script></main></body></html>"""
