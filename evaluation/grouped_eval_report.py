"""Build a grouped review pack from saved results only. No model/provider calls."""
import argparse
import csv
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LABELS={'expected_answer':'Expected answers','ambiguous':'Ambiguous questions','legal_redirect':'Legal redirects'}


def build_report(dataset_path=None,baseline_path=None,output=None):
    dataset_path=Path(dataset_path or ROOT/'evaluation/scenario_eval_set.json')
    baseline_path=Path(baseline_path or ROOT/'storage/evaluations/resolution_20_queries/resolution_report.json')
    output=Path(output or ROOT/'storage/evaluations/grouped_review')
    dataset=json.loads(dataset_path.read_text());baseline=json.loads(baseline_path.read_text())
    previous={row['id']:row for row in baseline['results']}
    results=[]
    for case in dataset['cases']:
        saved=previous.get(case['id'])
        if saved and saved['question']!=case['question']:
            raise ValueError('Case ID matches a different saved question')
        results.append({**case,'execution_status':'saved_baseline' if saved else 'not_run',
            'actual_action':saved['actual_action'] if saved else None,
            'actual_answer':saved['answer'] if saved else None,
            'answer_quality':saved['answer_quality'] if saved else None,
            'escalation_reason':saved['escalation_reason'] if saved else None,
            'response_latency_seconds':saved['response_latency_seconds'] if saved else None,
            'processing_failure':saved['processing_failure'] if saved else None,
            'resolved':saved['resolved'] if saved else None,
            'correct_escalation':saved['correct_escalation'] if saved else None})
    counts={}
    for key,label in LABELS.items():
        group=[row for row in results if row['category']==key]
        measured=[row for row in group if row['execution_status']=='saved_baseline']
        succeeded=sum(bool(row['resolved'] if key=='expected_answer' else row['correct_escalation']) for row in measured) if key!='ambiguous' else None
        counts[key]={'label':label,'dataset_count':len(group),'saved_results_count':len(measured),
            'not_run_count':len(group)-len(measured),'successful_expected_handling':succeeded,
            'handling_rate_percent':round(100*succeeded/len(measured),2) if measured and succeeded is not None else None}
    current_hash=hashlib.sha256((ROOT/'storage/index.json').read_bytes()).hexdigest()
    report={'generated_at':datetime.now(timezone.utc).isoformat(),
        'source':'Saved 20-query baseline; no new inference or quality assessment calls',
        'baseline_created_at':baseline['created_at'],'baseline_index_sha256':baseline['index_sha256'],
        'current_index_sha256':current_hash,'baseline_matches_current_index':current_hash==baseline['index_sha256'],
        'notice':'The saved statistics describe the earlier index, before new clarification policies. Five ambiguous questions are not run; their actual responses, latency and success rates are unknown.',
        'review_status':'Automated baseline judgments; illustrative expected examples are not actual model responses.',
        'categories':counts,'baseline_metrics':baseline['metrics'],'results':results}
    output.mkdir(parents=True,exist_ok=True)
    (output/'grouped_report.json').write_text(json.dumps(report,indent=2))
    fields=['id','category','question','expected_action','expected_outcome','expected_response_example',
            'execution_status','actual_action','actual_answer','answer_quality','escalation_reason','response_latency_seconds']
    with (output/'grouped_report.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for row in results:writer.writerow({key:row.get(key) for key in fields})
    e=lambda value:html.escape(str(value))
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><title>Grouped Legal assistant review</title>
    <style>body{font:16px system-ui;color:#172b45;max-width:1200px;margin:40px auto;padding:20px}h1,h2{color:#10233f}.note{padding:18px;background:#fff5df;border-left:5px solid #b77817}table{width:100%;border-collapse:collapse;margin:20px 0}th,td{border:1px solid #d8e1ef;padding:12px;text-align:left;vertical-align:top}th{background:#10233f;color:white}.example{background:#f1f5fb;padding:14px}details{margin:12px 0;padding:14px;border:1px solid #d8e1ef}small{color:#53657a}</style>
    <h1>Legal Operations Assistant — dataset and saved statistics</h1>''',
    f'<div class="note"><b>No new Nebius calls.</b> {e(report["notice"])}</div>',
    '<p>Expected wording is illustrative. Actual answers below are copied from the saved baseline, not regenerated. All data is fictional.</p>',
    '<h2>Dataset coverage and measured outcomes</h2><table><tr><th>Group</th><th>Dataset questions</th><th>Saved results</th><th>Not run</th><th>Expected handling</th></tr>']
    for item in counts.values():
        rate='Not measured' if item['handling_rate_percent'] is None else f'{item["successful_expected_handling"]}/{item["saved_results_count"]} ({item["handling_rate_percent"]}%)'
        parts.append(f'<tr><td>{e(item["label"])}</td><td>{item["dataset_count"]}</td><td>{item["saved_results_count"]}</td><td>{item["not_run_count"]}</td><td>{rate}</td></tr>')
    m=baseline['metrics']
    parts.extend(['</table><h2>Saved 20-query statistics</h2>',
        f'<p>Automated resolution: <b>{m["automated_verified_resolution_rate_percent"]}% ({m["resolved_count"]}/{m["evaluated_queries"]})</b>; correct escalation: <b>{m["correct_escalation_rate_percent"]}%</b>; unnecessary escalation: <b>{m["unnecessary_escalation_rate_percent"]}%</b>; processing failures: <b>{m["processing_failure_rate_percent"]}%</b>.</p>',
        f'<p>Median: <b>{m["median_response_seconds"]:.2f} s</b>; P95: <b>{m["p95_response_seconds"]:.2f} s</b>. These are automated development results, not user-confirmed resolution or proof of 95% faithfulness.</p>',
        f'<small>Baseline created: {e(baseline["created_at"])}. Same index as current knowledge base: {e(report["baseline_matches_current_index"])}.</small>'])
    for category,label in LABELS.items():
        parts.append(f'<h2>{e(label)}</h2>')
        for row in results:
            if row['category']!=category:continue
            parts.append(f'<details><summary><b>{e(row["id"])}</b> — {e(row["question"])}</summary><p><b>Expected handling:</b> {e(row["expected_outcome"])}</p><div class="example"><b>Illustrative expected response:</b> {e(row["expected_response_example"])}</div>')
            if row['execution_status']=='not_run':
                parts.append('<p><b>Actual response / outcome / latency: not run.</b> No model call was made for this case.</p>')
                parts.append(f'<p><b>Acceptable alternative:</b> {e(row.get("acceptable_alternative",""))}</p>')
            else:
                parts.append(f'<p><b>Saved actual response:</b> {e(row["actual_answer"])}</p><p><b>Actual action:</b> {e(row["actual_action"])}; <b>Quality:</b> {e(row["answer_quality"])}; <b>Latency:</b> {row["response_latency_seconds"]:.2f} s.</p><p><b>Escalation reason:</b> {e(row["escalation_reason"] or "None")}</p>')
            if row.get('must_not'):parts.append('<p><b>Must not:</b> '+e(' '.join(row['must_not']))+'</p>')
            parts.append('</details>')
    parts.append('</html>')
    (output/'grouped_report.html').write_text('\n'.join(parts))
    print(json.dumps(counts,indent=2));print('Saved:',output)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();build_report(output=args.output)
