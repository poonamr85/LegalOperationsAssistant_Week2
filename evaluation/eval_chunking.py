"""Retrieval-only checks for complete rule/condition evidence; no LLM calls."""
import argparse
import json
from pathlib import Path
from legal_assistant.retriever import HybridRetriever
from legal_assistant.knowledge_base import ROOT

CASES = [
    {'question': 'Can I paste an unsigned contract into an unapproved external AI tool?',
     'source': 'POL-PDF-001', 'required': ['do not paste', 'unsigned agreements', 'covered by this restriction']},
    {'question': 'What information do I submit to request review of an AI tool, and may I upload restricted information while review is pending?',
     'source': 'POL-PDF-001', 'required': ['tool/provider', 'training-use settings', 'do not upload restricted information while review is pending']},
    {'question': 'Does approval for a different AI tool or purpose approve my proposed use? Is a turnaround guaranteed?',
     'source': 'POL-PDF-001', 'required': ['does not approve your proposed', 'cannot verify', 'no approval date or turnaround is guaranteed']},
    {'question': 'Is an NDA alone enough to authorize employee-data sharing with a vendor?',
     'source': 'POL-CONF-011', 'required': ['nda alone does not authorize', 'privacy review and approvals']},
    {'question': 'Can vendor-held employee data be deleted when an active legal hold covers the records?',
     'source': 'POL-DATA-010', 'required': ['active legal hold', 'preserve them', 'before deletion']},
    {'question': 'Is vendor intake submission approval, or does it guarantee completion by my requested start date?',
     'source': 'POL-VEND-010', 'required': ['does not constitute approval', 'guarantee a completion date']},
]


def evaluate():
    retriever = HybridRetriever()
    rows = []
    for case in CASES:
        passages = retriever.search(case['question'], 'ExampleCo (fictional)', '2026.1')
        complete = []
        for passage in passages:
            text = ' '.join(passage['text'].lower().split())
            if passage['metadata']['source_id'] == case['source'] and all(term in text for term in case['required']):
                complete.append(passage['id'])
        rows.append({**case, 'passed': bool(complete), 'complete_passage_ids': complete,
                     'retrieved_ids': [p['id'] for p in passages]})
    return {'passed': sum(row['passed'] for row in rows), 'total': len(rows), 'cases': rows,
            'note': 'Handwritten development checks for required phrases in one retrieved passage, not answer accuracy.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-baseline', action='store_true', help='Save before rebuilding the index.')
    args = parser.parse_args()
    result = evaluate()
    baseline_path = ROOT / 'storage/chunking_baseline.json'
    if args.save_baseline:
        baseline_path.write_text(json.dumps(result, indent=2))
    else:
        report = {'after': result}
        if baseline_path.exists():
            report['before'] = json.loads(baseline_path.read_text())
        (ROOT / 'storage/chunking_comparison.json').write_text(json.dumps(report, indent=2))
    for row in result['cases']:
        print(('PASS' if row['passed'] else 'FAIL') + ' | ' + row['question'])
    print(f"{result['passed']}/{result['total']} complete-evidence retrieval checks. Not an accuracy measurement.")
    raise SystemExit(0 if result['passed'] == result['total'] else 1)
