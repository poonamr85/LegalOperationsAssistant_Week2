"""Development smoke checks; --llm additionally tests live grading and answers (billed)."""
import argparse
from evaluation.eval_set import CASES
from legal_assistant.workflow import run_workflow
from legal_assistant.retriever import HybridRetriever

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm',action='store_true',help='Call the configured provider; incurs API usage.')
    args=parser.parse_args()
    retriever=HybridRetriever()
    passed=0
    total=0
    for case in CASES:
        if not args.llm and case['expected_source'] is None:
            print('SKIP unsupported-question routing (requires --llm)')
            continue
        total+=1
        passages=retriever.search(case['question'],'ExampleCo (fictional)','2026.1')
        found=case['expected_source'] is None or any(p['metadata']['source_id']==case['expected_source'] for p in passages)
        ok=found
        detail='retrieval only'
        if args.llm:
            try:
                result=run_workflow(case['question'], 'ExampleCo (fictional)', '2026.1')
                ok=found and result['action']==case['expected_action']
                detail=f'action={result['action']}; expected={case["expected_action"]}'
            except Exception as exc:
                ok=False
                detail=f'pipeline error: {type(exc).__name__}'
        passed+=ok
        print(f'{"PASS" if ok else "FAIL"} | {case["question"]} | {detail}')
    print(f'{passed}/{total} smoke checks passed. Not a faithfulness measurement.')
    raise SystemExit(0 if passed==total else 1)
