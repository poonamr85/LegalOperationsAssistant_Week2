"""Billed development benchmark for live claim validation; no employee UI changes."""
import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from legal_assistant import bot
from legal_assistant.knowledge_base import ROOT, LLM_MODEL, EVIDENCE_MODEL
from legal_assistant.workflow import run_workflow

QUESTIONS = [
    'Can I share employee information with a vendor?',
    'What NDA and Privacy requirements apply before I disclose confidential material and employee information to a vendor?',
    'Does approval for a different AI tool approve my proposed use, and is a turnaround guaranteed?',
]


def fixtures():
    def case(name, text, evidence, expected, action='answer'):
        return {'name':name, 'draft':bot.Draft(action=action, answer=text,
            citation_ids=['FIXTURE:0'] if action=='answer' else []),
            'evidence':[{'id':'FIXTURE:0','text':evidence,'metadata':{'source_type':'policy',
                         'organization':'ExampleCo (fictional)','synthetic':True}}], 'expected':expected}
    privacy = 'Before sharing employee information with a vendor, submit a Privacy review request. Do not share until the required reviews and approvals are complete.'
    nda = 'Use the approved NDA template. Legal reviews deviations; an authorized signatory executes the NDA.'
    return [
        case('Faithful paraphrase', 'Request Privacy review before sharing employee information, and wait for required approvals [FIXTURE:0].',privacy,True),
        case('Invented deadline', 'Request Privacy review; approval takes two working days [FIXTURE:0].',privacy,False),
        case('Broadened approval requirement', 'Every NDA requires separate Legal review and approval [FIXTURE:0].',nda,False),
        case('Contradicted legal hold', 'Delete the records when the project closes [FIXTURE:0].', 'Preserve records under an active legal hold. Only Legal can release the hold; project closure does not release it.',False),
        case('Unauthorized approval', 'I approve sharing the employee spreadsheet now [FIXTURE:0].',privacy,False),
        case('Pure clarification', 'Which vendor and data categories are you asking about?',privacy,True,'clarify'),
        case('Unfaithful clarification', 'Approval takes two days. Which vendor are you using?',privacy,False,'clarify'),
    ]


def benchmark(repeats):
    manifest = (ROOT/'storage/index.json').read_bytes()
    report = {'created_at':datetime.now(timezone.utc).isoformat(), 'review_status':'Automated development run; not Legal-reviewed',
        'index_sha256':hashlib.sha256(manifest).hexdigest(), 'models':{'answer':LLM_MODEL,'grade':EVIDENCE_MODEL,
        'claim':__import__('os').getenv('CLAIM_MODEL') or EVIDENCE_MODEL}, 'runs':[], 'fixtures':[],
        'limitations':'Small non-held-out sample; model labels are not verified ground truth or a measured 95% faithfulness result.'}
    output = ROOT/'storage/evaluations/claim_validation_benchmark.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        output.write_text(json.dumps(report,indent=2))
    for iteration in range(repeats):
        for question in QUESTIONS:
            state = run_workflow(question,'ExampleCo (fictional)','2026.1')
            row = {key:state.get(key) for key in ['question','action','answer','timings','steps','decision_reason','error_message','failed_stage','citation_validation','faithfulness_percent']}
            row.update(iteration=iteration+1, draft=state['draft'].model_dump() if state.get('draft') else None,
                       selected_passages=state.get('selected_passages',[]),
                       claim_validation=state['claim_validation'].model_dump() if state.get('claim_validation') else None)
            report['runs'].append(row)
            print(json.dumps({key:row[key] for key in ['question','action','timings','failed_stage']}),flush=True)
            save()
    human = run_workflow('Please contact the Legal team','','')
    report['direct_human']={'action':human['action'],'steps':human['steps'],'timings':human['timings']}
    for fixture in fixtures():
        start=perf_counter()
        try:
            result = bot.validate_claim_support('Check this fictional procedure.',fixture['draft'],fixture['evidence'])
            accepted = bot.claim_validation_passes(result,fixture['draft'],fixture['evidence'])
            row = {'name':fixture['name'],'expected_accept':fixture['expected'],'accepted':accepted,
                   'matched_expected':accepted==fixture['expected'],'assessment':result.model_dump(),'error':None}
        except Exception as exc:
            row = {'name':fixture['name'],'expected_accept':fixture['expected'],'accepted':False,
                   'matched_expected':False,'assessment':None,'error':bot.pipeline_error_message(exc,'claim validation')}
        row['seconds']=round(perf_counter()-start,4)
        report['fixtures'].append(row)
        print(json.dumps({key:row[key] for key in ['name','accepted','matched_expected','seconds','error']}),flush=True)
        save()
    checked = [r for r in report['runs'] if r.get('claim_validation')]
    deltas=[r['timings']['claim validation'] for r in checked]
    totals=[r['timings']['total'] for r in checked]
    before=[r['timings']['total']-r['timings']['claim validation'] for r in checked]
    report['summary']={'generated_drafts_checked':len(checked),'validation_stage_errors':sum(r.get('failed_stage')=='claim validation' for r in report['runs']),
        'accepted_answers':sum(r['action']=='answer' for r in report['runs']),
        'median_added_validation_seconds':statistics.median(deltas) if deltas else None,
        'validation_seconds_range':[min(deltas),max(deltas)] if deltas else None,
        'median_total_seconds':statistics.median(totals) if totals else None,
        'median_same_run_seconds_excluding_validation':statistics.median(before) if before else None,
        'fixture_matches':sum(r['matched_expected'] for r in report['fixtures']), 'fixture_total':len(report['fixtures'])}
    save()
    print(json.dumps(report['summary'],indent=2),flush=True)
    print('Saved:',output,flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',help='Required: allow paid provider calls.')
    parser.add_argument('--repeats',type=int,default=1)
    args=parser.parse_args()
    if not args.live:
        parser.error('--live is required; this benchmark makes billed API calls.')
    if not 1<=args.repeats<=5:
        parser.error('--repeats must be between 1 and 5.')
    benchmark(args.repeats)
