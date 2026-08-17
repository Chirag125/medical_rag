import json
data = json.load(open('eval_results.json'))
for item in data['per_query']:
    status = 'HIT ' if item['hit'] else 'MISS'
    print(f"{status}  {item['qid']}: {item['question'][:70]}")