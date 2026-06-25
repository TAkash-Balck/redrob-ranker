import csv

rows = list(csv.DictReader(open('test_submission.csv')))
print(f'Total rows: {len(rows)}')
print(f'Headers: {list(rows[0].keys())}')

scores = [float(r['score']) for r in rows]
ranks = [int(r['rank']) for r in rows]

print(f'Rank range: {min(ranks)} to {max(ranks)}')
print(f'Score range: {min(scores):.6f} to {max(scores):.6f}')

mono = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
print(f'Monotonically non-increasing: {mono}')

unique_ranks = len(set(ranks)) == len(rows)
print(f'Unique ranks: {unique_ranks}')

all_in_range = all(0.0 <= s <= 1.0 for s in scores)
print(f'All scores in [0,1]: {all_in_range}')

print()
print('First 3 rows:')
for r in rows[:3]:
    print(f'  Rank {r["rank"]} | {r["candidate_id"]} | score={float(r["score"]):.4f}')
    print(f'  Reasoning: {r["reasoning"][:100]}')
    print()
