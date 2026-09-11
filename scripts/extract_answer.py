"""Extract the latest complete text event from a running or finished JSONL log."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('log', type=Path)
args = parser.parse_args()
answer = ''
for line in args.log.read_text().splitlines():
    try:
        event = json.loads(line)
    except ValueError:
        continue  # The writer may be partway through its latest line.
    if 'step' in event:
        answer = event['text']
    elif event.get('event') == 'complete':
        answer = event['output']
print(answer)
