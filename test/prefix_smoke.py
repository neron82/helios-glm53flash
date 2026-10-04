#!/usr/bin/env python3
"""Exercise prefix reuse and invalidation against a running vision-enabled server."""
import argparse
import base64
import io
import json
from pathlib import Path
import time
import urllib.request
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--output', default='/tmp/helios-prefix-smoke.json')
    args = parser.parse_args()
    results = []

    def call(name, messages, expected, *, thinking=False, stream=False, repeats=None):
        body = {'model': 'glm-5.3-flash-exl3', 'messages': messages, 'thinking': thinking,
                'temperature': 0, 'max_tokens': 128, 'stream': stream,
                'stream_options': {'include_usage': True}}
        start = time.monotonic()
        request = urllib.request.Request(args.url + '/v1/chat/completions',
                                         data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = response.read().decode()
        if stream:
            assert 'data: [DONE]' in raw, raw
            chunks = [json.loads(line[6:]) for line in raw.splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            assert not any('error' in chunk for chunk in chunks), chunks
            answer = ''.join(choice.get('delta', {}).get('content', '')
                             for chunk in chunks for choice in chunk.get('choices', []))
            usage = next(chunk['usage'] for chunk in chunks if 'usage' in chunk)
        else:
            response = json.loads(raw)
            answer = response['choices'][0]['message']['content']
            usage = response['usage']
        with urllib.request.urlopen(args.url + '/v1/metrics', timeout=5) as response:
            metrics = json.load(response)
        cached = usage['prompt_tokens_details']['cached_tokens']
        assert 0 <= cached < usage['prompt_tokens'], usage
        assert cached == metrics['prefix_last_resume'], (usage, metrics)
        assert expected(answer.lower()), (name, answer)
        if repeats is not None:
            assert cached >= repeats['usage']['prompt_tokens'] - 4, (name, usage)
        result = {'case': name, 'answer': answer, 'usage': usage,
                  'wall_s': time.monotonic() - start, 'metrics': metrics, 'pass': True}
        results.append(result)
        Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return result

    padding = 'This is neutral padding. '
    code = lambda answer: 'amber731' in answer
    mods = set()
    for i in range(4):
        system = f'Fixture {i}. The secret code is AMBER731. ' + padding * 350 + ' q' * i
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': 'What is the secret code? Reply with just the code.'}]
        cold = call(f'text-{i}', messages, code)
        mods.add(cold['usage']['prompt_tokens'] % 4)
        call(f'text-{i}-repeat', messages, code, repeats=cold)
    assert mods == {0, 1, 2, 3}, mods
    initial = call('thinking', messages, code, thinking=True)
    continued = messages + [{'role': 'assistant', 'content': initial['answer']},
                            {'role': 'user', 'content': 'Repeat the secret code.'}]
    call('thinking-history-cleared', continued, code, thinking=True, repeats=initial)

    def image(color):
        buffer = io.BytesIO()
        Image.new('RGB', (224, 224), color).save(buffer, format='PNG')
        return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                base64.b64encode(buffer.getvalue()).decode()}}

    red, blue = image('red'), image('blue')
    system = {'role': 'system', 'content': 'Answer briefly. ' + padding * 350}
    question = {'type': 'text', 'text': 'What color is this image? Reply with one word.'}
    red_messages = [system, {'role': 'user', 'content': [question, red]}]
    first = call('image-red', red_messages, lambda s: 'red' in s)
    repeat = call('image-red-repeat', red_messages, lambda s: 'red' in s, repeats=first)
    assert repeat['metrics']['image_encoder_cache_misses'] == first['metrics']['image_encoder_cache_misses']
    assert repeat['metrics']['image_encoder_cache_hits'] > first['metrics']['image_encoder_cache_hits']
    continued = red_messages + [{'role': 'assistant', 'content': repeat['answer']},
                               {'role': 'user', 'content': 'What color was the image? Reply with one word.'}]
    call('image-chat-extension', continued, lambda s: 'red' in s, repeats=first)
    blue_messages = [system, {'role': 'user', 'content': [question, blue]}]
    changed = call('image-changed-same-placeholders', blue_messages, lambda s: 'blue' in s)
    assert changed['usage']['prompt_tokens_details']['cached_tokens'] < first['usage']['prompt_tokens'] - 4
    call('image-blue-repeat', blue_messages, lambda s: 'blue' in s, repeats=changed)
    removed = [system, {'role': 'user', 'content': 'What is 2 + 2? Reply with just the number.'}]
    call('image-removed', removed, lambda s: s.strip() == '4')
    two = [system, {'role': 'user', 'content': [
        {'type': 'text', 'text': 'Name the colors in image order, separated by a comma.'}, red, blue]}]
    both = call('two-images', two, lambda s: 'red' in s and 'blue' in s and s.index('red') < s.index('blue'))
    call('two-images-repeat', two, lambda s: 'red' in s and 'blue' in s, repeats=both)

    def branch(context):
        return [{'role': 'system', 'content': context},
                {'role': 'user', 'content': 'What is the branch code? Reply with just the code.'}]
    old = call('long-old-branch', branch('The branch code is AMBER731. ' + padding * 2400), code)
    base = 'The branch code is AMBER731. ' + padding * 1850 + '\nThe updated branch code is ORCHID942. '
    orchid = lambda s: 'orchid942' in s
    shorter = call('branch-rollback', branch(base), orchid)
    assert shorter['usage']['prompt_tokens_details']['cached_tokens'] >= 8192
    longer = call('branch-regrown', branch(base + padding * 900), orchid)
    edited_messages = branch(base + padding * 650 + '\nThe suffix has changed. ' + padding * 250)
    edited = call('branch-edited-before-newest-snapshot', edited_messages, orchid)
    # The old branch's ~12k state would be eligible here if it had not been invalidated.
    assert edited['usage']['prompt_tokens_details']['cached_tokens'] < old['usage']['prompt_tokens'] - 4
    call('branch-repeat-SSE', edited_messages, orchid, stream=True, repeats=edited)
    call('unrelated-text', [{'role': 'user', 'content': 'What is 2 + 2? Reply with just the number.'}],
         lambda s: s.strip() == '4')
    print(f'ALL PASS: {len(results)} cache checks', flush=True)


if __name__ == '__main__':
    main()
