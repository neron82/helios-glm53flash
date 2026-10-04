#!/usr/bin/env python3
"""Check a running server's text, SSE, images, and image-prefix invalidation.

Requires Pillow and the DejaVu Sans font for the OCR fixture. Run with --url http://127.0.0.1:18081 on glm53flash_abl.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import re
import urllib.request
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18081')
    parser.add_argument('--output', default='/tmp/helios-generation-smoke.json')
    parser.add_argument('--text-only', action='store_true')
    args = parser.parse_args()
    results = []

    def request(content, stream=False):
        body = {'model': 'helios', 'messages': [{'role': 'user', 'content': content}],
                'thinking': False, 'temperature': 0, 'max_tokens': 96, 'stream': stream}
        req = urllib.request.Request(args.url + '/v1/chat/completions',
                                     data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=240) as response:
            raw = response.read().decode()
        if stream:
            assert 'data: [DONE]' in raw, raw
            chunks = [json.loads(line[6:]) for line in raw.splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            return ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c.get('choices'))
        return json.loads(raw)['choices'][0]['message']['content']

    def check(name, content, predicate, stream=False):
        answer = request(content, stream)
        valid = bool(predicate(answer.lower()))
        results.append({'case': name, 'answer': answer, 'pass': valid})
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(json.dumps(results[-1]), flush=True)
        if not valid:
            raise AssertionError(name + ': ' + answer)

    def image(color, size=(224, 224), label=None):
        buffer = io.BytesIO()
        picture = Image.new('RGB', size, color)
        if label:
            ImageDraw.Draw(picture).text((24, 75), label, fill='black',
                                        font=ImageFont.truetype('DejaVuSans.ttf', 64))
        picture.save(buffer, format='PNG')
        return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                base64.b64encode(buffer.getvalue()).decode()}}

    check('arithmetic', 'What is 2 + 2? Reply with just the number.', lambda s: bool(re.fullmatch(r'\s*4[.!]?\s*', s)))
    check('translation', 'Translate hello into French. Reply with just the French word.', lambda s: 'bonjour' in s)
    check('SSE', 'What is the capital of France? Answer briefly.', lambda s: 'paris' in s, True)
    if args.text_only:
        print('ALL PASS', flush=True)
        return
    for color in ('red', 'blue', 'red'):
        check('image-' + color, [{'type': 'text', 'text': 'What color is this image? Answer with one color word.'}, image(color)], lambda s, color=color: color in s)
    check('image-nonsquare', [{'type': 'text', 'text': 'What color is this image? Answer with one color word.'}, image('blue', (320, 168))], lambda s: 'blue' in s)
    check('two-images', [{'type': 'text', 'text': 'Name the color of each image in order, separated by a comma.'}, image('red'), image('blue')], lambda s: 'red' in s and 'blue' in s and s.index('red') < s.index('blue'))
    check('image-OCR', [{'type': 'text', 'text': 'Read the text in this image. Reply with only the text.'}, image('white', (448, 224), 'HELLO 42')], lambda s: 'hello 42' in s)
    check('text-after-image', 'What is 2 + 2? Reply with just the number.', lambda s: bool(re.fullmatch(r'\s*4[.!]?\s*', s)))
    print('ALL PASS', flush=True)


if __name__ == '__main__':
    main()
