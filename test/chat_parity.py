#!/usr/bin/env python3
"""Byte-for-byte check of render_chat against the checkpoint's own chat_template.jinja.

The template file in the model directory is authoritative: what the model saw in training is
whatever that jinja renders, so any divergence changes the prompt. Prompt bugs of this class are
invisible in the engine's output - they look like the model misbehaving - so this compares the
exact bytes instead.

Renders CASE1..CASE4 through transformers' apply_chat_template (with the checkpoint's template,
not the one embedded in tokenizer_config.json) and compares against the same cases printed by
build/helios_chat_test. Keep the case list in sync with src/core/test_chat.cpp.

    ~/shared-venv-gpu/bin/python test/chat_parity.py     # needs transformers
"""
import json
import os
import subprocess
import sys

MODEL = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/models/glm53flash")
BIN = "./build/helios_chat_test"

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("transformers not importable - run with ~/shared-venv-gpu/bin/python")

TEMPLATE = open(os.path.join(MODEL, "chat_template.jinja")).read()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]

c1 = [{"role": "user", "content": "Weather in Paris?"}]
c2 = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ],
    },
    {"role": "tool", "content": "18C and sunny", "tool_call_id": "call_1"},
]
c3 = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello", "reasoning_content": "step 1"},
    {"role": "user", "content": "bye"},
]

# CASE4: two tools, one carrying the keys the reference template strips (strict, defer_loading),
# plus non-string argument types in the parameters schema.
params_term = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "timeout": {"type": "integer", "default": 30},
        "flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["command"],
}
params_search = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
tools2 = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command",
            "parameters": params_term,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search",
            "defer_loading": False,
            "strict": True,
            "parameters": params_search,
        },
    },
]
c4 = [{"role": "user", "content": "how many pythons are running?"}]


def render(msgs, tools_arg, effort="max"):
    kwargs = {"tokenize": False, "add_generation_prompt": True, "chat_template": TEMPLATE}
    kwargs["tools"] = tools_arg
    return tok.apply_chat_template(msgs, reasoning_effort=effort, **kwargs)


reference = {
    "CASE1": render(c1, tools),
    "CASE2": render(c2, tools),
    "CASE3": render(c3, None),
    "CASE4": render(c4, tools2, effort="high"),
}

proc = subprocess.run([BIN], capture_output=True, text=True)
if proc.returncode != 0:
    sys.exit("helios_chat_test failed:\n" + proc.stdout + proc.stderr)

engine = {}
for line in proc.stdout.splitlines():
    if line.startswith("CASE"):
        key, value = line.split(" ", 1)
        engine[key] = json.loads(value)

failed = 0
for key in sorted(reference):
    want, got = reference[key], engine.get(key)
    if got is None:
        print("%s MISSING from helios output" % key)
        failed += 1
    elif want == got:
        print("%s IDENTICAL (%d bytes)" % (key, len(want)))
    else:
        i = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
        print("%s DIFFERS at byte %d (ref %d, ours %d)" % (key, i, len(want), len(got)))
        print("   ref :", json.dumps(want[max(0, i - 70):i + 90]))
        print("   ours:", json.dumps(got[max(0, i - 70):i + 90]))
        failed += 1

print("PARITY %s (%d/%d cases)" % ("FAIL" if failed else "OK", len(reference) - failed, len(reference)))
sys.exit(1 if failed else 0)
