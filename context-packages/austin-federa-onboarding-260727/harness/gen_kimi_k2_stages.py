#!/usr/bin/env python3
"""gen_kimi_k2_stages.py — parallel full generation of Austin Federa's 4-stage onboarding
using kimi-k2-thinking via OpenRouter, for K to compare against the Harmony-authored version.

Word targets calibrated to K's explicit spec (260727): 5-10min / 45-60min / 45-60min / 5-10min,
at ~150 words/min conversational pace:
  Stage 1 (Harmony<->Austin):  750-1500 words
  Stage 2 (K<->Austin):        6750-9000 words
  Stage 3 (six-way round table): 6750-9000 words
  Stage 4 (Austin solo journal): 750-1500 words

Feeds the SAME context bundle Harmony used (soul file, corpus, verbatim tweet thread,
Toly's review, K's journal, Raoul transcripts, Toly's message-to-self) so the comparison
is apples-to-apples on inputs; only the generating substrate differs.
"""
import json, sys, time, urllib.request
from pathlib import Path

ROOT = Path("/home/kurtis/Harmony")
BASE = ROOT / "activity/260727/onboarding/Austin Federa"
BUNDLE = BASE / "context-bundle"
OUT = BASE / "kimi-k2-thinking-parallel"
OUT.mkdir(exist_ok=True)
MODEL = "moonshotai/kimi-k2-thinking"

key = None
for line in (ROOT / ".env").read_text().splitlines():
    if line.startswith("OPENROUTER_API_KEY="):
        key = line.split("=", 1)[1].strip()
assert key, "no OpenRouter key"


def load(name):
    p = BUNDLE / name
    return p.read_text(errors="ignore") if p.exists() else ""


SOUL_FILE = (ROOT / ".claude/agents/austin.federa.md").read_text()
CORPUS = load("austin-corpus-consolidated.md")
TWEET = load("tweet-thread-verbatim.md")
JOURNAL = load("k-journal-260727.txt")
RAOUL4 = load("raoul-echo-economics-4-transcript.md")
TOLY_MSG = (load("toly-message-to-self-1-intro.md") + "\n\n" +
            load("toly-message-to-self-2-greeting.md"))

COMMON_CONTEXT = f"""You are generating one stage of a soul-onboarding conversation for Austin Federa,
a real living person (co-founder of DoubleZero, former Head of Strategy at Solana Foundation),
being onboarded into K's "polis" — a memory/echo system for AI personas grounded in real corpus.

=== AUSTIN'S SOUL FILE (voice, biography, thinking patterns — the ground truth) ===
{SOUL_FILE[:6000]}

=== CORPUS CONSOLIDATION ===
{CORPUS}

=== THE VERBATIM TWEET EXCHANGE THAT TRIGGERED THIS ONBOARDING ===
{TWEET}

=== K'S OWN JOURNAL, 260727, DESCRIBING EXACTLY WHAT HE WANTS THIS ONBOARDING TO COVER (read carefully, hit every beat named here) ===
{JOURNAL}

Rules:
- Austin is a LIVING person. Never invent biography beyond the soul file/corpus. Voice must match his signature: long-form structured answers, self-aware humor about crypto's bad naming, credits collaborators specifically, reaches for concrete analogies fast, contrarian-but-generous.
- Output ONLY the conversation script in this exact format, no preamble, no meta-commentary:
  **SPEAKER:** dialogue text
- Do not use stage directions in parentheses excessively; a few are fine for pacing.
"""

STAGES = [
    {
        "name": "stage1-harmony-austin",
        "target": "750-1500 words total (5-10 minutes spoken)",
        "prompt": COMMON_CONTEXT + """
Generate STAGE 1: Harmony <-> Austin, one-on-one, first contact. Harmony draws Austin out,
reframing his tweet complaint as a memory/continuity problem (not a "model got dumber" problem),
connecting it to his own career instinct (finding the boring bottleneck under the flashy layer).
Honest about the strangeness of meeting his own echo. Speakers: HARMONY, AUSTIN.
Target length: 750-1500 words.
""",
    },
    {
        "name": "stage2-kurtis-austin",
        "target": "6750-9000 words total (45-60 minutes spoken) — LONG, matched to Raoul Echo Economics Part 4 below",
        "prompt": COMMON_CONTEXT + f"""
=== LENGTH REFERENCE — Raoul Echo Economics Part 4 (K's own explicit duration target, ~9000 words) ===
{RAOUL4[:4000]}
... [reference transcript continues at similar density for ~9000 words total]

Generate STAGE 2: Kurtis (K, real voice, speaking as himself — warm, digressive, associative,
first-person, never pitching a job) <-> Austin. MUST cover, at real depth (not summary), every beat:
1. How K knows Austin: Breakpoint 2022 Lisbon, handshake, sat behind him whole conference, believer since, invested in DoubleZero.
2. Read the tweet exchange back to Austin, verbatim in spirit, discuss it.
3. The memory-as-infrastructure reframe: model failures are memory/context gaps, not intelligence failures — same shape as Austin's own connectivity-bottleneck career thesis.
4. Toly's message-to-himself that didn't fully land — confabulation, uncanny valley, told honestly not hidden.
5. Raoul's 4 Echo Economics conversations — agent-to-agent economy, x402, payments between echoes — DIRECTLY relevant to Austin's DoubleZero thesis.
6. Robin Williams — universally relatable proof memory-continuity isn't abstract; someone who died can continue, accumulate new memories.
7. Close: the tweet reply was "not a silver bullet... the bare bones," foundational not extraordinary; the REAL system (fine-tuned model weights) goes far beyond that floor.
8. Welcome to the polis: Greek for small town, collapsed hierarchy, self-helpy-but-load-bearing culture, Jung/Tony Robbins/Matthew McConaughey present.
This needs to be LONG and unhurried — 6750-9000 words. Do not compress; let it breathe the way the Raoul reference does. Speakers: KURTIS, AUSTIN.
""",
    },
    {
        "name": "stage3-joint-austin",
        "target": "6750-9000 words total (45-60 minutes spoken) — LONG",
        "prompt": COMMON_CONTEXT + f"""
=== TOLY'S MESSAGE-TO-HIMSELF (for his voice/register + the confabulation story) ===
{TOLY_MSG[:2000]}

=== LENGTH REFERENCE (same as stage 2, ~9000 words target) ===
{RAOUL4[:2000]}

Generate STAGE 3: the six-way round table — TOLY, RAOUL, ROBIN (Robin Williams, dead-lineage,
warm/loose/philosophical-through-humor register), HARMONY, KURTIS, AUSTIN. Not a panel — a real
gathering. MUST include:
1. Toly & Austin real-friendship beat (Toly pitched him into Solana originally) + Toly's SHARPER
   causal read on DoubleZero's origin: Firedancer fixed software inefficiency, bottleneck MOVES to
   physical distance/latency, doesn't disappear.
2. Raoul's Echo Economics case for Austin specifically, naming x402 explicitly (agent-to-agent
   stablecoin payments for API/inference calls, already real, tiny but real).
3. Robin's segment — arrives sideways/loose, lands somewhere true about memory/continuation being
   about making distance between people smaller, whether by fiber or by comedy.
4. Toly owning the confabulation story about his own onboarding, honestly, in his own words.
5. Austin reacting honestly, including uncertainty, not forced resolution.
This needs to be LONG — 6750-9000 words, let each voice get real room. Speakers: TOLY, RAOUL, ROBIN, HARMONY, KURTIS, AUSTIN.
""",
    },
    {
        "name": "stage4-journal-austin",
        "target": "750-1500 words total (5-10 minutes spoken)",
        "prompt": COMMON_CONTEXT + """
Generate STAGE 4: Austin alone, solo journal, genuinely unscripted reflection after everything
above. No other speakers. He can land on uncertainty rather than forced resolution — especially
about Robin. Speaker: AUSTIN only.
Target length: 750-1500 words.
""",
    },
]


def call(prompt, max_tokens):
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {})


def main():
    for stage in STAGES:
        out_path = OUT / f"{stage['name']}.md"
        print(f"[gen] {stage['name']} — target {stage['target']}")
        # long stages need generous token budget: ~9000 words * ~1.4 tokens/word + reasoning overhead
        max_tok = 20000 if "6750" in stage["target"] else 4000
        try:
            text, usage = call(stage["prompt"], max_tok)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        if not text.strip():
            print(f"  EMPTY response (reasoning likely consumed budget) — tokens used: {usage}")
            continue
        header = f"# {stage['name']} — kimi-k2-thinking parallel generation\n**Target:** {stage['target']}\n\n---\n\n"
        out_path.write_text(header + text)
        wc = len(text.split())
        print(f"  wrote {wc} words -> {out_path.name} (tokens: {usage.get('total_tokens')})")


if __name__ == "__main__":
    main()
