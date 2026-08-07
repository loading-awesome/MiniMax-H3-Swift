#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sean Kammerich
"""Transcribe a rendered waveform and score it against the prompt's dialogue.

The second half of an oracle that needs no golden. `h3-parity face-check` asks
whether a face is there; this asks whether the words are.

**Whisper hallucinates on non-speech, so a transcript alone proves nothing.**
Run on the audio of a coffee-cup render — a scene with no speech in it at all —
it returned:

    " Can he take a camera in camera camera camera camera camera camera ..."

with `no_speech_prob` only 0.47. A check that just printed that transcript would
have reported speech. So three guards run before any score is believed:

  1. **`no_speech_prob`** per segment, averaged and reported.
  2. **Degenerate repetition** — the fraction of the transcript made up of its
     single most common word, and the longest run of one repeated word. The
     hallucination above is 95% one token; real dialogue is not.
  3. **Scoring against the expected words**, not just "did it produce text".
     Reported as keyword recall (which of the expected content words appear)
     and word error rate.

Exit status is 0 only when speech is found AND matches. Silence, hallucination
and wrong-words are three different failures and are named separately.

    speech_check.py render.wav --expect "It stopped raining."
"""
import argparse
import collections
import json
import re
import sys
import warnings

warnings.filterwarnings("ignore")

STOP = {"a", "an", "the", "is", "it", "to", "of", "and", "in", "on", "at", "i",
        "you", "we", "they", "he", "she", "that", "this", "for", "with", "as",
        "was", "were", "be", "been", "are", "am", "s", "t"}


# Whisper writes numbers as numerals: "The train leaves at nine" comes back as
# "at 9", and "the library closes at seven" as "at 7". Scored literally that is
# a substitution plus a missed keyword — WER 0.20 and 2/3 recall on a render
# whose speech was perfect. The oracle was charging renders for the ASR's
# formatting, so both sides normalise to digits before anything is compared.
NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}


def norm(text):
    return [NUMBERS.get(w, w) for w in re.findall(r"[a-z0-9']+", text.lower())]


def wer(ref, hyp):
    """Levenshtein over words, normalised by reference length."""
    if not ref:
        return 0.0 if not hyp else 1.0
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[len(ref)][len(hyp)] / len(ref)


def repetition(words):
    """(dominant-word share, longest single-word run). Whisper's tell."""
    if not words:
        return 0.0, 0
    share = collections.Counter(words).most_common(1)[0][1] / len(words)
    longest = run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        longest = max(longest, run)
    return share, longest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--expect", default=None,
                    help="the dialogue the prompt asked for")
    # large-v3, not small. `small` reports false failures on exactly the
    # material this check exists to judge: on a line shouted over a cheering
    # crowd it returned "Inheld! The line helped!" and FAILed at WER 0.60,
    # where large-v3 transcribed "It held! The line held!" exactly, WER 0.00.
    # A gate whose default fails clean audio is worse than no gate.
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    ap.add_argument("--json-out")
    a = ap.parse_args()

    import mlx_whisper
    r = mlx_whisper.transcribe(a.audio, path_or_hf_repo=a.model, verbose=False)
    text = r["text"].strip()
    segs = r.get("segments", [])
    words = norm(text)
    share, longest = repetition(words)
    nsp = [s.get("no_speech_prob", 0.0) for s in segs]
    mean_nsp = sum(nsp) / len(nsp) if nsp else 1.0

    print(f"speech-check {a.audio}")
    print(f"  model      {a.model}")
    print(f"  language   {r.get('language')}")
    print(f"  segments   {len(segs)}   mean no_speech_prob {mean_nsp:.2f}")
    print(f"  words      {len(words)}   dominant-word share {share:.2f}   "
          f"longest run {longest}")
    print(f"\n  transcript: {text[:400]}{'...' if len(text) > 400 else ''}")

    # ---- guards
    problems = []
    hallucinating = share > 0.4 or longest >= 6
    if hallucinating:
        problems.append(
            f"degenerate repetition (dominant word {share:.0%}, run of {longest}) "
            "— this is Whisper looping on non-speech, not a transcript")
    if not words:
        problems.append("no words at all")
    if mean_nsp > 0.6:
        problems.append(f"mean no_speech_prob {mean_nsp:.2f} — model thinks this is not speech")

    result = {"audio": a.audio, "text": text, "words": len(words),
              "dominant_share": share, "longest_run": longest,
              "mean_no_speech_prob": mean_nsp, "hallucinating": hallucinating}

    # ---- scoring, only meaningful once the guards pass
    if a.expect:
        exp = norm(a.expect)
        got = words
        w = wer(exp, got)
        keys = [x for x in exp if x not in STOP]
        hit = [k for k in keys if k in got]
        recall = len(hit) / len(keys) if keys else 0.0
        print(f"\n  expected  : {' '.join(exp)}")
        print(f"  WER       : {w:.2f}")
        print(f"  keywords  : {len(hit)}/{len(keys)} recalled ({recall:.0%})"
              + (f"  hit: {', '.join(hit)}" if hit else ""))
        missed = [k for k in keys if k not in got]
        if missed:
            print(f"  missed    : {', '.join(missed)}")
        result.update(wer=w, keyword_recall=recall, expected=" ".join(exp))
        if not hallucinating and recall < 0.5:
            problems.append(f"only {recall:.0%} of expected keywords present")

    print("")
    if problems:
        for p in problems:
            print(f"  FAIL: {p}")
    else:
        print("  PASS: intelligible speech" + (" matching the prompt" if a.expect else ""))
    result["problems"] = problems

    if a.json_out:
        json.dump(result, open(a.json_out, "w"), indent=1)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
