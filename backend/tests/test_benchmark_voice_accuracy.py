# Covers word_error_rate() from scripts/benchmark_voice_accuracy.py -
# the WER math the accuracy benchmark's whole output depends on, so a
# regression here wouldn't just be a bug, it'd be a silently wrong
# accuracy report.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmark_voice_accuracy import word_error_rate


def test_identical_transcripts_have_zero_wer():
    assert word_error_rate("what is the profit", "what is the profit") == 0.0


def test_one_substitution_out_of_four_words():
    # The exact real bug this whole benchmark exists to catch, as a
    # worked WER example: "profit" -> "prefer" is 1 wrong word out of 4.
    assert word_error_rate("what is the profit", "what is the prefer") == 0.25


def test_one_deletion_out_of_four_words():
    assert word_error_rate("what is the profit", "what is profit") == 0.25


def test_completely_different_transcript_is_high_wer():
    assert word_error_rate("what is the profit", "hello there world") >= 0.75


def test_case_and_punctuation_are_ignored():
    assert word_error_rate("What is the profit?", "what is the profit") == 0.0


def test_empty_expected_and_empty_actual_is_zero_wer():
    assert word_error_rate("", "") == 0.0


def test_empty_expected_but_actual_has_words_is_full_wer():
    assert word_error_rate("", "unexpected words here") == 1.0
