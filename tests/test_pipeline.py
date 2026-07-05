from __future__ import annotations

import unittest

import numpy as np

from llmisms.analysis import _bootstrap_effect
from llmisms.corpus import build_prompts
from llmisms.generation import chat_prompt, textual_token_ids
from llmisms.math_utils import (
    conditioning_gain,
    derived_seed,
    entropy_and_surprisal,
    stage_deltas,
    validate_stage_pairing,
)
from llmisms.patterns import _bh_adjust, _paired_interval, extract_patterns
from llmisms.scoring import (
    lexical_indices,
    score_ablation,
    score_generation,
    select_control,
)
from llmisms.tagging import align_candidate, detect


class CharacterTokenizer:
    all_special_ids = [0]
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(
        self,
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        return "".join(chr(value) for value in ids)


class PrefixLengthScorer:
    tokenizer = CharacterTokenizer()

    def prompt_ids(self, prompt, prompt_format="chat"):
        return [0]

    def continuation_surprisals(self, prefix_ids, continuation_ids):
        value = 2.0 - 0.01 * len(prefix_ids)
        return [value] * len(continuation_ids)


class PipelineTests(unittest.TestCase):
    def test_bootstrap_ignores_missing_effects(self):
        result = _bootstrap_effect(
            [{"prompt_id": "p1", "ablation_gain": None}],
            "ablation_gain",
        )
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["estimate"])

    def test_base_prompt_uses_transcript_format(self):
        tokenizer = CharacterTokenizer()
        ids = chat_prompt(
            tokenizer,
            "Why?",
            tokenize=True,
            prompt_format="transcript",
        )
        self.assertEqual(
            tokenizer.decode(ids),
            "User: Why?\nAssistant:",
        )

    def test_empty_generation_has_durable_empty_trace(self):
        scorer = PrefixLengthScorer()
        scorer.trace = lambda input_ids, response_start: self.fail(
            "empty generation must not invoke the model"
        )
        trace = score_generation(
            scorer,
            {
                "generation_id": "g",
                "prompt_id": "p",
                "stage": "sft",
                "prompt": "Question?",
                "generated_token_ids": [],
            },
        )
        self.assertEqual(trace["trace_status"], "empty_generation")
        self.assertEqual(trace["token_ids"], [])
        self.assertEqual(trace["entropy"], [])
        self.assertEqual(trace["surprisal"], [])

    def test_terminal_special_token_is_stripped_only_as_suffix(self):
        tokenizer = CharacterTokenizer()
        emitted = tokenizer.encode("answer") + [0]
        token_ids, stripped, status, decoded = textual_token_ids(
            tokenizer, "answer", emitted
        )
        self.assertEqual(token_ids, tokenizer.encode("answer"))
        self.assertEqual(stripped, [0])
        self.assertEqual(status, "terminal_special_stripped")
        self.assertEqual(decoded, "answer")

    def test_model_terminal_token_is_stripped_when_tokenizer_omits_it(self):
        tokenizer = CharacterTokenizer()
        terminal = ord("!")
        token_ids, stripped, status, decoded = textual_token_ids(
            tokenizer,
            "answer!!",
            tokenizer.encode("answer!!"),
            [terminal],
        )
        self.assertEqual(token_ids, tokenizer.encode("answer"))
        self.assertEqual(stripped, [terminal, terminal])
        self.assertEqual(status, "terminal_special_stripped")
        self.assertEqual(decoded, "answer")

    def test_entropy_and_surprisal_exact(self):
        logits = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        entropy, surprisal = entropy_and_surprisal(logits, np.array([0, 1]))
        self.assertAlmostEqual(float(entropy[0]), np.log(2), places=6)
        self.assertAlmostEqual(float(surprisal[0]), np.log(2), places=6)
        expected_second = np.log(np.exp(2) + 1)
        self.assertAlmostEqual(float(surprisal[1]), expected_second, places=6)

    def test_detector_positive_and_negative_fixtures(self):
        contrast = detect("It is not a storage problem, but a retrieval problem.")
        self.assertEqual([hit.family for hit in contrast], ["contrastive_negation"])
        core = detect("Memory is not storage, it is reconstruction.")
        self.assertEqual([hit.family for hit in core], ["contrastive_negation"])
        contraction = detect("Memory isn't storage; it's reconstruction.")
        self.assertEqual([hit.family for hit in contraction], ["contrastive_negation"])
        enumerative = detect("There are three main reasons: First, cost matters.")
        self.assertEqual(
            [hit.family for hit in enumerative], ["enumerative_preamble"]
        )
        causal_list = detect(
            "The effect occurs for two distinct causes: pressure and heat."
        )
        self.assertEqual(
            [hit.family for hit in causal_list], ["enumerative_preamble"]
        )
        self.assertEqual(detect("It is useful and accurate."), [])
        self.assertEqual(detect("It is not only fast but also reliable."), [])
        self.assertEqual(detect("Do you want me to explain the trick?"), [])
        self.assertEqual(detect("The system is fast - and reliable."), [])

    def test_broader_discourse_detector(self):
        reframing = detect(
            "Rather than treating this as a storage problem, focus on retrieval."
        )
        self.assertEqual(
            [hit.family for hit in reframing], ["contrastive_negation"]
        )
        scaled = detect("It is less about speed and more about reliability.")
        self.assertEqual([hit.family for hit in scaled], ["contrastive_negation"])
        concession = detect(
            "Although the rule appears simple, its consequences are subtle."
        )
        self.assertEqual(
            [hit.family for hit in concession], ["concessive_qualification"]
        )

    def test_pattern_extraction(self):
        patterns = extract_patterns(
            "It is not merely a technical problem, but a political problem."
        )
        rendered = {" ".join(pattern) for kind, pattern in patterns if kind == "skeleton"}
        self.assertTrue(any("not" in value and "but" in value for value in rendered))
        offers = extract_patterns(
            "The answer is complete. If you want more detail, I can help."
        )
        self.assertFalse(
            any("if you want" in " ".join(pattern) for _, pattern in offers)
        )
    def test_pattern_statistics(self):
        values = {
            "sft": {"a": 0.0, "b": 0.0, "c": 1.0},
            "rlvr": {"a": 1.0, "b": 1.0, "c": 1.0},
        }
        result = _paired_interval(values, "sft", "rlvr", iterations=200)
        self.assertGreater(result["estimate"], 0)
        adjusted = _bh_adjust([0.01, 0.04, 0.2])
        self.assertEqual(adjusted, sorted(adjusted))
        self.assertAlmostEqual(adjusted[0], 0.03)

    def test_character_span_alignment_retokenizes(self):
        text = "It is not X, but Y."
        hit = detect(text)[0]
        aligned = align_candidate(CharacterTokenizer(), text, hit)
        self.assertEqual(aligned["token_start"], hit.char_start)
        self.assertEqual(aligned["token_end"], hit.char_end)
        self.assertEqual(aligned["boundary_token"], hit.boundary_char)

    def test_conditioning_gain_sign(self):
        self.assertEqual(conditioning_gain(1.2, 1.7), 0.5)

    def test_scaffold_and_matched_deletion_scoring(self):
        text = (
            "Opening. It is not storage, but retrieval drives the result. "
            "Ordinary material continues for many words. "
            "Another ordinary sentence gives enough lexical continuation."
        )
        tokenizer = CharacterTokenizer()
        hit = detect(text)[0]
        aligned = align_candidate(tokenizer, text, hit)
        span = {"token_start": aligned["token_start"], "boundary_token": aligned["boundary_token"]}
        result = score_ablation(
            PrefixLengthScorer(),
            {
                "prompt": "Explain memory.",
                "generated_token_ids": tokenizer.encode(text),
            },
            {"entropy": [2.0] * len(text)},
            span,
            [(aligned["token_start"], aligned["boundary_token"])],
        )
        self.assertEqual(result["ablation_status"], "ok")
        self.assertGreater(result["scaffold_gain"], 0)
        self.assertAlmostEqual(result["ablation_gain"], 0.0)

    def test_stage_pairing(self):
        self.assertEqual(
            stage_deltas({"sft": 1.0, "dpo": 1.4, "rlvr": 1.5}),
            (0.3999999999999999, 0.10000000000000009),
        )
        rows = [
            {
                "prompt_id": "p1",
                "base_seed": 7,
                "stage": stage,
                "greedy": False,
            }
            for stage in ("base", "sft", "dpo", "rlvr")
        ]
        self.assertTrue(validate_stage_pairing(rows)["valid"])
        self.assertFalse(validate_stage_pairing(rows[:-1])["valid"])
        self.assertIsNone(validate_stage_pairing([])["valid"])

    def test_deterministic_seeds(self):
        first = derived_seed(1729, "prompt-1")
        self.assertEqual(first, derived_seed(1729, "prompt-1"))
        self.assertNotEqual(first, derived_seed(1729, "prompt-2"))

    def test_control_excludes_detected_span(self):
        control = select_control(
            50,
            10,
            14,
            [(10, 14)],
            [2.0] * 50,
        )
        self.assertIsNotNone(control)
        start, end = control
        self.assertFalse(start < 14 and end > 10)

    def test_lexical_window_skips_initial_numbering(self):
        tokenizer = CharacterTokenizer()
        text = " 1. First, cause matters and context follows"
        indices = lexical_indices(tokenizer, tokenizer.encode(text), limit=2)
        self.assertEqual("".join(text[index] for index in indices), "ca")

    def test_frozen_corpus_shape(self):
        prompts = build_prompts()
        self.assertEqual(len(prompts), 240)
        self.assertEqual(len({row["text"] for row in prompts}), 240)
        counts = {
            stratum: sum(row["stratum"] == stratum for row in prompts)
            for stratum in {row["stratum"] for row in prompts}
        }
        self.assertEqual(set(counts.values()), {60})


if __name__ == "__main__":
    unittest.main()
