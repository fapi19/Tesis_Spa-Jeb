SA-BiNLLB Implementation Plan (Phases 1-8)

Adaptation rules

The plan literally prescribes a project/ tree, but we adapt onto existing dirs:





New NMT code: src/nmt/ (legacy moved to src/nmt/_legacy/)



NMT data outputs: data/processed/05_nmt_canonical/, 06_nmt_filtered/, 07_nmt_augmented/



Configs: config/nmt/*.yaml



Checkpoints: models/nmt/<run_name>/ (e.g. models/nmt/nllb_bidi_lora_v0/)



Reports: reports/05_nmt/{preprocessing,training,evaluation,reranking,augmentation}/



Phase-runner scripts: scripts/nmt/NN_<step>.py, invoked via python -m



Source of truth for raw pairs: existing data/processed/04_splits/{train,valid,test}.jsonl (closed by embeddings work). The NMT pipeline never re-derives splits — it inherits the same pair_id/group_id to avoid leakage between embeddings retrieval and NMT training.



Phase 0 — Environment + scaffolding (prerequisite)

Python version: 3.12 (not 3.11 as plan §6 prescribes)

We deviate from plan §6 with explicit rationale, to be documented in the thesis methodology and in the README:





Every NMT pin in plan §7 supports Python 3.12 (verified against package metadata: torch 2.5.1, transformers 4.52.4, peft 0.15.2, accelerate 1.7.0, sentence-transformers 4.1.0, faiss-cpu 1.11.0, unbabel-comet 2.2.4, bert-score 0.3.13, sacrebleu 2.5.1, evaluate 0.4.3, numpy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1).



The existing project (embeddings, thesis tooling) already requires >=3.12 — using 3.12 avoids installing a second Python interpreter and keeps the development surface unified.



Plan §6's "3.11" reads as a recipe-level convention, not a hard dependency constraint.



The real isolation requirement is separate dependency env (because plan §7 uses exact pins while pyproject.toml uses loose >= pins for the embeddings stack), not a separate Python.

Steps





Create requirements/nmt.txt with the exact pins from plan section 7 (transformers 4.52.4, peft 0.15.2, unbabel-comet 2.2.4, faiss-cpu 1.11.0, sentence-transformers 4.1.0, sacrebleu 2.5.1, bert-score 0.3.13, evaluate 0.4.3, etc.).



Create a dedicated .venv-nmt/ with Python 3.12 (python3.12 -m venv .venv-nmt or uv venv --python 3.12 .venv-nmt). Keep .conda-emb/ and the Poetry-managed embeddings env intact and untouched.



Verify with python -c "import torch, transformers, peft, sentence_transformers, faiss, comet, sacrebleu, bert_score; print('ok')".



Document activation and the 3.11→3.12 deviation in README.md ("Pipeline NMT (NLLB + LoRA)" section appended).



Move legacy NMT code: src/nmt/{train_nmt,clean_train_data,prepare_clean_splits,evaluate_mt,model,dataset,utils}.py → src/nmt/_legacy/. Keep them runnable for back-references but don't import them from new code.



New empty package: src/nmt/{preprocessing,training,reranking,evaluation,inference,augmentation}/__init__.py.



New configs:





config/nmt/filter.yaml — thresholds 0.45 / 0.60 (plan §13).



config/nmt/training.yaml — LoRA r=16, alpha=32, lr=2e-4, etc. (plan §23-25).



config/nmt/inference.yaml — beam=5, length_penalty=1.0, max_new_tokens=128 (plan §32).



config/nmt/reranker.yaml — weights 0.7 / 0.3 (plan §34).



config/nmt/eval.yaml — metric registry (plan §30).



Phase 1 — Dataset freeze (canonical CSV)

The plan specifies the canonical schema id,source,target,source_lang,target_lang,split (§9-10). The existing data/processed/04_splits/{train,valid,test}.jsonl already preserves morphology and apostrophes, so this phase is a re-export, not re-cleaning.





New script: scripts/nmt/10_canonicalize_dataset.py → python -m scripts.nmt.10_canonicalize_dataset.



For each split (train, valid, test) and each row, emit two rows: one shw → spa and one spa → shw. Inherit pair_id and group_id from the embeddings split to keep downstream traceable.



Keep data/processed/04_splits/ untouched; write to data/processed/05_nmt_canonical/{train,valid,test}.csv with columns: id,pair_id,group_id,source,target,source_lang,target_lang,split,has_audit_flags,origin_source.



id = f"{pair_id}__{src_lang}2{tgt_lang}".



Use shw and spa as language codes (matching plan §19 <2shw> / <2spa>); we map them to NLLB tokenizer codes (shw_Latn, spa_Latn) at training time.



Manifest: reports/05_nmt/preprocessing/canonical_manifest.json — counts per direction and split, sha256 of inputs.



Sanity: row counts should be 2 × (2563+320+321) = 6408. Same group_id never appears in two splits.



Phase 2 — Semantic filtering + FAISS

Strongest contribution per plan §11 and §42. Uses the closed embedding model models/sentence_transformers/v3_iterative_hn_e5_base_bidirectional exactly as-is.





Module: src/nmt/preprocessing/semantic_filter.py. Loads the SBERT model, encodes source and target per row, computes cosine similarity (rows are L2-normalized inside the model since it has a 2_Normalize block).



For e5-base, the model expects the prefixes query:  and passage: . We probe the embedding model's tokenizer; if the closure was trained with raw inputs (no prefix), match what v3_iterative_hn_e5_base_bidirectional was trained with — see src/embeddings/train_embedding_model.py and reports/04_embeddings/v3_iterative_hn_e5_base_bidirectional/ — and replicate that exact preprocessing here. No silent divergence.



Apply rules from config/nmt/filter.yaml:

# pseudo
if score < 0.45:
    label = "removed"
elif score <= 0.60:
    label = "flagged_for_review"
else:
    label = "accepted"





Filter is applied only to train. valid and test are passed through unchanged with their score column attached (so we can audit them but never lose them — gold splits stay frozen).



Important: filter is computed once per pair_id (not per direction), so the two directional rows for the same pair share a label. Avoids inconsistent treatment.



Outputs:





data/processed/06_nmt_filtered/train.csv — accepted only, with score column.



data/processed/06_nmt_filtered/train_flagged.csv — manual-review queue.



data/processed/06_nmt_filtered/train_removed.csv — kept for traceability.



data/processed/06_nmt_filtered/{valid,test}.csv — full passthrough with score.



Reports: reports/05_nmt/preprocessing/semantic_filter.json (counts per bucket, score histogram, mean/std per source flashcards vs pdf_textos, list of top-k worst pairs in train).

FAISS index (plan §14)





Module: src/nmt/preprocessing/faiss_index.py. Builds faiss.IndexFlatIP on accepted-train embeddings, separately for the Shiwilu side and Spanish side.



Stores: data/processed/06_nmt_filtered/faiss_{shw,spa}.index + faiss_{shw,spa}_meta.parquet (id ↔ row mapping).



Used downstream by:





Phase 7 (mining bilingual neighbors for augmentation).



Reranking sanity checks (find nearest train neighbor of a candidate to detect hallucinations / memorization).



Optional: detect near-duplicates by IP > 0.98 within Shiwilu side; report only, don't auto-remove (the embedding pipeline already deduplicated exact pairs).



Phase 3 — SentencePiece Unigram (analytic artifact)

The plan mandates SP Unigram, vocab=8000, char_coverage=1.0 (§15-17). NLLB ships with its own tokenizer, so this SP model is not the runtime tokenizer. Its purpose:





Vocabulary/morphology analysis to defend the agglutinative-friendly choice.



Comparison fixture against NLLB's tokenization (token-count and segmentation deltas).



Possible Shiwilu vocabulary extension proposal for the NLLB tokenizer (registered as a follow-up artifact, not built here).





Module: src/nmt/preprocessing/train_sentencepiece.py. Trains on the accepted train (Shiwilu + Spanish concatenated, one sentence per line) → data/processed/05_nmt_canonical/all_text_for_sp_nmt.txt.



Output: models/nmt/sentencepiece/sp_unigram_8k.{model,vocab}.



Report: reports/05_nmt/preprocessing/sentencepiece_stats.json — vocab size, char coverage, sample tokenizations of 50 Shiwilu sentences contrasted with NLLB tokenizer output.



Phase 4 — NLLB bidirectional LoRA fine-tuning

This is the model-training phase.

4a. Adding Shiwilu to NLLB tokenizer

NLLB language codes are BCP-47–style (e.g. spa_Latn). Shiwilu has none. We register shw_Latn as a new special language token (standard low-resource extension to NLLB) plus keep the plan's <2shw> / <2spa> syntax available as alternative source-side tags (§19).





Module: src/nmt/training/model_setup.py:





Load facebook/nllb-200-distilled-600M tokenizer.



Add new tokens via tokenizer.add_special_tokens({"additional_special_tokens": ["shw_Latn", "<2shw>", "<2spa>"]}).



Extend tokenizer.lang_code_to_id and id_to_lang_code to include shw_Latn.



model.resize_token_embeddings(len(tokenizer)). Initialize the new shw_Latn token's embedding from the mean of [quy_Latn, ayr_Latn, grn_Latn] (Andean / South-American Indigenous neighbors NLLB does support) — better than random init.



Save extended tokenizer to models/nmt/tokenizer_shw_extended/.

4b. Bidirectional dataset builder





Module: src/nmt/training/dataset.py. Reads data/processed/06_nmt_filtered/{train,valid,test}.csv. Each row already has source_lang and target_lang (shw or spa).



For each batch item:





Set tokenizer.src_lang = "shw_Latn" or "spa_Latn" based on source_lang.



Tokenize source (truncate to 128).



With tokenizer.as_target_tokenizer() set tgt_lang and tokenize target.



Use HF DataCollatorForSeq2Seq for dynamic padding.

4c. LoRA training





Module: src/nmt/training/train_lora.py.



LoRA via peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=["q_proj","v_proj"], task_type="SEQ_2_SEQ_LM") (plan §23-24).



Use Seq2SeqTrainingArguments:





learning_rate=2e-4, per_device_train_batch_size=8, gradient_accumulation_steps=4, num_train_epochs=20, warmup_ratio=0.1, weight_decay=0.01, lr_scheduler_type="cosine", optim="adamw_torch", fp16=True (CUDA) / bf16=True (MPS or A100), predict_with_generate=True, generation_num_beams=5, generation_max_length=128 (plan §25-27, §32).



evaluation_strategy="steps", eval_steps=250, save_strategy="steps", save_steps=500, load_best_model_at_end=True, metric_for_best_model="eval_chrf" (plan §28-29, §31).



label_smoothing_factor=0.1.



Custom compute_metrics calls sacrebleu.corpus_chrf (chrF++ via word_order=2) and sacrebleu.corpus_bleu on decoded validation predictions. Skipping COMET/BERTScore in-loop (too slow); they run in Phase 5.



Evaluate validation in both directions separately so we get four headline numbers per checkpoint: bleu_shw2spa, bleu_spa2shw, chrf_shw2spa, chrf_spa2shw. The selector uses the average chrF++ over both directions.



Output:





models/nmt/nllb_bidi_lora_v0/ — adapter weights, tokenizer, training_args.json.



reports/05_nmt/training/nllb_bidi_lora_v0/training_log.json (loss/eval per step), summary.json (best step + headline metrics).

flowchart LR
    subgraph data [Data]
        canonical[05_nmt_canonical CSVs]
        filtered[06_nmt_filtered CSVs]
    end

    subgraph emb [Embedding model v3]
        sbert[v3_iterative_hn_e5_base_bidirectional]
        faiss[FAISS IndexFlatIP]
    end

    subgraph train [Training]
        nllb["NLLB-200 distilled 600M"]
        lora["LoRA r=16, q_proj/v_proj"]
        ckpt["models/nmt/nllb_bidi_lora_v0"]
    end

    canonical --> sfilter[semantic_filter]
    sbert --> sfilter
    sfilter --> filtered
    filtered --> faiss
    filtered --> dataset[Bidirectional dataset shw to spa and spa to shw]
    dataset --> nllb
    nllb --> lora --> ckpt



Phase 5 — Full evaluation suite





Module: src/nmt/evaluation/metrics.py — wrappers for sacrebleu BLEU + chrF++, evaluate.load("bertscore") (model xlm-roberta-large for multilingual support; default for Spanish; for Shiwilu we use the same as a reasonable proxy with the limitation flagged in the metrics output), comet.load_from_checkpoint("Unbabel/wmt22-comet-da"). COMET on the Shiwilu side is reported as an indicative score with explicit caveats (no native Shiwilu support in COMET training) emitted into the metrics JSON.



Module: src/nmt/inference/generate.py — beam=5, length_penalty=1.0, max_new_tokens=128 (plan §32). Returns top-N candidates with sequence_scores for reranking later.



Script: scripts/nmt/30_evaluate.py --checkpoint models/nmt/nllb_bidi_lora_v0 --split test.



Outputs:





reports/05_nmt/evaluation/nllb_bidi_lora_v0/test_predictions.jsonl (one line per test row: id, source, reference, hypothesis, sequence_score).



reports/05_nmt/evaluation/nllb_bidi_lora_v0/test_metrics.json (BLEU, chrF++, BERTScore P/R/F1, COMET, broken down by direction).



Phase 6 — Semantic reranking

The architectural contribution that ties R2 (embeddings) into NMT (plan §33-35).





Module: src/nmt/reranking/semantic_reranker.py:





For each test source, generate num_return_sequences=5 with beam=5 and output_scores=True (greedy/beam scores → length-normalized log-probability → exponentiated to a probability proxy normalized over the n-best).



Encode the source and each candidate with the same embedding model used in Phase 2 (same prefix policy).



cos_sim = candidate_emb · source_emb (both L2-normalized).



final_score = 0.7 * trans_prob + 0.3 * cos_sim (plan §34). Constants come from config/nmt/reranker.yaml.



Re-pick best candidate.



Module: scripts/nmt/40_rerank.py --predictions reports/05_nmt/evaluation/nllb_bidi_lora_v0/test_predictions_topk.jsonl --out ....



Outputs:





reports/05_nmt/reranking/nllb_bidi_lora_v0/test_predictions_reranked.jsonl.



reports/05_nmt/reranking/nllb_bidi_lora_v0/test_metrics_reranked.json — same metric set as Phase 5.



reports/05_nmt/reranking/nllb_bidi_lora_v0/ablation.json — sweep over weight α ∈ {0.0, 0.3, 0.5, 0.7, 1.0} so the chosen 0.7/0.3 split is empirically supported, not arbitrary.



Phase 7 — Backtranslation + augmentation

Strict per plan §36-37: only after Phase 5 is stable. No random synthetic data.

7a. Backtranslation





Source of monolingual Shiwilu: extract additional Shiwilu sentences from data/raw/II_TEXTOS_SHIWILU.pdf that did not make it into the parallel corpus (i.e. Shiwilu-only paragraphs from PDF extraction where Spanish was missing/ambiguous), plus any Shiwilu-only text in the existing all_text_for_sp.txt. Catalog them to data/processed/07_nmt_augmented/mono_shw.txt.



Module: src/nmt/augmentation/backtranslation.py:





Use the v0 model (Spanish→Shiwilu) frozen, generate Spanish from monolingual Shiwilu using beam=5.



Filter the synthetic pairs through the same Phase 2 semantic filter (≥ 0.60). Only accepted pairs survive.



Outputs data/processed/07_nmt_augmented/train_bt.csv with same canonical schema, origin_source="backtranslation_v0".



Documented limit: BT volume is capped to ≤ 2x parallel size to avoid drowning out gold pairs.

7b. Embedding-based mining





Module: src/nmt/augmentation/embedding_mining.py:





Use FAISS index from Phase 2 to find Shiwilu nearest neighbors of each Spanish sentence (cross-lingual mining via the bilingual embedding space).



Accept a synthetic pair only if reciprocal nearest neighbor (R-NN) and IP > 0.65.



Outputs data/processed/07_nmt_augmented/train_mined.csv.

7c. Morphological variants (controlled)





Reuse the Shiwilu suffix inventory at data/processed/04_splits/shiwilu_suffixes.json (already produced by src/embeddings/build_suffix_aware_corpus.py).



Module: src/nmt/augmentation/morphological_variants.py:





For pairs whose Shiwilu side is a single word with a known suffix, generate variants by swapping for closely-related person/aspect suffixes from the inventory with linguist-supervised mapping. Without that supervision, this stage is off by default — the script is registered but emits a "manual review required" status into its run report.

7d. Re-train





Script: scripts/nmt/51_train_with_augmented.py — same training entrypoint as Phase 4, but reads train from concat of 06_nmt_filtered/train.csv ∪ 07_nmt_augmented/train_bt.csv ∪ 07_nmt_augmented/train_mined.csv.



Output: models/nmt/nllb_bidi_lora_v1_bt/.



Reports: reports/05_nmt/training/nllb_bidi_lora_v1_bt/.



Phase 8 — Final evaluation + thesis integration





Re-run Phase 5 (full metrics) and Phase 6 (reranking) on nllb_bidi_lora_v1_bt.



Side-by-side comparison report: reports/05_nmt/evaluation/comparison_v0_vs_v1_bt.md with three columns per direction (BLEU / chrF++ / BERTScore-F1 / COMET) for: nllb_bidi_lora_v0, nllb_bidi_lora_v0 + reranker, nllb_bidi_lora_v1_bt, nllb_bidi_lora_v1_bt + reranker.



Human evaluation (plan §38):





Sampling protocol: stratified random sample of 100 test items per direction, balancing source distribution (flashcards vs pdf_textos) and length buckets.



Output template reports/05_nmt/evaluation/human_eval_template.csv with columns id, source, reference, hypothesis_v0, hypothesis_v1_bt, hypothesis_v1_bt_reranked, adequacy_1_5, fluency_1_5, cultural_relevance_1_5, notes.



Anonymize columns (random model-letter mapping) so reviewers can't bias by knowing which is which.



Update thesis/latex/tesis.tex with NMT chapter sections referencing the comparison + ablations + sample translations. Tables generated from the JSON reports via a small scripts/generate_nmt_tables.py.



Risks and known constraints (worth surfacing in the thesis)





2.5k parallel pairs is extremely low. The plan's 20 epochs × LoRA-only is the right scale; full FT or larger LoRA would overfit. We must report variance over ≥ 3 seeds for the headline run (v1_bt).



fp16 on Apple Silicon (MPS) is unstable; switch to bf16 automatically when device.type == "mps". CUDA path uses fp16 per plan.



COMET / BERTScore have no native Shiwilu support; report them as proxy scores with a clear caveat. chrF++ is the headline metric (plan §31), and BLEU is reported as secondary.



The plan's <2shw> tag style coexists with NLLB's shw_Latn extension. We pick shw_Latn because it integrates with NLLB's forced_bos_token_id API; <2shw>/<2spa> are added as additional special tokens for compatibility, not used at training time. Document this design decision in the thesis methodology.



Backtranslation depends on a volume of monolingual Shiwilu that may be small. If mono_shw.txt < 500 sentences after dedup, BT will not move metrics; we document this limit and avoid claiming gains that don't exist.



Human evaluation requires native speakers — out of scope for me to schedule, but the protocol artifact and anonymized template are produced.



Python 3.12 is used instead of plan §6's 3.11 to align with the existing project stack. All pinned NMT dependencies in plan §7 support 3.12; isolation is achieved via a separate .venv-nmt. Deviation is recorded in the README and in the thesis reproducibility section.



Pinning is "middle ground": exact == pins on the ML stack (torch, transformers, accelerate, datasets, sentencepiece, peft, sentence-transformers) and the eval stack (evaluate, sacrebleu, bert-score, unbabel-comet, faiss-cpu) to keep model/metric behavior reproducible, with >= on glue libs (numpy<2, pandas, scikit-learn, tqdm, pyyaml, pyarrow, matplotlib). Plan §7 prescribes exact pins on every line; the relaxation for glue libs is a deliberate trade-off to avoid resolver brittleness, recorded here and in requirements/nmt.txt.



numpy<2 upper bound is mandatory, regardless of pinning strategy, because unbabel-comet==2.2.4 hard-pins numpy<2.0.0. Verified by pip's resolver. COMET is non-negotiable per plan §30.



setuptools<81 is required because setuptools 81+ removed pkg_resources (Oct 2025), which torchmetrics (transitive dep of pytorch_lightning, transitive dep of unbabel-comet 2.2.4) still imports at module load. Without this pin, import comet fails with ModuleNotFoundError: No module named 'pkg_resources'. Recorded in requirements/nmt.txt.



torch bumped from 2.5.1 (plan §7) to 2.7.1, plus matching bumps transformers 4.52.4 -> 4.55.0, accelerate 1.7.0 -> 1.8.1, peft 0.15.2 -> 0.16.0. Reason: training will run on an NVIDIA RTX 5060 Ti (Blackwell consumer, compute capability sm_120), and torch 2.5.1 has no prebuilt CUDA kernels for sm_120 — would fall back to PTX JIT with warnings. torch 2.7 added sm_120 support. Documented in requirements/nmt.txt header.



Hardware target: training runs on RTX 5060 Ti 16GB (CUDA 12.8). Mac (MPS) is the dev/exploration env. Estimated 5-10x speedup vs MPS for NLLB-600M+LoRA training.

