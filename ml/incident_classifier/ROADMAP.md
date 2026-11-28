# Development roadmap

Select one useful, bounded improvement at a time. Record tests and measured results.
Avoid changing the fixed synthetic test partition to improve reported metrics.

1. Completed 2026-11-23: batch JSONL inference, line-level validation, and documented error behaviour. See BATCH_INFERENCE.md.
2. Completed 2026-11-24: multinomial naive Bayes comparison and training-only stratified cross-validation. See CROSS_VALIDATION.md.
3. Completed 2026-11-27: multi-seed summaries and paired differences. See MULTI_SEED_COMPARISON.md. Inferential confidence intervals remain deferred pending a design that accounts for dependent folds and representative evaluation data.
4. Completed 2026-11-28: training-only out-of-fold ensemble uncertainty and confidence/review curves. See ENSEMBLE_REVIEW.md. No deployment threshold is selected.
5. Add a licensed public incident-text dataset importer and provenance tracking.
6. Compare learned mean pooling against attention pooling and a small transformer.
7. Introduce multi-label classification for incidents spanning multiple categories.
8. Evaluate English/Swahili inputs separately using reviewed, representative data.
9. Add a local inference API with schema validation and request limits.
10. Add an opt-in dashboard preview with human confirmation before accepting suggestions.

Other portfolio work: resume the owner's image-captioning repository with a
reproducible data interface, image encoder, attention decoder, and honest evaluation.
Keep distinct projects in appropriate existing repositories or newly created
repositories when repository-creation access is available.
