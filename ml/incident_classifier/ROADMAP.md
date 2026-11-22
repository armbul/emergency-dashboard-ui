# Development roadmap

Select one useful, bounded improvement at a time. Record tests and measured results.
Avoid changing the fixed synthetic test partition to improve reported metrics.

1. Add batch JSONL inference, line-level validation, and documented error behaviour.
2. Add a real text baseline comparison and stratified cross-validation on training data.
3. Add multi-seed summaries and confidence intervals with explicit sample-size caveats.
4. Measure uncertainty using ensembles; compare review coverage and error rates.
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
