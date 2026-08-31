# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional project polish: LICENSE, Dockerfile, CHANGELOG, CI status badges.

## [0.3.0] - 2026-08-31

### Added
- **Parallel per-MR loops**: each of the `k` latest merged PRs now runs its own
  deep-dive → review → refine loop concurrently (thread pool), followed by a
  **final combine loop** that merges the k polished sections into the digest.
- Per-agent **model selection in a loop**: each sub-agent (MR worker, combine,
  reviewer, refine) selects a working model at call time rather than sharing
  one hardcoded model.

### Changed
- Default review budget raised to `max_reviews=6` and quality bar to
  `min_quality_score=8.0` (quality over time).

## [0.2.0] - 2026-08-31

### Added
- Orchestrator-Workers: per-PR deep-dive sub-agents with a final "chair" reviewer.
- Report identity: every run stamped with a UID, version, and UTC timestamp.
- Color-coded PDF diffs (green added / red removed / purple hunks).

## [0.1.0] - 2026-08-31

### Added
- Initial Hexagonal (Ports & Adapters) architecture.
- Multi-provider LLM fallback: Gemini → OpenRouter → Groq → Mistral (all free).
- Deterministic "Executive Snapshot" prepended above the AI summary.
- Review committee (3 specialist reviewers) with a bounded refine loop.
- Professional PDF rendering with cover band and page-number footers.
- SMTP (Gmail/Ethereal) and Resend notifiers.
- GitHub Actions daily scheduling.