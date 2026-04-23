# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While `0.y.z`, minor versions may contain breaking changes.

## [Unreleased]

## [0.0.1] - 2026-04-23
### Added
- Initial skeleton: pydantic schema / types / config for all four modalities
  (tabular, text, image, graph).
- `LLMProvider` Protocol + `MockLLMProvider`.
- `PromptTemplate` ABC + `EchoPromptTemplate`.
- `Explainer` orchestrator with sync `explain()`.
- ADRs 0001–0004 recording scope, API style, data-model, and counterfactual-payload
  decisions.
