# Contributing

Thanks for helping make `cardinal-harness` reliable and easy to integrate.

## Development setup

- Install Rust (stable) with `rustfmt` and `clippy`.
- Use `cargo test` before opening a PR.

## Local checks

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo doc --no-deps
```

Optional (recommended):

```bash
cargo audit
```

## Expectations

- Keep changes focused and well-scoped.
- Add tests for bug fixes and non-trivial behavior.
- Prefer clear error messages and deterministic behavior.
- Avoid `unsafe` (this project forbids it).

## PRs

- Include a short “What/Why” in the description.
- Call out any API changes explicitly.
- Update docs/README when behavior changes.

