from __future__ import annotations

import json
import os

import pytest
from collect_local import (
    REPO_ROOT,
    CandidateStore,
    collect_path_commands,
    collect_workspace,
    is_safe_output_path,
    non_negative,
    normalize_scoped_term,
    normalize_term,
    parse_cargo_lock,
    parse_gh_extensions,
    parse_manpages,
    parse_npm_lock,
    parse_toolbelt,
    parse_uv_lock,
    validate_output_paths,
    workspace_files,
)


def store() -> CandidateStore:
    return CandidateStore(english={"computer", "keyboard"}, known={"known-term"})


def test_normalize_keeps_technical_identifiers_only() -> None:
    assert normalize_term("io_uring") == "io_uring"
    assert normalize_term("clang++") == "clang++"
    assert normalize_term("check.py") is None
    assert normalize_term("../secret") is None
    assert normalize_term("git status") is None


def test_scoped_npm_names_keep_only_the_bare_package() -> None:
    """A scope names the publisher, so only the package half may become a term."""
    assert normalize_scoped_term("@babel/core") is None
    assert normalize_scoped_term("@scope/obscure-core") == "obscure-core"
    assert normalize_scoped_term("@private/invalid name") is None
    assert normalize_scoped_term("no-scope-here") is None
    candidate = store()
    assert candidate.add("@babel/core", "npm-local") is False
    assert candidate.add("@scope/obscure-core", "npm-local") is True
    assert normalize_term("computer") == "computer"


def test_candidate_store_filters_known_and_plain_english() -> None:
    candidate = store()
    assert candidate.add("io_uring", "manpage") is True
    assert candidate.add("IO_URING", "path-command") is True
    assert candidate.add("computer", "manpage") is False
    assert candidate.add("known-term", "manpage") is False
    row = candidate.ranked()[0]
    assert row["cats"] == ["manpage", "path-command"]
    assert row["priority"] > 0.85


def test_toolbelt_and_manpage_parsers() -> None:
    candidate = store()
    parse_toolbelt(
        "[network]\n  modelctl  /path\n  exq /path\n",
        candidate,
    )
    parse_manpages(
        "io_uring(7) - Linux entry point\n"
        "git(1), git-foo(1) - multi-name manual entry\n"
        "SSL_free(3ssl) - library\nbad page - no section\n",
        candidate,
    )
    terms = {row["term"] for row in candidate.ranked()}
    assert {"modelctl", "exq", "io_uring", "git", "git-foo"}.issubset(terms)
    assert "SSL_free" not in terms


def test_registry_lock_parsers_skip_local_packages() -> None:
    candidate = store()
    cargo = {
        "package": [
            {
                "name": "obscure-crate",
                "source": "registry+https://github.com/rust-lang/crates.io-index",
            },
            {
                "name": "private-registry-crate",
                "source": "registry+https://example.invalid/index",
            },
            {"name": "private-crate"},
        ]
    }
    npm = {
        "packages": {
            "": {"name": "workspace-root"},
            "node_modules/obscure-npm": {
                "version": "1.0.0",
                "resolved": "https://registry.npmjs.org/obscure-npm/-/obscure-npm-1.0.0.tgz",
            },
            "node_modules/private-workspace": {"link": True},
            "node_modules/private-registry-npm": {
                "version": "1.0.0",
                "resolved": "https://example.invalid/obscure-npm.tgz",
            },
            "node_modules/@scope/obscure-scoped": {
                "version": "1.0.0",
                "resolved": "https://registry.npmjs.org/@scope/obscure-scoped/-/obscure-scoped-1.0.0.tgz",
            },
        }
    }
    uv = {
        "package": [
            {
                "name": "obscure-pypi",
                "source": {"registry": "https://pypi.org/simple"},
            },
            {"name": "local-package", "source": "editable+file:///tmp/private"},
            {
                "name": "private-registry-pypi",
                "source": "registry+https://example.invalid/simple",
            },
        ]
    }
    parse_cargo_lock(cargo, candidate)
    parse_npm_lock(npm, candidate)
    parse_uv_lock(uv, candidate)
    terms = {row["term"] for row in candidate.ranked()}
    assert terms == {
        "obscure-crate",
        "obscure-npm",
        "obscure-scoped",
        "obscure-pypi",
    }
    assert not any("@" in term for term in terms)


def test_collect_workspace_parses_cargo_toml_locks(tmp_path) -> None:
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        '[[package]]\nname = "obscure-crate-toml"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n',
        encoding="utf-8",
    )
    candidate = store()

    collect_workspace(tmp_path, candidate)

    assert {row["term"] for row in candidate.ranked()} == {"obscure-crate-toml"}


def test_ranked_rows_are_serializable_and_sorted() -> None:
    candidate = store()
    candidate.add("zephyr-lite", "manpage")
    candidate.add("a11y-tree", "toolbelt")
    rows = candidate.ranked()
    assert [row["term"] for row in rows] == ["a11y-tree", "zephyr-lite"]
    assert isinstance(json.dumps(rows), str)


def test_toolbelt_parser_ignores_prose_lines() -> None:
    """Only catalog entries with a path field are commands, not indented prose."""
    candidate = store()
    parse_toolbelt(
        "[agent-cli]\n"
        "  modelctl  /Users/kearm/.local/bin/modelctl\n"
        "  Personal privacy considerations\n"
        "  this is indented prose that is not a command\n",
        candidate,
    )
    terms = {row["term"] for row in candidate.ranked()}
    assert terms == {"modelctl"}


def test_gh_extension_parser_uses_native_text_rows() -> None:
    candidate = store()
    added = parse_gh_extensions(
        "gh-copilot  github/copilot-extension  GitHub Copilot\n"
        "gh-pilot    example/gh-pilot          Pilot\n",
        candidate,
    )
    assert added == 2
    assert {row["term"] for row in candidate.ranked()} == {"gh-copilot", "gh-pilot"}


def test_output_guard_and_worktree_skipping(tmp_path) -> None:
    assert is_safe_output_path(tmp_path / "candidates.json")
    assert is_safe_output_path(REPO_ROOT / "build" / "candidates.json")
    assert not is_safe_output_path(REPO_ROOT / "tools" / "candidates.json")

    worktree = tmp_path / ".worktrees" / "agent"
    project = tmp_path / "project"
    worktree.mkdir(parents=True)
    project.mkdir()
    (worktree / "Cargo.lock").write_text("{}", encoding="utf-8")
    (project / "Cargo.lock").write_text("{}", encoding="utf-8")
    assert [path.parent.name for path in workspace_files(tmp_path)] == ["project"]


def test_output_paths_must_be_distinct(tmp_path) -> None:
    first = tmp_path / "report"
    second = tmp_path / "report.manifest.json"
    validate_output_paths([first, tmp_path / "terms.json"])
    with pytest.raises(ValueError, match="distinct"):
        validate_output_paths(
            [
                first,
                second,
                tmp_path / "lexicon.json",
                first.with_suffix(".manifest.json"),
            ]
        )


def test_non_negative_count_arguments() -> None:
    assert non_negative("0") == 0
    assert non_negative("12") == 12
    with pytest.raises(Exception, match=">= 0"):
        non_negative("-1")


def test_empty_path_components_do_not_scan_the_working_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_tool = tmp_path / "private-local-tool"
    local_tool.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    local_tool.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f":{os.pathsep}")

    candidate = store()
    collect_path_commands(candidate)

    assert candidate.ranked() == []


def test_public_source_constants_are_exact_registries() -> None:
    """Registry gating must be exact, not a prefix or truthiness check."""
    from collect_local import (
        PUBLIC_CARGO_SOURCE,
        PUBLIC_NPM_SOURCE,
        PUBLIC_PYPI_SOURCES,
    )

    assert PUBLIC_CARGO_SOURCE.startswith("registry+https://github.com/rust-lang/")
    assert PUBLIC_NPM_SOURCE == "https://registry.npmjs.org/"
    assert all(
        source in {"https://pypi.org/simple", "https://pypi.org/simple/"}
        for source in PUBLIC_PYPI_SOURCES
    )
