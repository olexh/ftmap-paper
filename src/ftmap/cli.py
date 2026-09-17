"""Command line entry point."""

from __future__ import annotations

import glob
import json

import typer

import os

from ftmap import __version__
from ftmap.config import Config
from ftmap.inventory import inventory as do_inventory
from ftmap.pipeline import run_corpus
from ftmap.plan.client import LlamaClient
from ftmap.vocab.catalogue import Catalogue

app = typer.Typer(add_completion=False, help=__doc__)


def _resolve(config: str | None) -> str | None:
    """An explicit --config wins; otherwise ./ftmap.toml if the run directory
    has one; otherwise the packaged defaults. Config.load stays explicit so
    tests never depend on the working directory."""
    if config:
        return config
    return "ftmap.toml" if os.path.exists("ftmap.toml") else None


@app.command()
def version() -> None:
    """Print the version and the resolved configuration."""
    cfg = Config.load(_resolve(None))
    typer.echo(f"ftmap {__version__}")
    typer.echo(f"model: {cfg.model_url} thinking={cfg.enable_thinking}")


@app.command()
def inventory(path: str, config: str = typer.Option(None, "--config")) -> None:
    """List the sources under PATH, one JSON object per line."""
    Config.load(_resolve(config))
    for ref in do_inventory(path):
        typer.echo(json.dumps(ref.to_dict(), ensure_ascii=False))


@app.command()
def run(
    path: str,
    config: str = typer.Option(None, "--config"),
    out: str = typer.Option("work", "--out"),
    limit: int = typer.Option(None, "--limit"),
    only: list[str] = typer.Option(
        None, "--only",
        help="Run only the sources whose path, sheet or source id contains "
             "this text; repeatable."),
) -> None:
    """Read PATH, plan each source with the local model, write FtM entities."""
    cfg = Config.load(_resolve(config))
    client = LlamaClient(cfg, cache_path=os.path.join(out, "prompts.jsonl"))
    report = run_corpus(path, cfg, Catalogue.load(), client, out, limit=limit,
                        only=list(only or ()))
    typer.echo(json.dumps({k: v for k, v in report.items()
                           if k not in ("summaries", "failures",
                                        "declined_sources")},
                          ensure_ascii=False, indent=2))
    if report["failed"]:
        typer.echo(f"{report['failed']} sources failed; see failures.jsonl in "
                   "the aggregate generation this run committed")
    if report["declined"]:
        typer.echo(f"{report['declined']} sources produced no entity and were "
                   "declined; each says why in its own summary.json")
    if report["incomplete"]:
        typer.echo("this run is INCOMPLETE — it was cancelled or lost a "
                   "source — and its output is not publication-grade")


@app.command()
def upgrade(source: str, dest: str) -> None:
    """Copy a retained workspace into a new generation-aware one."""
    from ftmap.workspace import upgrade as do_upgrade

    workspace = do_upgrade(source, dest)
    typer.echo(json.dumps({"mode": workspace.mode, "root": workspace.root,
                           "sources": len(workspace.source_ids())},
                          ensure_ascii=False, indent=2))


@app.command()
def publish(out: str = typer.Option("work", "--out"),
            config: str = typer.Option(None, "--config")) -> None:
    """Rebuild the corpus exports from the current committed source
    generations.
    """
    from ftmap.pipeline import PublishRefused, publish_corpus
    from ftmap.workspace import open_workspace

    cfg = Config.load(_resolve(config))
    try:
        report = publish_corpus(out, cfg, Catalogue.load())
    except PublishRefused as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    workspace = open_workspace(out)
    typer.echo(json.dumps({
        "sources": report["sources"],
        "statements": report["statements"],
        "stale": workspace.stale_sources(),
        "publication_grade": workspace.publication_grade(),
    }, ensure_ascii=False, indent=2))


@app.command()
def retention(out: str = typer.Option("work", "--out"),
              apply: bool = typer.Option(False, "--apply")) -> None:
    """Report what could be reclaimed, and with --apply, reclaim it."""
    from ftmap.workspace import Retention, open_workspace

    keeper = Retention(open_workspace(out))
    plan = keeper.prune() if apply else keeper.reclaimable()
    typer.echo(json.dumps({
        "applied": apply,
        "generations": len(plan["generations"]),
        "snapshots": len(plan["snapshots"]),
        "abandoned_staging": len(plan["staging"]),
        "derived_indexes": len(plan["derived"]),
        "megabytes": round(plan["bytes"] / (1 << 20), 1),
    }, ensure_ascii=False, indent=2))


@app.command()
def etalon(path: str,
           out: str = typer.Option("work", "--out"),
           as_json: bool = typer.Option(False, "--json")) -> None:
    """Score a run against an etalon file, or a directory of them."""
    from ftmap.etalon import load
    from ftmap.etalon.score import as_markdown, find_summary, score as do_score

    files = ([path] if os.path.isfile(path)
             else sorted(glob.glob(os.path.join(path, "*.etalon.yaml"))))
    if not files:
        raise typer.BadParameter(f"no *.etalon.yaml under {path}")
    cat = Catalogue.load()
    results, unscored = [], []
    for f in files:
        doc = load(f, cat)
        try:
            results.append(do_score(doc, find_summary(doc, out), cat))
        except FileNotFoundError:
            unscored.append(os.path.basename(f).replace(".etalon.yaml", ""))
    if unscored:
        typer.echo(f"# {len(unscored)} etalon(s) have no source in {out}: "
                   f"{', '.join(unscored)}", err=True)
    broken = [r.source for r in results if r.accounting_valid is False]
    legacy = [r.source for r in results if r.accounting_valid is None]
    if broken:
        typer.echo(f"# LEDGER INVALID on {len(broken)} source(s): "
                   f"{', '.join(broken)} — these figures are NOT eligible "
                   f"for publication.", err=True)
    if legacy:
        typer.echo(f"# LEDGER UNVOUCHED on {len(legacy)} source(s): "
                   f"{', '.join(legacy)} — version-1 summaries that predate "
                   f"the accounting gate; not publication-eligible without a "
                   f"schema-2 rerun.", err=True)
    if as_json:
        typer.echo(json.dumps({r.source: r.totals() for r in results},
                              ensure_ascii=False, indent=2))
    else:
        typer.echo("\n\n".join(as_markdown(r) for r in results))
    if broken:
        raise typer.Exit(code=3)


if __name__ == "__main__":
    app()
