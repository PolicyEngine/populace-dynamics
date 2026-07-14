"""Typed identity helpers for the versioned gate contract."""

from __future__ import annotations

import importlib.util
import platform as platform_module
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

__all__ = ["ContractRef", "contract_revision", "environment_block"]

#: The fitting stack the M6/W1 refit runs against, as (distribution, import)
#: pairs.  It lives in a dedicated env (populace-fit pins scikit-learn <1.9 and
#: cannot coexist with the base venv); the base environment block never recorded
#: its vintage, leaving a scored run's frame-stack provenance unrecoverable from
#: the sidecar (grading #42 comment 4972045579).
_FITTING_STACK_PACKAGES: tuple[tuple[str, str], ...] = (
    ("populace-fit", "populace.fit"),
    ("populace-frame", "populace.frame"),
)

_REPOSITORY_HINT = Path(__file__).resolve().parents[2]
_CONTRACT_PATH = Path("gates.yaml")


def _git(start: Path, *args: str) -> str:
    """Run Git from a path inside the checkout and return stripped stdout."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        command = " ".join(("git", *args))
        raise RuntimeError(
            f"Unable to resolve contract identity with `{command}`"
        ) from error


@dataclass(frozen=True)
class ContractRef:
    """A contract blob and the checkout commit that embeds it.

    Attributes:
        blob_sha: Git blob identity of the committed contract bytes.
        head_sha: Git commit checked out when the reference was captured.
        path: Portable repository-relative path to the contract.
    """

    blob_sha: str
    head_sha: str
    path: str

    @classmethod
    def current(
        cls,
        root: str | Path | None = None,
        path: str | Path = _CONTRACT_PATH,
    ) -> ContractRef:
        """Capture the contract reference for the current Git checkout.

        Args:
            root: A directory inside the checkout. The package checkout is
                used by default.
            path: Contract path, relative to the repository root or absolute.

        Returns:
            The committed contract blob, checkout HEAD, and relative path.
        """
        start = Path(root).resolve() if root is not None else _REPOSITORY_HINT
        repository = Path(_git(start, "rev-parse", "--show-toplevel"))
        contract_path = Path(path)
        if contract_path.is_absolute():
            try:
                relative_path = contract_path.resolve().relative_to(repository)
            except ValueError as error:
                raise ValueError(
                    f"Contract path {contract_path} is outside {repository}"
                ) from error
        else:
            relative_path = contract_path

        portable_path = relative_path.as_posix()
        revisions = _git(
            repository,
            "rev-parse",
            "HEAD",
            f"HEAD:{portable_path}",
        ).splitlines()
        if len(revisions) != 2:
            raise RuntimeError("Git returned an invalid contract reference")
        head_sha, blob_sha = revisions
        working_blob_sha = _git(
            repository,
            "hash-object",
            "--",
            portable_path,
        )
        if working_blob_sha != blob_sha:
            raise RuntimeError(
                f"{portable_path} differs from HEAD; refusing to record "
                "an ambiguous contract reference"
            )
        return cls(
            blob_sha=blob_sha,
            head_sha=head_sha,
            path=portable_path,
        )


def contract_revision(root: str | Path | None = None) -> str:
    """Return the Git blob identity of the committed ``gates.yaml``."""
    return ContractRef.current(root).blob_sha


def _package_git_revision(import_name: str) -> str:
    """Return the git HEAD of an installed package's source tree, or ``unknown``.

    An editable install (the fitting stack's dedicated env) resolves to a git
    checkout, so its revision is recoverable; a wheel install resolves to a
    non-git directory and degrades to ``unknown``.
    """
    try:
        spec = importlib.util.find_spec(import_name)
    except (ImportError, ValueError):
        return "unknown"
    if spec is None:
        return "unknown"
    locations: list[Path] = []
    if spec.origin and spec.origin not in ("built-in", "frozen"):
        locations.append(Path(spec.origin).parent)
    for entry in spec.submodule_search_locations or ():
        locations.append(Path(entry))
    for location in locations:
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=location,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if revision:
            return revision
    return "unknown"


def _fitting_stack_block() -> dict[str, object]:
    """Fitting-stack provenance: version + git rev when importable, else absent.

    Each package is recorded as ``{"version", "git_rev"}`` when installed and
    ``"absent"`` otherwise, so the sidecar carries the frame-stack vintage a
    scored M6/W1 run was fit against.
    """
    block: dict[str, object] = {}
    for dist_name, import_name in _FITTING_STACK_PACKAGES:
        key = dist_name.replace("-", "_")
        try:
            dist_version = version(dist_name)
        except PackageNotFoundError:
            block[key] = "absent"
            continue
        block[key] = {
            "version": dist_version,
            "git_rev": _package_git_revision(import_name),
        }
    return block


def environment_block() -> dict[str, object]:
    """Return runtime versions required to reproduce an artifact.

    Alongside the interpreter and core-library versions, this records the
    fitting-stack provenance (populace-fit / populace-frame) the base block
    previously omitted.
    """
    return {
        "python": platform_module.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "sklearn": version("scikit-learn"),
        "scipy": version("scipy"),
        "platform": platform_module.platform(),
        "fitting_stack": _fitting_stack_block(),
    }
