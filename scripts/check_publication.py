"""Check public files for common credential/private-path patterns and result integrity."""

import hashlib
import ipaddress
import json
import re
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/ami-4090-2026-09-27"
PATTERNS = {
    "credential-shaped value": re.compile(
        r"hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"
        r"|sk-or-v1-[A-Za-z0-9]{32,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    ),
    "machine-specific home path": re.compile(r"/(?:home|Users|root)/[A-Za-z0-9_.-]+"),
    "private domain": re.compile(r"https?://[^\s/]+\.(?:lan|local|internal)(?:/|\b)"),
}


def sensitive_patterns(text, ip_text=None):
    findings = [name for name, pattern in PATTERNS.items() if pattern.search(text)]
    for candidate in re.findall(
        r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", text if ip_text is None else ip_text
    ):
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.is_private:
            findings.append("private IP address")
            break
    return findings


def ip_scan_text(path, text):
    """Package versions can look like addresses; all original text still gets secret checks."""
    if path.name == "environment.json":
        data = json.loads(text)
        versions = data["packages"]
    elif path.parent.name == "packages" and path.name.endswith("-packages.json"):
        data = json.loads(text)
        versions = data
    else:
        return text
    for name, version in versions.items():
        if re.fullmatch(r"\d+(?:\.\d+){3}", version):
            versions[name] = "numeric package version"
    return json.dumps(data)


def check():
    output = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    paths = sorted({ROOT / name.decode() for name in output.split(b"\0") if name})
    findings = []
    for path in paths:
        text = path.read_text()
        findings.extend(
            f"{path.relative_to(ROOT)}: {kind}"
            for kind in sensitive_patterns(text, ip_scan_text(path, text))
        )
        if path.suffix == ".md":
            for link in re.findall(r"\]\(([^)]+)\)", text):
                if not link.startswith(("https://", "http://", "#", "mailto:")):
                    target = (path.parent / link.split("#")[0]).resolve()
                    if not target.exists():
                        findings.append(f"{path.relative_to(ROOT)}: missing local link")
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    for package in lock["package"]:
        registry = package.get("source", {}).get("registry")
        if registry and registry != "https://pypi.org/simple":
            findings.append("lockfile contains a non-public package registry")
        artifacts = ([package["sdist"]] if "sdist" in package else []) + package.get("wheels", [])
        for artifact in artifacts:
            url = urlsplit(artifact["url"])
            if url.scheme != "https" or url.hostname != "files.pythonhosted.org" or url.username:
                findings.append("lockfile contains a non-public artifact URL")
    inventory = json.loads((RESULTS / "checksums.json").read_text())
    actual = {str(p.relative_to(RESULTS)) for p in RESULTS.rglob("*") if p.is_file()}
    if actual != set(inventory) | {"checksums.json"}:
        findings.append("result inventory does not cover exactly the published result files")
    for name, expected in inventory.items():
        path = (RESULTS / name).resolve()
        if not path.is_relative_to(RESULTS.resolve()):
            raise ValueError("Result inventory path escapes its directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            findings.append(f"results: checksum mismatch for {name}")
    summary = json.loads((RESULTS / "comparison/comparison.json").read_text())["summary"]
    readme = (ROOT / "README.md").read_text().replace("**", "")
    names = [
        "MOSS 0.9B",
        "VibeVoice offline",
        "Qwen3 + pyannote",
        "Qwen3 + Nemotron",
        "VibeVoice Streaming",
    ]
    for row, name in zip(summary, names, strict=True):
        der = "—" if row["der"] is None else f"{row['der'] * 100:.2f}%"
        pace = ", paced" if row["model"] == "vibevoice_streaming" else ""
        expected = (
            f"| {name} | {row['cpwer'] * 100:.2f}% | {der} | "
            f"{row['wall_seconds_attempted'] / 60:.2f} min{pace} | "
            f"{row['max_cuda_allocated_bytes'] / 2**30:.2f} GiB | {row['completed']}/25 |"
        )
        if expected not in readme:
            findings.append(f"README table differs from saved results: {row['model']}")
    if findings:
        # Report locations and categories only; never print a suspected credential value.
        raise SystemExit("\n".join(findings))
    print(f"Checked {len(paths)} public files and {len(inventory)} result hashes; no findings.")


if __name__ == "__main__":
    check()
