import json
import runpy
from pathlib import Path

checks = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_publication.py"))
scan = checks["sensitive_patterns"]


def test_publication_scan_detects_credentials_without_printing_values():
    secret = "hf_" + "a" * 32
    assert scan(secret) == ["credential-shaped value"]
    assert secret not in str(scan(secret))


def test_publication_scan_distinguishes_paths_addresses_and_public_metadata():
    assert scan("/" + "home" + "/example/project") == ["machine-specific home path"]
    assert scan("10" + ".0.0.1") == ["private IP address"]
    assert scan("https://example" + ".internal/model") == ["private domain"]
    assert scan("https://huggingface.co/model BF16 Python 3.12.3 ${BENCHMARK_ROOT}") == []


def test_package_version_is_not_an_address_but_other_fields_still_get_scanned():
    value = "10" + ".4.0.35"
    path = Path("environment.json")
    text = json.dumps({"packages": {"example": value}})
    assert scan(text, checks["ip_scan_text"](path, text)) == []
    text = json.dumps({"packages": {"example": value}, "host": value})
    assert scan(text, checks["ip_scan_text"](path, text)) == ["private IP address"]
