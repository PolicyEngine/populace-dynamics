from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADR_PATH = ROOT / "docs" / "adr" / "0001-populace-axiom-seam-ownership.md"
TECHNICAL_SPECIFICATIONS_PATH = ROOT / "docs" / "technical-specifications.md"
RETIRED_CLAIM = (
    "The benefit calculations themselves (AIME, PIA, spousal benefits, "
    "family maximum) then run through PolicyEngine-US's existing Social "
    "Security implementation."
)


def test_adr_records_seam_ownership():
    assert ADR_PATH.is_file()

    adr = ADR_PATH.read_text(encoding="utf-8")
    for workstream in ("W1", "W2", "W3"):
        assert workstream in adr
    assert "issue #100" in adr


def test_technical_specification_retires_false_claim():
    specification = TECHNICAL_SPECIFICATIONS_PATH.read_text(encoding="utf-8")

    assert RETIRED_CLAIM not in specification
