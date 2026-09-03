from src.ingestion.transform.text_spacing_repair import (
    TextSpacingRepairConfig,
    TextSpacingRepairer,
)


def test_repairs_known_pdf_missing_space_runs():
    repairer = TextSpacingRepairer(TextSpacingRepairConfig(enabled=True))

    result = repairer.repair(
        "Thisimpliesthatsystemoutputshouldbetailoredtodifferenttypesofusers."
    )

    assert result.changed is True
    assert (
        result.text
        == "This implies that system output should be tailored to different types of users."
    )
    assert result.repairs[0]["reason"] == "accepted"


def test_repairs_lowercase_explainability_phrase():
    repairer = TextSpacingRepairer(TextSpacingRepairConfig(enabled=True))

    result = repairer.repair(
        "meaningfulinterpretationsfrompreciseexplanationsofmodeloutput"
    )

    assert result.text == "meaningful interpretations from precise explanations of model output"


def test_skips_identifier_like_runs():
    repairer = TextSpacingRepairer(TextSpacingRepairConfig(enabled=True))

    result = repairer.repair("abc123systemoutputshouldbetailoredtodifferenttypesofusers")

    assert result.changed is False
    assert result.text == "abc123systemoutputshouldbetailoredtodifferenttypesofusers"


def test_disabled_repair_is_noop():
    repairer = TextSpacingRepairer(TextSpacingRepairConfig(enabled=False))
    text = "systemoutputshouldbetailoredtodifferenttypesofusers"

    result = repairer.repair(text)

    assert result.changed is False
    assert result.text == text
    assert result.repairs == []
    assert result.skipped == []
