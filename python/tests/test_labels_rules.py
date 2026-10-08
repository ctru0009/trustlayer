"""Unit tests for trustlayer.labels.rules (pure Python, no Spark)."""

from trustlayer.labels.rules import Label, weak_label


def test_plain_text_is_public() -> None:
    weak = weak_label("Lunch plans", "Want to grab sandwiches at noon?")
    assert weak.label is Label.PUBLIC
    assert weak.pii == ()
    assert weak.rules == ()


def test_email_address_is_internal() -> None:
    weak = weak_label("Contact", "Reach me at jane.doe@example.com please.")
    assert weak.label is Label.INTERNAL
    assert weak.pii == ("email",)
    assert weak.rules == ("email-address",)


def test_phone_number_is_internal() -> None:
    weak = weak_label("Call me", "My number is 415-555-0132 after 5pm.")
    assert weak.label is Label.INTERNAL
    assert "phone" in weak.pii


def test_ssn_is_confidential() -> None:
    weak = weak_label("Form", "SSN on file: 123-45-6789 for payroll.")
    assert weak.label is Label.CONFIDENTIAL
    assert "ssn" in weak.pii


def test_luhn_card_is_confidential() -> None:
    # 4111 1111 1111 1111 is the standard Luhn-valid test number.
    weak = weak_label("Payment", "Charge 4111 1111 1111 1111 exp 01/30.")
    assert weak.label is Label.CONFIDENTIAL
    assert "card" in weak.pii


def test_non_luhn_digits_are_not_card() -> None:
    weak = weak_label("Order", "Reference 4111 1111 1111 1112 for tracking.")
    assert "card" not in weak.pii
    assert weak.label is Label.PUBLIC


def test_password_assignment_is_confidential() -> None:
    weak = weak_label("Access", "Temporary password: hunter2-change-me!")
    assert weak.label is Label.CONFIDENTIAL
    assert "credentials" in weak.pii


def test_confidential_marker_without_pii() -> None:
    weak = weak_label("Privileged", "Attorney review attached. Do not forward.")
    assert weak.label is Label.CONFIDENTIAL
    assert "confidential-marker" in weak.rules


def test_business_keywords_are_internal() -> None:
    weak = weak_label("Q3 forecast", "The budget review meeting is Thursday.")
    assert weak.label is Label.INTERNAL
    assert "internal-keyword" in weak.rules


def test_subject_line_also_matches() -> None:
    weak = weak_label("Salary bands for 2025", "See attached spreadsheet.")
    assert weak.label is Label.INTERNAL
