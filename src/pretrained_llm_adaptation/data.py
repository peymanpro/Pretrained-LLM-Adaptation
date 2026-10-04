from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from .seed import set_seed

INTENTS = (
    "card_arrival",
    "card_linking",
    "exchange_rate",
    "card_payment_wrong_exchange_rate",
    "extra_charge_on_statement",
    "pending_cash_withdrawal",
    "fiat_currency_support",
    "card_delivery_estimate",
    "automatic_top_up",
    "card_not_working",
    "exchange_via_app",
    "lost_or_stolen_card",
    "age_limit",
    "pin_blocked",
    "contactless_not_working",
    "top_up_by_bank_transfer_charge",
    "pending_top_up",
    "cancel_transfer",
    "top_up_limits",
    "wrong_amount_of_cash_received",
    "card_payment_fee_charged",
    "transfer_not_received_by_recipient",
    "supported_cards_and_currencies",
    "getting_virtual_card",
    "card_acceptance",
    "top_up_reverted",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "card_payment_not_recognised",
    "edit_personal_details",
    "why_verify_identity",
    "unable_to_verify_identity",
    "get_physical_card",
    "visa_or_mastercard",
    "topping_up_by_card",
    "disposable_card_limits",
    "compromised_card",
    "atm_support",
    "direct_debit_payment_not_recognised",
    "passcode_forgotten",
    "declined_cash_withdrawal",
    "pending_card_payment",
    "lost_or_stolen_phone",
    "request_refund",
    "declined_transfer",
    "Refund_not_showing_up",
    "declined_card_payment",
    "pending_transfer",
    "terminate_account",
    "card_swallowed",
    "transaction_charged_twice",
    "verify_source_of_funds",
    "transfer_timing",
    "reverted_card_payment?",
    "change_pin",
    "beneficiary_not_allowed",
    "transfer_fee_charged",
    "receiving_money",
    "failed_transfer",
    "transfer_into_account",
    "verify_top_up",
    "getting_spare_card",
    "top_up_by_cash_or_cheque",
    "order_physical_card",
    "virtual_card_not_working",
    "wrong_exchange_rate_for_cash_withdrawal",
    "get_disposable_virtual_card",
    "top_up_failed",
    "balance_not_updated_after_bank_transfer",
    "cash_withdrawal_not_recognised",
    "exchange_charge",
    "top_up_by_card_charge",
    "activate_my_card",
    "cash_withdrawal_charge",
    "card_about_to_expire",
    "apple_pay_or_google_pay",
    "verify_my_identity",
    "country_support",
)


@dataclass(frozen=True)
class Sample:
    text: str
    label: str


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def download_banking77(raw_dir: str | Path, revision: str) -> dict[str, str]:
    raw = Path(raw_dir)
    urls = {
        "train": (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{revision}/banking_data/train.csv"
        ),
        "test": (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{revision}/banking_data/test.csv"
        ),
        "categories": (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{revision}/banking_data/categories.json"
        ),
    }
    paths = {
        "train": raw / "train.csv",
        "test": raw / "test.csv",
        "categories": raw / "categories.json",
    }
    for key, path in paths.items():
        if not path.exists():
            _download(urls[key], path)
    return {key: str(path) for key, path in paths.items()}


def read_csv(path: str | Path) -> list[Sample]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0].keys()) != {"text", "category"}:
        raise ValueError(f"Unexpected schema in {path}; expected text,category")
    return [Sample(text=str(row["text"]), label=str(row["category"])) for row in rows]


def read_categories(path: str | Path) -> tuple[str, ...]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"Unexpected categories schema in {path}")
    return tuple(values)


def validate_categories(categories: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    if len(categories) != len(INTENTS):
        errors.append(f"expected {len(INTENTS)} categories, found {len(categories)}")
    if categories != INTENTS:
        errors.append("categories.json does not match the pinned BANKING77 intent order")
    return errors


def validate_samples(samples: list[Sample]) -> list[str]:
    errors: list[str] = []
    for index, sample in enumerate(samples):
        if not sample.text.strip():
            errors.append(f"row {index}: empty text")
        if sample.label not in INTENTS:
            errors.append(f"row {index}: unknown label {sample.label!r}")
    return errors


def duplicate_texts(samples: list[Sample]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for sample_index, sample in enumerate(samples):
        digest = hashlib.sha256(sample.text.strip().casefold().encode("utf-8")).hexdigest()
        index.setdefault(digest, []).append(sample_index)
    return {digest: positions for digest, positions in index.items() if len(positions) > 1}


def leakage(train: list[Sample], other: list[Sample]) -> set[str]:
    train_keys = {sample.text.strip().casefold() for sample in train}
    return train_keys.intersection(sample.text.strip().casefold() for sample in other)


def split_train_validation(
    samples: list[Sample],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    set_seed(seed)
    labels = [sample.label for sample in samples]
    train, validation = train_test_split(
        samples,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels,
    )
    return list(train), list(validation)


def write_manifest(
    destination: str | Path,
    source_revision: str,
    samples: dict[str, list[Sample]],
    validation_fraction: float,
    seed: int,
) -> None:
    payload = {
        "dataset": "PolyAI/banking77",
        "source_revision": source_revision,
        "counts": {split: len(items) for split, items in samples.items()},
        "label_counts": {
            split: {
                label: sum(sample.label == label for sample in items)
                for label in INTENTS
            }
            for split, items in samples.items()
        },
        "validation_fraction": validation_fraction,
        "seed": seed,
        "labels": list(INTENTS),
    }
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
