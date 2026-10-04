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
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire", "card_acceptance",
    "card_arrival", "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
    "card_swallowed", "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card", "getting_virtual_card",
    "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
    "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque",
    "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card",
    "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity",
    "verify_my_identity", "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
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
        "train": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{revision}/banking_data/train.csv",
        "test": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{revision}/banking_data/test.csv",
        "categories": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{revision}/banking_data/categories.json",
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
    for i, sample in enumerate(samples):
        digest = hashlib.sha256(sample.text.strip().casefold().encode("utf-8")).hexdigest()
        index.setdefault(digest, []).append(i)
    return {digest: positions for digest, positions in index.items() if len(positions) > 1}

def leakage(train: list[Sample], other: list[Sample]) -> set[str]:
    train_keys = {sample.text.strip().casefold() for sample in train}
    return train_keys.intersection(
        sample.text.strip().casefold() for sample in other
    )

def split_train_validation(
    samples: list[Sample], validation_fraction: float, seed: int
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
