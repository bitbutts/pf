import requests
import json
import csv
import os
import base64
import binascii
import datetime

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

XRPL_RPC_URL = "https://s1.ripple.com:51234/"

# Your token (IOU) details
CURRENCY_CODE = "PFT"    # e.g. "USD", "FOO", etc.
ISSUER_ADDRESS = "rnQUEEg8yyjrwk9FhyXpKavHyCRJM9BDMW"  # Replace with the actual issuer

# Which account to check transactions for:
ACCOUNT_TO_CHECK = ISSUER_ADDRESS

# Output CSV filename
OUTPUT_CSV_FILENAME = "xrp_token_payments.csv"

# “State” file to remember the last processed ledger index
LAST_LEDGER_INDEX_FILE = "last_ledger_index.txt"

# ------------------------------------------------------------------------------
# Helpers to fetch and filter transactions
# ------------------------------------------------------------------------------

def fetch_account_transactions(account, ledger_index_min, limit=50, marker=None):
    """
    Fetch transactions for a given account using the 'account_tx' method
    from the XRPL JSON RPC. Returns (transactions, marker).

    - ledger_index_min: the minimum ledger index to start from
    - marker: for pagination
    """
    request_body = {
        "method": "account_tx",
        "params": [
            {
                "account": account,
                "ledger_index_min": ledger_index_min,
                "ledger_index_max": -1,  # no upper limit
                "limit": limit
            }
        ]
    }
    if marker:
        request_body["params"][0]["marker"] = marker

    response = requests.post(XRPL_RPC_URL, json=request_body, timeout=20)
    response_json = response.json()

    if "result" not in response_json:
        raise Exception(f"Unexpected response: {response_json}")

    result = response_json["result"]
    txs = result.get("transactions", [])
    next_marker = result.get("marker", None)

    return txs, next_marker

def is_token_payment(tx, currency_code, issuer):
    """
    Check if a Payment transaction involves the specified token (IOU).
    """
    if tx.get("TransactionType") != "Payment":
        return False
    amount = tx.get("Amount")
    # For an IOU Payment, 'Amount' is a dictionary with 'currency', 'issuer', 'value'.
    if isinstance(amount, dict):
        if (amount.get("issuer") == issuer and
            amount.get("currency") == currency_code):
            return True
    return False

# ------------------------------------------------------------------------------
# Memo decoding
# ------------------------------------------------------------------------------

def decode_hex_or_base64(encoded_str):
    """
    Attempts to decode a string from hex or base64.
    Returns decoded ASCII (if possible) or the raw string if decoding fails.
    """
    if not encoded_str:
        return ""

    # Try hex decode first
    try:
        decoded_bytes = bytes.fromhex(encoded_str)
        return decoded_bytes.decode('utf-8', errors='replace')
    except ValueError:
        pass

    # If that fails, try base64 decode
    try:
        decoded_bytes = base64.b64decode(encoded_str)
        return decoded_bytes.decode('utf-8', errors='replace')
    except (binascii.Error, ValueError):
        # If all decoding fails, return the original encoded string
        return encoded_str

def extract_memos(transaction):
    """
    Extract and decode all memos from a transaction, concatenating them into a single string.
    """
    memos = transaction.get("Memos", [])
    if not memos:
        return ""

    decoded_memos = []
    for memo_entry in memos:
        memo = memo_entry.get("Memo", {})
        memo_type   = decode_hex_or_base64(memo.get("MemoType", ""))   # optional
        memo_format = decode_hex_or_base64(memo.get("MemoFormat", "")) # optional
        memo_data   = decode_hex_or_base64(memo.get("MemoData", ""))   # main content
        # Combine them in some sensible way. Here, we'll just store the data.
        # You could also include type/format if desired.
        combined = memo_data
        decoded_memos.append(combined)

    # Join multiple memos with a separator (e.g. newline or semicolon)
    return "\n".join(decoded_memos)

# ------------------------------------------------------------------------------
# Date/Time Handling
# ------------------------------------------------------------------------------

def ripple_to_unix_time(ripple_time):
    """
    Convert Ripple epoch time (seconds since 1/1/2000) to Unix epoch time (seconds since 1/1/1970).
    Ripple epoch starts at 2000-01-01 00:00:00 UTC,
    which is 946684800 seconds after Unix epoch start (1970-01-01).
    """
    return ripple_time + 946684800

def format_ripple_timestamp(ripple_time):
    """
    Given a Ripple ledger time (int), return a human-readable UTC datetime string.
    """
    unix_timestamp = ripple_to_unix_time(ripple_time)
    return datetime.datetime.utcfromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S')

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    # 1) Read last processed ledger index (if file exists)
    if os.path.exists(LAST_LEDGER_INDEX_FILE):
        with open(LAST_LEDGER_INDEX_FILE, "r") as f:
            last_ledger_index = int(f.read().strip())
        ledger_index_min = last_ledger_index + 1
        print(f"Resuming from ledger_index > {last_ledger_index}")
    else:
        ledger_index_min = -1  # means no lower bound
        print("No last ledger index file found. Fetching all ledgers (this might be large).")

    # 2) Prepare CSV writer (append mode). If file doesn’t exist, write header.
    file_exists = os.path.isfile(OUTPUT_CSV_FILENAME)
    csv_file = open(OUTPUT_CSV_FILENAME, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    if not file_exists:
        csv_writer.writerow(["timestamp", "from", "to", "amount", "memo", "ledger_index"])

    # 3) Fetch & filter transactions
    print(f"Fetching transactions for account: {ACCOUNT_TO_CHECK}")
    print(f"Filtering for Payment transactions of token '{CURRENCY_CODE}' from issuer {ISSUER_ADDRESS}\n")

    marker = None
    highest_ledger_this_run = ledger_index_min

    while True:
        # Fetch one batch
        transactions_batch, marker = fetch_account_transactions(
            account=ACCOUNT_TO_CHECK,
            ledger_index_min=ledger_index_min,
            marker=marker
        )

        for entry in transactions_batch:
            tx = entry.get("tx", {})
            meta = entry.get("meta", {})
            ledger_index = tx.get("ledger_index", 0)

            # Track highest ledger index we see, so we can save it later
            if ledger_index > highest_ledger_this_run:
                highest_ledger_this_run = ledger_index

            # Only handle Payment transactions that match our token
            if is_token_payment(tx, CURRENCY_CODE, ISSUER_ADDRESS):
                # Extract data
                from_addr = tx.get("Account", "")
                to_addr = tx.get("Destination", "")
                amount = tx.get("Amount", {})
                # If it's an IOU, amount is a dict: {currency, issuer, value}
                # For a Payment in IOU, "value" is the actual numeric amount
                if isinstance(amount, dict):
                    value_str = amount.get("value", "0")
                else:
                    # If it were XRP, it's in drops as a string.
                    # But theoretically, we won't be here if it's not our IOU.
                    value_str = amount

                # Convert the Ripple "date" (ledger close time) to a human-readable string
                # "date" is the ledger time in seconds since 1/1/2000
                ripple_time = tx.get("date")
                if ripple_time is not None:
                    timestamp_str = format_ripple_timestamp(ripple_time)
                else:
                    timestamp_str = ""

                # Get memos
                memo_str = extract_memos(tx)

                # Write a row to CSV
                csv_writer.writerow([
                    timestamp_str,
                    from_addr,
                    to_addr,
                    value_str,
                    memo_str,
                    ledger_index
                ])

        # Pagination check
        if not marker:
            # No more pages
            break

    csv_file.close()

    # 4) Store the new highest ledger index so next run starts after this
    if highest_ledger_this_run > 0:
        with open(LAST_LEDGER_INDEX_FILE, "w") as f:
            f.write(str(highest_ledger_this_run))

    print("Done.")
    print(f"Last processed ledger index: {highest_ledger_this_run}")
    print(f"Data appended to CSV: {OUTPUT_CSV_FILENAME}")

if __name__ == "__main__":
    main()
