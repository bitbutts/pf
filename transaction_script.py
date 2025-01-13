import requests
import json
import csv
import os
import base64
import binascii
import datetime
import time

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
OUTPUT_CSV_FILENAME = "xrp_token_payments1.csv"

# How many transactions to fetch per batch
BATCH_LIMIT = 500

# If you hit rate limits, how long to wait (seconds) before retrying
RATE_LIMIT_PAUSE_SECONDS = 30

# Only fetch transactions from the last X days
DAYS_TO_FETCH = 30

# ------------------------------------------------------------------------------
# Helpers
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
        memo_data = decode_hex_or_base64(memo.get("MemoData", ""))   # main content
        # You could also decode MemoType / MemoFormat, if needed
        decoded_memos.append(memo_data)

    # Join multiple memos with a newline (or choose another separator)
    return "\n".join(decoded_memos)

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

def is_token_payment(tx, currency_code, issuer):
    """
    Check if a Payment transaction involves the specified token (IOU).
    """
    if tx.get("TransactionType") != "Payment":
        return False
    amount = tx.get("Amount")
    # For an IOU Payment, 'Amount' is a dict with 'currency', 'issuer', 'value'.
    if isinstance(amount, dict):
        if (amount.get("issuer") == issuer and
            amount.get("currency") == currency_code):
            return True
    return False

# ------------------------------------------------------------------------------
# Fetch from XRPL (with rate-limit handling)
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

    while True:
        try:
            response = requests.post(XRPL_RPC_URL, json=request_body, timeout=20)
            
            # Rate limit handling: If we get 503, wait and retry
            if response.status_code == 503:
                print("[WARN] Rate limit (HTTP 503). Sleeping and retrying...")
                time.sleep(RATE_LIMIT_PAUSE_SECONDS)
                continue
            
            response_json = response.json()
            
            if "result" not in response_json:
                raise Exception(f"Unexpected response: {response_json}")

            result = response_json["result"]
            txs = result.get("transactions", [])
            next_marker = result.get("marker", None)
            return txs, next_marker

        except requests.exceptions.RequestException as e:
            # Network or timeout error, you may also want to retry
            print(f"[ERROR] {e}. Retrying in {RATE_LIMIT_PAUSE_SECONDS} seconds...")
            time.sleep(RATE_LIMIT_PAUSE_SECONDS)
        except json.JSONDecodeError:
            print("[ERROR] Could not decode JSON from server. Retrying...")
            time.sleep(RATE_LIMIT_PAUSE_SECONDS)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    """
    Main routine:
      - Determine a 30-day-ago cutoff time (UTC).
      - Fetch transactions in descending order (newest first).
      - Stop when we reach transactions older than 30 days.
      - Write matching transactions to CSV.
    """

    # Calculate the oldest allowed Unix timestamp
    # so we only process transactions from the last X days.
    now_utc = datetime.datetime.utcnow()
    cutoff_utc = now_utc - datetime.timedelta(days=DAYS_TO_FETCH)
    cutoff_unix = cutoff_utc.timestamp()

    # We'll open the CSV in append mode (or create if it doesn't exist).
    file_exists = os.path.isfile(OUTPUT_CSV_FILENAME)
    csv_file = open(OUTPUT_CSV_FILENAME, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    # If file did not exist, write header
    if not file_exists:
        csv_writer.writerow(["timestamp", "from", "to", "amount", "memo", "ledger_index"])

    print(f"[INFO] Fetching last {DAYS_TO_FETCH} days of transactions for {ACCOUNT_TO_CHECK}")
    print(f"[INFO] Looking for Payment transactions of '{CURRENCY_CODE}' from issuer '{ISSUER_ADDRESS}'")

    # We'll just start from ledger_index_min = -1 (which means earliest),
    # but we will STOP as soon as we see a transaction older than 30 days.
    ledger_index_min = -1
    marker = None

    # Because account_tx returns transactions in descending order (newest first),
    # we can break out of the loop as soon as we hit anything older than the cutoff.
    while True:
        transactions_batch, marker = fetch_account_transactions(
            account=ACCOUNT_TO_CHECK,
            ledger_index_min=ledger_index_min,
            limit=BATCH_LIMIT,
            marker=marker
        )
        
        if not transactions_batch:
            # No more transactions in this range
            break

        stop_fetching = False

        for entry in transactions_batch:
            tx = entry.get("tx", {})
            ledger_index = tx.get("ledger_index", 0)

            # Convert XRPL "date" to Unix epoch
            ripple_time = tx.get("date")  # Ripple epoch
            if ripple_time is None:
                # Some transactions might not have a date (very rare), skip them
                continue
            
            tx_unix_time = ripple_to_unix_time(ripple_time)
            # If transaction is older than our cutoff, we can stop entirely
            if tx_unix_time < cutoff_unix:
                stop_fetching = True
                break

            # Now check if it's a Payment for our specific IOU
            if is_token_payment(tx, CURRENCY_CODE, ISSUER_ADDRESS):
                # Prepare CSV fields
                timestamp_str = format_ripple_timestamp(ripple_time)
                from_addr = tx.get("Account", "")
                to_addr = tx.get("Destination", "")

                amount_field = tx.get("Amount", {})
                if isinstance(amount_field, dict):
                    value_str = amount_field.get("value", "0")
                else:
                    value_str = str(amount_field)  # Could be XRP in drops

                memo_str = extract_memos(tx)

                csv_writer.writerow([
                    timestamp_str,
                    from_addr,
                    to_addr,
                    value_str,
                    memo_str,
                    ledger_index
                ])

        # Important: flush after each batch so we don't lose partial results
        csv_file.flush()

        if stop_fetching:
            # We've encountered older transactions beyond 30 days
            print("[INFO] We've reached transactions older than our 30-day cutoff. Stopping.")
            break

        if not marker:
            # No more pages left
            break

    csv_file.close()
    print("[INFO] Done. Data appended to CSV:", OUTPUT_CSV_FILENAME)


if __name__ == "__main__":
    main()
