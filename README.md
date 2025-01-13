XRP Last-30-Days IOU Payment Fetcher for PFT
This script pulls Payment transactions from the XRP Ledger related to PFT within the last 30 days, writing the results to a CSV file. It handles rate limiting (HTTP 503) by retrying requests, so you don’t lose progress. It also flushes partial results to disk after each batch, in case the script is interrupted.

Features
Fetches last 30 days of transactions for a specified account on the XRP Ledger.
Filters for Payment transactions of PFT (currency code + issuer).
Handles rate limiting (HTTP 503) by pausing and retrying automatically.
Partial saves: Writes each batch to CSV immediately, so you don’t lose data on interruption.

Requirements
Python 3.7+
Requests (for HTTP requests)
Install requests with:

bash
Copy code
pip install requests
Installation
Clone or download this repository (or save the script as a .py file).
(Optional) Create a virtual environment to isolate dependencies:
bash
Copy code
python -m venv venv
source venv/bin/activate     # (macOS/Linux)

venv\Scripts\activate        # (Windows)
Install dependencies:
bash
Copy code
pip install requests
Configuration
At the top of the script, you can modify the following constants:

XRPL_RPC_URL
The XRPL node to connect to. Default is "https://s1.ripple.com:51234/". You may switch to another public node if desired.

CURRENCY_CODE
The currency code of your IOU (e.g., "USD", "EUR", "FOO"). In this case "PFT"

ISSUER_ADDRESS
The XRPL address issuing that IOU (e.g., "rEXAMPLEISSUERADDRESS1234"). In this case "rnQUEEg8yyjrwk9FhyXpKavHyCRJM9BDMW".

ACCOUNT_TO_CHECK
The account whose transactions you want to fetch. You can use the same address as ISSUER_ADDRESS or a different address that holds the token.

OUTPUT_CSV_FILENAME
The CSV file to which results will be appended (or created if it doesn’t exist). Default is "xrp_token_payments.csv".

BATCH_LIMIT
How many transactions to request per batch. Default is 50. If you need fewer requests (and the node supports it), you can raise this to 200, 500, etc.

RATE_LIMIT_PAUSE_SECONDS
How many seconds to wait before retrying after a 503 "Server is overloaded" response. Default is 30.

DAYS_TO_FETCH
How many days in the past to retrieve transactions for. Default is 30.

Running the Script
Configure the script variables mentioned above.
Run the script:
bash
Copy code
python fetch_last_30_days.py
(where fetch_last_30_days.py is the filename of your script).
Expected Output
A CSV file named xrp_token_payments.csv (by default).
Each row includes:
timestamp (UTC date/time for when the ledger closed)
from (the sending address)
to (the receiving address)
amount (the numeric value of the IOU transferred)
memo (concatenated memos, if any)
ledger_index (the ledger index in which the transaction was included)
You’ll see terminal output with progress logs, warnings about rate-limiting if it occurs, and a final message stating that data has been written to the CSV file.

How It Works
Time-Based Cutoff
The script calculates now - DAYS_TO_FETCH in UTC, then converts each transaction’s Ripple epoch time to Unix time. If a transaction is older than that cutoff, the script stops fetching further.

Batch Fetching & Pagination

Uses the account_tx method on the XRPL node.
Paginates with the marker returned by the node, continuing to fetch in descending order (newest to oldest) until either the cutoff or no more transactions remain.
Rate-Limit Handling

If an HTTP 503 Service Unavailable is returned, the script sleeps (RATE_LIMIT_PAUSE_SECONDS) and retries the same request.
This ensures long-running data pulls succeed even under throttling.
Partial Writes

After each batch of transactions is processed, the script calls csv_file.flush(), so you won’t lose any progress if the script is interrupted mid-run.
Troubleshooting & Tips
Rate Limits
If you hit rate limits often, try:

Increasing RATE_LIMIT_PAUSE_SECONDS
Decreasing BATCH_LIMIT
Switching to a different XRPL node that might be less busy
Not Enough History
Public nodes have a rolling history window. If you cannot retrieve older data from s1.ripple.com, try a different node or consider running your own rippled node with more historical ledgers.

Incorrect Time
Double-check your system clock if your 30-day filter yields unexpected results. The script uses your machine’s UTC time to determine the cutoff.

Memo Decoding
The script attempts to decode memos as hex first, then base64. If you need advanced memo handling or decoding MemoType / MemoFormat, update extract_memos().

