import json
import os
import statistics
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_blocks() -> List[Dict[str, Any]]:
    """Load all block files from the blocks directory."""
    start_time = time.time()
    blocks_dir = Path("blocks")
    if not blocks_dir.exists():
        raise FileNotFoundError(f"Blocks directory not found at {blocks_dir}")

    blocks = []
    total_size = 0
    for block_file in sorted(blocks_dir.glob("*.json")):
        try:
            file_size = os.path.getsize(block_file)
            total_size += file_size
            with open(block_file, "r") as f:
                block = json.load(f)
                blocks.append(block)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error loading {block_file}: {e}")
            continue

    end_time = time.time()
    print(
        f"Loaded {len(blocks)} blocks ({total_size / (1024*1024):.2f} MB) in {end_time - start_time:.2f} seconds"
    )
    return blocks


def calculate_gas_fees(
    blocks: List[Dict[str, Any]]
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Calculate gas fees for all transactions in blocks."""
    start_time = time.time()
    all_gas_fees = []
    transactions = []
    total_txs = 0

    for block in blocks:
        for tx in block.get("transactions", []):
            total_txs += 1
            if "gasPrice" in tx:
                try:
                    gas_fee = int(tx["gasPrice"])
                    if gas_fee <= 0:
                        continue
                    all_gas_fees.append(gas_fee)
                    transactions.append(
                        {
                            "block_number": block["number"],
                            "tx_hash": tx["hash"],
                            "gas_price": gas_fee,
                            "from": tx["from"],
                            "to": tx.get("to", ""),
                            "timestamp": block.get("timestamp", 0),
                        }
                    )
                except (ValueError, TypeError) as e:
                    print(
                        f"Warning: Invalid gas price in transaction {tx.get('hash', 'unknown')}: {e}"
                    )
                    continue

    end_time = time.time()
    print(
        f"Processed {total_txs} total transactions, found {len(transactions)} valid transactions in {end_time - start_time:.2f} seconds"
    )
    return all_gas_fees, transactions


def calculate_ema(gas_fees: List[int], alpha: float = 0.1) -> List[float]:
    """Calculate Exponential Moving Average of gas fees."""
    start_time = time.time()
    if not gas_fees:
        return []

    ema = float(gas_fees[0])  # Initialize with first value
    ema_values = [ema]

    for fee in gas_fees[1:]:
        ema = alpha * float(fee) + (1 - alpha) * ema
        ema_values.append(ema)

    end_time = time.time()
    print(
        f"Calculated EMA for {len(gas_fees)} values in {end_time - start_time:.2f} seconds"
    )
    return ema_values


def calculate_rolling_std(gas_fees: List[int], window_size: int) -> List[float]:
    """Calculate rolling standard deviation using running sums for better performance."""
    if not gas_fees:
        return []

    n = len(gas_fees)
    rolling_std = []

    # Initialize running sums
    sum_x = 0
    sum_x2 = 0
    count = 0

    # Process first window
    for i in range(min(window_size, n)):
        x = gas_fees[i]
        sum_x += x
        sum_x2 += x * x
        count += 1
        if count > 1:
            mean = sum_x / count
            variance = (sum_x2 / count) - (mean * mean)
            std = math.sqrt(variance) if variance > 0 else 0
        else:
            std = 0
        rolling_std.append(std)

    # Process remaining elements
    for i in range(window_size, n):
        # Remove oldest element
        old_x = gas_fees[i - window_size]
        sum_x -= old_x
        sum_x2 -= old_x * old_x
        count -= 1

        # Add new element
        new_x = gas_fees[i]
        sum_x += new_x
        sum_x2 += new_x * new_x
        count += 1

        # Calculate standard deviation
        mean = sum_x / count
        variance = (sum_x2 / count) - (mean * mean)
        std = math.sqrt(variance) if variance > 0 else 0
        rolling_std.append(std)

    return rolling_std


def find_anomalies_with_ema(
    transactions: List[Dict[str, Any]],
    window_size: int = 100,
    threshold_multiplier: float = 2.0,
) -> List[Dict[str, Any]]:
    """Find transactions with gas fees significantly higher than their local EMA."""
    start_time = time.time()
    if not transactions:
        return []

    # Sort transactions by block number
    sort_start = time.time()
    sorted_txs = sorted(transactions, key=lambda x: x["block_number"])
    sort_end = time.time()
    print(
        f"Sorted {len(transactions)} transactions in {sort_end - sort_start:.2f} seconds"
    )

    # Extract gas fees in order
    gas_fees = [tx["gas_price"] for tx in sorted_txs]

    # Calculate EMA
    ema_values = calculate_ema(gas_fees)

    # Calculate rolling standard deviation
    std_start = time.time()
    rolling_std = calculate_rolling_std(gas_fees, window_size)
    std_end = time.time()
    print(f"Calculated rolling standard deviation in {std_end - std_start:.2f} seconds")

    # Find anomalies
    anomaly_start = time.time()
    anomalies = []
    for i, tx in enumerate(sorted_txs):
        if i == 0:  # Skip first transaction as we need it for EMA initialization
            continue

        current_ema = ema_values[i]
        current_std = rolling_std[i]
        threshold = current_ema + (threshold_multiplier * current_std)

        if tx["gas_price"] > threshold:
            anomalies.append(
                {
                    **tx,
                    "local_ema": current_ema,
                    "local_std": current_std,
                    "threshold": threshold,
                    "deviation": tx["gas_price"] - current_ema,
                    "deviation_percentage": (
                        (tx["gas_price"] - current_ema) / current_ema
                    )
                    * 100
                    if current_ema > 0
                    else float("inf"),
                }
            )

    anomaly_end = time.time()
    print(
        f"Found {len(anomalies)} anomalies in {anomaly_end - anomaly_start:.2f} seconds"
    )

    end_time = time.time()
    print(f"Total anomaly detection took {end_time - start_time:.2f} seconds")
    return anomalies


def main():
    try:
        total_start_time = time.time()

        print("Loading blocks...")
        blocks = load_blocks()

        window_size = 10000
        threshold_multiplier = 9.0

        print("Calculating gas fees...")
        all_gas_fees, transactions = calculate_gas_fees(blocks)

        if not all_gas_fees:
            print("No transactions found!")
            return

        print(f"\nAnalyzing {len(transactions)} transactions...")

        # Find anomalies using EMA
        anomalies = find_anomalies_with_ema(
            transactions, window_size, threshold_multiplier
        )

        print(f"\nFound {len(anomalies)} anomalous transactions")

        # Calculate some statistics about the anomalies
        if anomalies:
            anomaly_gas_fees = [a["gas_price"] for a in anomalies]
            print(f"\nAnomaly Statistics:")
            print(f"Highest gas fee: {max(anomaly_gas_fees)}")
            print(
                f"Average gas fee of anomalies: {statistics.mean(anomaly_gas_fees):.2f}"
            )
            print(f"Median gas fee of anomalies: {statistics.median(anomaly_gas_fees)}")
            print(
                f"Average deviation percentage: {statistics.mean([a['deviation_percentage'] for a in anomalies]):.2f}%"
            )

        # Save anomalies to file
        output_file = "gas_fee_anomalies.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "analysis_parameters": {
                        "window_size": window_size,
                        "threshold_multiplier": threshold_multiplier,
                        "total_transactions": len(transactions),
                    },
                    "anomalies": anomalies,
                },
                f,
                indent=2,
            )

        total_end_time = time.time()
        print(
            f"\nTotal execution time: {total_end_time - total_start_time:.2f} seconds"
        )
        print(f"Anomalies have been saved to '{output_file}'")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
