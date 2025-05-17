import json
import os
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.ticker as ticker
import numpy as np
from scipy.signal import find_peaks

def load_blocks() -> List[Dict[str, Any]]:
    """Load all block files from the blocks directory."""
    blocks_dir = Path('blocks')
    if not blocks_dir.exists():
        raise FileNotFoundError(f"Blocks directory not found at {blocks_dir}")
    
    blocks = []
    for block_file in sorted(blocks_dir.glob('*.json')):
        try:
            with open(block_file, 'r') as f:
                block = json.load(f)
                blocks.append(block)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error loading {block_file}: {e}")
            continue
    
    print(f"Loaded {len(blocks)} blocks")
    return blocks

def detect_trend_changes(gas_prices: List[float], block_numbers: List[int], window_size: int = 100) -> List[Dict[str, Any]]:
    """Detect significant changes in gas price trends."""
    if len(gas_prices) < window_size * 2:
        return []
    
    # Calculate moving averages
    moving_avg = np.convolve(gas_prices, np.ones(window_size)/window_size, mode='valid')
    
    # Calculate rate of change
    rate_of_change = np.diff(moving_avg)
    
    # Calculate threshold based on standard deviation
    std_threshold = np.std(rate_of_change) * 2.0  # Increased threshold for more significant changes
    
    # Find peaks and troughs in rate of change with higher threshold
    peaks, _ = find_peaks(rate_of_change, distance=window_size*2, height=std_threshold)
    troughs, _ = find_peaks(-rate_of_change, distance=window_size*2, height=std_threshold)
    
    trend_changes = []
    
    # Analyze peaks (potential start of increase)
    for peak in peaks:
        if peak + window_size < len(block_numbers):
            # Only include if the change is significant (e.g., > 50% increase)
            current_price = gas_prices[peak + window_size]
            prev_price = moving_avg[peak]
            if current_price > prev_price * 1.5:  # 50% increase threshold
                trend_changes.append({
                    'block_number': block_numbers[peak + window_size],
                    'type': 'increase_start',
                    'gas_price': current_price,
                    'moving_avg': prev_price,
                    'change_percentage': ((current_price - prev_price) / prev_price) * 100
                })
    
    # Analyze troughs (potential start of decrease)
    for trough in troughs:
        if trough + window_size < len(block_numbers):
            # Only include if the change is significant (e.g., > 30% decrease)
            current_price = gas_prices[trough + window_size]
            prev_price = moving_avg[trough]
            if current_price < prev_price * 0.7:  # 30% decrease threshold
                trend_changes.append({
                    'block_number': block_numbers[trough + window_size],
                    'type': 'decrease_start',
                    'gas_price': current_price,
                    'moving_avg': prev_price,
                    'change_percentage': ((current_price - prev_price) / prev_price) * 100
                })
    
    return trend_changes

def analyze_trend_change_cause(blocks: List[Dict[str, Any]], change_point: Dict[str, Any], window_size: int = 10) -> Dict[str, Any]:
    """Analyze what caused a significant gas price change by examining transactions around the change point."""
    block_number = change_point['block_number']
    block_index = next((i for i, b in enumerate(blocks) if b['number'] == block_number), None)
    
    if block_index is None:
        return {}
    
    # Get blocks before and after the change point
    start_idx = max(0, block_index - window_size)
    end_idx = min(len(blocks), block_index + window_size + 1)
    relevant_blocks = blocks[start_idx:end_idx]
    
    # Analyze transaction patterns
    tx_counts = []
    gas_used = []
    gas_prices = []
    unique_senders = set()
    unique_recipients = set()
    
    for block in relevant_blocks:
        transactions = block.get('transactions', [])
        tx_counts.append(len(transactions))
        gas_used.append(block['gasUsed'])
        
        block_gas_prices = []
        for tx in transactions:
            if 'gasPrice' in tx and tx['gasPrice'] > 0:
                block_gas_prices.append(tx['gasPrice'])
                unique_senders.add(tx['from'])
                if 'to' in tx and tx['to']:
                    unique_recipients.add(tx['to'])
        
        avg_price = statistics.mean(block_gas_prices) if block_gas_prices else 0
        gas_prices.append(avg_price)
    
    # Calculate metrics
    avg_tx_count = statistics.mean(tx_counts)
    avg_gas_used = statistics.mean(gas_used)
    avg_gas_price = statistics.mean(gas_prices)
    
    # Calculate changes in metrics
    tx_count_change = ((tx_counts[-1] - tx_counts[0]) / tx_counts[0] * 100) if tx_counts[0] > 0 else float('inf')
    gas_used_change = ((gas_used[-1] - gas_used[0]) / gas_used[0] * 100) if gas_used[0] > 0 else float('inf')
    
    return {
        'block_number': block_number,
        'type': change_point['type'],
        'gas_price_change': change_point['change_percentage'],
        'metrics': {
            'avg_transactions_per_block': avg_tx_count,
            'avg_gas_used_per_block': avg_gas_used,
            'avg_gas_price': avg_gas_price,
            'unique_senders': len(unique_senders),
            'unique_recipients': len(unique_recipients),
            'tx_count_change_percentage': tx_count_change,
            'gas_used_change_percentage': gas_used_change
        }
    }

def analyze_gas_usage(blocks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Analyze gas usage patterns and their relationship with gas prices."""
    block_analysis = []
    total_stats = {
        'total_blocks': len(blocks),
        'total_transactions': 0,
        'total_gas_used': 0,
        'avg_gas_used': 0,
        'max_gas_used': 0,
        'min_gas_used': float('inf'),
        'gas_price_stats': {
            'min': float('inf'),
            'max': 0,
            'avg': 0
        },
        'filtered_blocks': 0  # Track number of blocks filtered out
    }
    
    for block in blocks:
        block_number = block['number']
        gas_used = block['gasUsed']
        gas_limit = block['gasLimit']
        timestamp = block['timestamp']
        
        # Get all transactions in the block
        transactions = block.get('transactions', [])
        
        # Skip blocks with no transactions or all transactions with 0 gasPrice
        if not transactions:
            total_stats['filtered_blocks'] += 1
            continue
            
        # Check if all transactions have 0 gasPrice
        all_zero_gas_price = all(tx.get('gasPrice', 0) == 0 for tx in transactions)
        if all_zero_gas_price:
            total_stats['filtered_blocks'] += 1
            continue
        
        # Calculate average gas price for transactions in this block
        gas_prices = []
        for tx in transactions:
            if 'gasPrice' in tx and tx['gasPrice'] > 0:
                gas_prices.append(tx['gasPrice'])
        
        # Calculate average gas price (use 0 if no non-zero gas prices found)
        avg_gas_price = statistics.mean(gas_prices) if gas_prices else 0
        
        block_analysis.append({
            'block_number': block_number,
            'gas_used': gas_used,
            'gas_limit': gas_limit,
            'gas_used_percentage': (gas_used / gas_limit) * 100,
            'avg_gas_price': avg_gas_price,
            'transaction_count': len(transactions),
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Update total statistics
        total_stats['total_transactions'] += len(transactions)
        total_stats['total_gas_used'] += gas_used
        total_stats['max_gas_used'] = max(total_stats['max_gas_used'], gas_used)
        total_stats['min_gas_used'] = min(total_stats['min_gas_used'], gas_used)
        
        if gas_prices:
            total_stats['gas_price_stats']['min'] = min(total_stats['gas_price_stats']['min'], min(gas_prices))
            total_stats['gas_price_stats']['max'] = max(total_stats['gas_price_stats']['max'], max(gas_prices))
            total_stats['gas_price_stats']['avg'] = statistics.mean(gas_prices)
    
    # Calculate averages
    if block_analysis:  # Only calculate if we have blocks after filtering
        total_stats['avg_gas_used'] = total_stats['total_gas_used'] / len(block_analysis)
    
    return block_analysis, total_stats

def format_gas_price(x, pos):
    """Format gas price in Gwei."""
    return f'{x/1e9:.1f} Gwei'

def plot_gas_analysis(block_analysis: List[Dict[str, Any]], blocks: List[Dict[str, Any]], output_dir: str = 'analysis'):
    """Create plots to visualize gas usage patterns."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    block_numbers = [b['block_number'] for b in block_analysis]
    avg_gas_prices = [b['avg_gas_price'] for b in block_analysis]
    
    # Detect trend changes
    trend_changes = detect_trend_changes(avg_gas_prices, block_numbers)
    
    # Analyze causes of trend changes
    trend_analysis = []
    for change in trend_changes:
        cause_analysis = analyze_trend_change_cause(blocks, change)
        if cause_analysis:
            trend_analysis.append(cause_analysis)
    
    # Save trend changes and their causes to JSON
    trend_changes_file = os.path.join(output_dir, 'gas_price_trends.json')
    with open(trend_changes_file, 'w') as f:
        json.dump({
            'increases': [c for c in trend_changes if c['type'] == 'increase_start'],
            'decreases': [c for c in trend_changes if c['type'] == 'decrease_start'],
            'trend_analysis': trend_analysis
        }, f, indent=2)
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot gas prices
    plt.plot(block_numbers, avg_gas_prices, label='Gas Price', color='gray', alpha=0.5)
    
    # Highlight trend changes
    increases = [c for c in trend_changes if c['type'] == 'increase_start']
    decreases = [c for c in trend_changes if c['type'] == 'decrease_start']
    
    if increases:
        plt.scatter([c['block_number'] for c in increases], 
                   [c['gas_price'] for c in increases],
                   color='red', marker='^', s=100, label='Significant Increase')
    
    if decreases:
        plt.scatter([c['block_number'] for c in decreases], 
                   [c['gas_price'] for c in decreases],
                   color='green', marker='v', s=100, label='Significant Decrease')
    
    # Set logarithmic scale for gas price
    plt.yscale('log')
    
    # Format gas price axis to show Gwei
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_gas_price))
    
    plt.xlabel('Block Number')
    plt.ylabel('Average Gas Price (Gwei)')
    plt.title('Significant Gas Price Trend Changes')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add annotations for significant changes
    for change in trend_changes:
        plt.annotate(f"{change['change_percentage']:.1f}%",
                    (change['block_number'], change['gas_price']),
                    xytext=(10, 10 if change['type'] == 'increase_start' else -20),
                    textcoords='offset points',
                    ha='left',
                    va='bottom' if change['type'] == 'increase_start' else 'top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gas_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print trend change information with causes
    print("\nSignificant Gas Price Trend Changes and Their Causes:")
    for analysis in trend_analysis:
        print(f"\nBlock {analysis['block_number']}: {analysis['type'].replace('_', ' ').title()}")
        print(f"Gas Price Change: {analysis['gas_price_change']:.1f}%")
        print("Causes:")
        print(f"  - Average Transactions per Block: {analysis['metrics']['avg_transactions_per_block']:.1f}")
        print(f"  - Average Gas Used per Block: {analysis['metrics']['avg_gas_used_per_block']:,.0f}")
        print(f"  - Average Gas Price: {analysis['metrics']['avg_gas_price']/1e9:.2f} Gwei")
        print(f"  - Unique Senders: {analysis['metrics']['unique_senders']}")
        print(f"  - Unique Recipients: {analysis['metrics']['unique_recipients']}")
        print(f"  - Transaction Count Change: {analysis['metrics']['tx_count_change_percentage']:.1f}%")
        print(f"  - Gas Used Change: {analysis['metrics']['gas_used_change_percentage']:.1f}%")
    
    print(f"\nTrend changes and their causes have been saved to '{trend_changes_file}'")

def main():
    try:
        print("Loading blocks...")
        blocks = load_blocks()
        
        print("Analyzing gas usage patterns...")
        block_analysis, total_stats = analyze_gas_usage(blocks)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total Blocks Analyzed: {total_stats['total_blocks']}")
        print(f"Blocks Filtered Out (0 gasPrice only): {total_stats['filtered_blocks']}")
        print(f"Blocks Used in Analysis: {len(block_analysis)}")
        print(f"Total Transactions: {total_stats['total_transactions']}")
        print(f"Average Gas Used per Block: {total_stats['avg_gas_used']:,.2f}")
        print(f"Maximum Gas Used in a Block: {total_stats['max_gas_used']:,.2f}")
        print(f"Minimum Gas Used in a Block: {total_stats['min_gas_used']:,.2f}")
        print("\nGas Price Statistics:")
        print(f"Minimum Gas Price: {total_stats['gas_price_stats']['min']/1e9:.2f} Gwei")
        print(f"Maximum Gas Price: {total_stats['gas_price_stats']['max']/1e9:.2f} Gwei")
        print(f"Average Gas Price: {total_stats['gas_price_stats']['avg']/1e9:.2f} Gwei")
        
        # Generate plots
        print("\nGenerating plots...")
        plot_gas_analysis(block_analysis, blocks)
        
        # Save detailed analysis to JSON
        output_file = 'gas_usage_analysis.json'
        with open(output_file, 'w') as f:
            json.dump({
                'summary_statistics': total_stats,
                'block_analysis': block_analysis
            }, f, indent=2)
        
        print(f"\nAnalysis complete! Results have been saved to '{output_file}'")
        print("Plots have been saved to the 'analysis' directory")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main() 