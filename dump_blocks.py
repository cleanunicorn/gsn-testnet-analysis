import json
import os
import time
import signal
import sys
from web3 import Web3
from eth_utils import to_hex

def ensure_blocks_directory():
    """Create blocks directory if it doesn't exist."""
    if not os.path.exists('blocks'):
        os.makedirs('blocks')

def get_latest_saved_block():
    """Get the latest block number that has been saved."""
    if not os.path.exists('blocks'):
        return None
    
    # Get all block files and find the highest number
    block_files = [f for f in os.listdir('blocks') if f.endswith('.json')]
    if not block_files:
        return None
    
    # Extract block numbers from filenames and find the maximum
    block_numbers = [int(f.split('.')[0]) for f in block_files]
    return max(block_numbers)

def convert_to_serializable(obj):
    """Convert Web3.py objects to JSON serializable format."""
    if isinstance(obj, (bytes, bytearray)):
        return to_hex(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    return obj

def save_block_to_file(block):
    """Save a block to a JSON file with proper formatting."""
    ensure_blocks_directory()
    
    # Get block number and format it with leading zeros
    block_number = str(block['number']).zfill(9)
    filename = f'blocks/{block_number}.json'
    
    # Convert block to serializable format
    block_dict = convert_to_serializable(block)
    
    # Save to file with proper formatting
    with open(filename, 'w') as f:
        json.dump(block_dict, f, indent=2)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\nGracefully shutting down...")
    sys.exit(0)

def main():
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Connect to the network
    w3 = Web3(Web3.HTTPProvider('https://gensyn-testnet.g.alchemy.com/v2/U_kbGcKPghVKFgYl3tBe_vpVjTxXKK_x'))
    
    if not w3.is_connected():
        print("Failed to connect to the network")
        return
    
    print("Press Ctrl+C to stop the script")
    
    # Get the latest saved block number
    latest_saved = get_latest_saved_block()
    start_block = latest_saved + 1 if latest_saved is not None else 2714870
    end_block = 2893263
    
    if start_block > end_block:
        print("All blocks have been saved. Waiting for new blocks...")
        return
    
    print(f"Starting to save blocks from {start_block} to {end_block}")
    
    for block_number in range(start_block, end_block + 1):
        try:
            block = w3.eth.get_block(block_number, True)
            save_block_to_file(block)
            print(f"Saved block {block_number}")
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error saving block {block_number}: {str(e)}")
            # Continue with next block even if there's an error
            continue

if __name__ == "__main__":
    main()
