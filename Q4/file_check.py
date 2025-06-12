# import pandas as pd
# import os

# # Read the parquet file
# print("Reading parquet file...")
# df = pd.read_parquet("feature_matrix.parquet", engine="pyarrow")

# # Display information
# print("Columns:", df.columns)
# print("\nSample data:")
# print(df.head())

# # Get size information
# memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
# print(f"\nDataFrame size in memory: {memory_usage:.2f} MB")
# print(f"Number of rows: {len(df)}")

# # Create CSV filename
# csv_filename = os.path.splitext(os.path.basename("pat.parquet"))[0] + ".csv"
# print(f"\nConverting to CSV: {csv_filename}")

# # Save to CSV
# df.to_csv(csv_filename, index=False)
# print(f"CSV file saved: {csv_filename}")

# # Check CSV file size
# csv_size_mb = os.path.getsize(csv_filename) / (1024 * 1024)
# print(f"CSV file size: {csv_size_mb:.2f} MB")


















# create_sample.py
import os

# --- Configuration ---
input_filename = 'all_market_data.jsonl'
output_filename = 'market_data_5mb_sample.jsonl'
# 5 MB in bytes (5 * 1024 * 1024)
sample_size_bytes = 5 * 1024 * 1024 
# -------------------

print(f"Creating a {sample_size_bytes / 1024 / 1024:.2f}MB sample from '{input_filename}'...")

try:
    with open(input_filename, 'rb') as infile, open(output_filename, 'wb') as outfile:
        # Read the specified number of bytes from the input file
        data_chunk = infile.read(sample_size_bytes)
        # Write that chunk to the output file
        outfile.write(data_chunk)
        
    # We need to make sure the last line is complete.
    # Open the new file, read all lines, and write back all but the potentially incomplete last one.
    with open(output_filename, 'r') as f:
        lines = f.readlines()
    
    with open(output_filename, 'w') as f:
        f.writelines(lines[:-1])

    final_size = os.path.getsize(output_filename)
    print(f"\nSuccessfully created sample file '{output_filename}'")
    print(f"Final size: {final_size / 1024 / 1024:.2f}MB")

except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")