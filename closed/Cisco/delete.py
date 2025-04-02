import os

# Define the root directory where the search will begin
root_directory = './'  # Change this to the path you want to start the search from

# Function to delete all metadata.json files in all subdirectories
def delete_metadata_json_files(directory):
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(directory):
        # Check if 'metadata.json' exists in the current directory
        if 'metadata.json' in files:
            file_path = os.path.join(root, 'metadata.json')
            try:
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")

# Start the process from the root directory
delete_metadata_json_files(root_directory)
