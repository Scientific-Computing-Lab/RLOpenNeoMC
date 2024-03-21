import os
import shutil
import sys

def copy_to_tmp(input_path):
    # Ensure the input path exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The specified path '{input_path}' does not exist.")

    # Get the base name of the input path (e.g., directory or file name)
    base_name = os.path.basename(input_path)

    # Create the destination path in /tmp
    tmp_destination = os.path.join("/tmp", base_name)

    try:
        # Copy the entire contents of the input path to /tmp
        shutil.copytree(input_path, tmp_destination)
        print(f"Successfully copied '{input_path}' to '{tmp_destination}'")
        return tmp_destination
    except Exception as e:
        print(f"Error copying '{input_path}' to '/tmp': {e}")
        return None

if __name__ == "__main__":
    # Check if the user provided a path as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    result_path = copy_to_tmp(input_path)

    if result_path:
        print(f"Full path of copied content in /tmp: {result_path}")
    else:
        print("Copying failed. Check the error messages for details.")

