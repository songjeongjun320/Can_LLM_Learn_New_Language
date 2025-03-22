#!/bin/bash

# Default values
DEFAULT_BASE_PATH="/scratch/jsong132/Can_LLM_Learn_New_Language/DB"
DEFAULT_VERSION="v2"

# Help text
print_usage() {
    echo "Usage: $0 --drama DRAMA_FOLDER --host OLLAMA_HOST [--path BASE_PATH] [--version VERSION]"
    echo ""
    echo "Required arguments:"
    echo "  --drama    Name of the drama folder to process"
    echo "  --host     Ollama host address (e.g., http://sg022:11434)"
    echo ""
    echo "Optional arguments:"
    echo "  --path     Base path where data folders are located (default: $DEFAULT_BASE_PATH)"
    echo "  --version  Version string for output directories (default: $DEFAULT_VERSION)"
    echo "  --help     Display this help message"
}

# Check for no arguments
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

# Parse command line arguments
drama_folder_name=""
ollama_host=""
base_path="$DEFAULT_BASE_PATH"
version="$DEFAULT_VERSION"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --drama)
            drama_folder_name="$2"
            shift
            shift
            ;;
        --host)
            ollama_host="$2"
            shift
            shift
            ;;
        --path)
            base_path="$2"
            shift
            shift
            ;;
        --version)
            version="$2"
            shift
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if required parameters are provided
if [[ -z "$drama_folder_name" || -z "$ollama_host" ]]; then
    echo "Error: Missing required parameters"
    print_usage
    exit 1
fi

# Submit the batch job
sbatch processor.sh --drama "$drama_folder_name" --host "$ollama_host" --path "$base_path" --version "$version"

echo "Job submitted for drama folder: $drama_folder_name"