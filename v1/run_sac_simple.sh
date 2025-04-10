#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display help menu
show_help() {
    echo -e "${BLUE}===== Rocket Landing SAC Script =====${NC}"
    echo ""
    echo "Usage: ./run_sac_simple.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  setup        Create and activate a virtual environment and install dependencies"
    echo "  train        Train a new SAC model from scratch"
    echo "  continue     Continue training from the last checkpoint"
    echo "  run          Run the trained SAC model for visualization"
    echo "  compare      Compare SAC with the existing A2C algorithm"
    echo "  help         Display this help menu"
    echo ""
    echo "Examples:"
    echo "  ./run_sac_simple.sh setup        # Set up the environment"
    echo "  ./run_sac_simple.sh train        # Start training from scratch"
}

# Check if the script was called with an argument
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: Missing argument${NC}"
    show_help
    exit 1
fi

# Process command line arguments
case "$1" in
    setup)
        echo -e "${GREEN}Setting up environment...${NC}"
        
        # Create a virtual environment
        python3 -m venv sac_venv
        
        # Activate the virtual environment
        source sac_venv/bin/activate
        
        # Install dependencies
        pip install -r requirements.txt
        
        echo -e "${GREEN}Setup complete!${NC}"
        echo -e "To activate the environment in the future, run:"
        echo -e "${YELLOW}source sac_venv/bin/activate${NC}"
        echo -e "To train the model, run:"
        echo -e "${YELLOW}./run_sac_simple.sh train${NC}"
        ;;
        
    train)
        echo -e "${GREEN}Starting SAC training from scratch...${NC}"
        
        # Check if we're in the virtual environment
        if [[ "$VIRTUAL_ENV" == "" ]]; then
            echo -e "${YELLOW}Warning: Virtual environment not activated. Attempting to activate...${NC}"
            if [ -d "sac_venv" ]; then
                source sac_venv/bin/activate
            else
                echo -e "${RED}Error: Virtual environment not found. Please run './run_sac_simple.sh setup' first.${NC}"
                exit 1
            fi
        fi
        
        # Set FROM_ZERO to True in the script
        sed -i '' 's/FROM_ZERO = False/FROM_ZERO = True/g' sac_train.py 2>/dev/null || \
        sed -i 's/FROM_ZERO = False/FROM_ZERO = True/g' sac_train.py
        
        # Run the training script
        python3 sac_train.py
        ;;
        
    continue)
        echo -e "${GREEN}Continuing SAC training from last checkpoint...${NC}"
        
        # Check if we're in the virtual environment
        if [[ "$VIRTUAL_ENV" == "" ]]; then
            echo -e "${YELLOW}Warning: Virtual environment not activated. Attempting to activate...${NC}"
            if [ -d "sac_venv" ]; then
                source sac_venv/bin/activate
            else
                echo -e "${RED}Error: Virtual environment not found. Please run './run_sac_simple.sh setup' first.${NC}"
                exit 1
            fi
        fi
        
        # Set FROM_ZERO to False in the script
        sed -i '' 's/FROM_ZERO = True/FROM_ZERO = False/g' sac_train.py 2>/dev/null || \
        sed -i 's/FROM_ZERO = True/FROM_ZERO = False/g' sac_train.py
        
        # Run the training script
        python3 sac_train.py
        ;;
        
    run)
        echo -e "${GREEN}Running trained SAC model...${NC}"
        
        # Check if we're in the virtual environment
        if [[ "$VIRTUAL_ENV" == "" ]]; then
            echo -e "${YELLOW}Warning: Virtual environment not activated. Attempting to activate...${NC}"
            if [ -d "sac_venv" ]; then
                source sac_venv/bin/activate
            else
                echo -e "${RED}Error: Virtual environment not found. Please run './run_sac_simple.sh setup' first.${NC}"
                exit 1
            fi
        fi
        
        # Check if the checkpoint exists
        if [ ! -f "landing_sac_ckpt/sac_checkpoint.pt" ]; then
            echo -e "${RED}Error: No trained model found. Please train the model first.${NC}"
            exit 1
        fi
        
        # Run the evaluation script
        python3 sac_run.py
        ;;
        
    compare)
        echo -e "${GREEN}Comparing SAC with A2C algorithm...${NC}"
        
        # Check if we're in the virtual environment
        if [[ "$VIRTUAL_ENV" == "" ]]; then
            echo -e "${YELLOW}Warning: Virtual environment not activated. Attempting to activate...${NC}"
            if [ -d "sac_venv" ]; then
                source sac_venv/bin/activate
            else
                echo -e "${RED}Error: Virtual environment not found. Please run './run_sac_simple.sh setup' first.${NC}"
                exit 1
            fi
        fi
        
        # Check if both checkpoints exist
        if [ ! -f "landing_sac_ckpt/sac_checkpoint.pt" ]; then
            echo -e "${YELLOW}Warning: No trained SAC model found. SAC comparison will be skipped.${NC}"
        fi
        
        # Check for A2C checkpoints
        A2C_CKPT=$(find landing_ckpt -name "*.pt" 2>/dev/null | wc -l)
        if [ "$A2C_CKPT" -eq 0 ]; then
            echo -e "${YELLOW}Warning: No trained A2C model found. A2C comparison will be skipped.${NC}"
        fi
        
        if [ ! -f "landing_sac_ckpt/sac_checkpoint.pt" ] && [ "$A2C_CKPT" -eq 0 ]; then
            echo -e "${RED}Error: No trained models found. Please train at least one model first.${NC}"
            exit 1
        fi
        
        # Run the comparison script
        python3 compare_algorithms.py
        echo -e "\n${GREEN}Comparison results saved in algorithm_comparison_landing directory${NC}"
        ;;
        
    help|*)
        show_help
        ;;
esac

exit 0 