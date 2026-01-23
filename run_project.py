#!/usr/bin/env python3
"""Main runner script for Breast Cancer Identification Project.

This script runs all phases of the project sequentially without using notebooks.
It will use your GPU for training if available.
"""

import os
import sys
from pathlib import Path
import torch

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70 + "\n")

def check_gpu():
    """Check and print GPU information."""
    print_header("GPU Configuration")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("\n✓ GPU is ready for training!")
        return 'cuda'
    else:
        print("\n⚠ No GPU detected. Training will be slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
        return 'cpu'

def check_dataset():
    """Check if dataset is available."""
    print_header("Dataset Check")
    breakhis_path = project_root / "data" / "BreaKHis_v1"
    
    if breakhis_path.exists():
        print(f"✓ BreakHis dataset found at: {breakhis_path}")
        
        # Count images
        benign_count = len(list((breakhis_path / "benign").rglob("*.png")))
        malignant_count = len(list((breakhis_path / "malignant").rglob("*.png")))
        
        print(f"\nDataset Statistics:")
        print(f"  Benign images: {benign_count}")
        print(f"  Malignant images: {malignant_count}")
        print(f"  Total images: {benign_count + malignant_count}")
        return True
    else:
        print(f"✗ BreakHis dataset NOT found at: {breakhis_path}")
        print("\nPlease download the dataset first:")
        print("  python download_breakhis.py")
        print("\nOr download manually from:")
        print("  https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/")
        return False

def run_phase1():
    """Run Phase 1: Data Preparation."""
    print_header("Phase 1: Data Preparation")
    
    try:
        from phase1_data_preparation.prepare_data import main as prepare_data_main
        prepare_data_main()
        print("\n✓ Phase 1 completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_phase2(device='cuda'):
    """Run Phase 2: Model Training."""
    print_header("Phase 2: Model Training")
    
    print(f"Training device: {device}")
    print("This may take several hours depending on your GPU...")
    
    try:
        from phase2_model_development.train import main as train_main
        train_main()
        print("\n✓ Phase 2 completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_phase3():
    """Run Phase 3: Multi-Modal Fusion."""
    print_header("Phase 3: Multi-Modal Fusion")
    
    try:
        from phase3_multimodal_fusion.fusion import main as fusion_main
        fusion_main()
        print("\n✓ Phase 3 completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_phase4():
    """Run Phase 4: Explainability (XAI)."""
    print_header("Phase 4: Explainability & Visualization")
    
    try:
        from phase4_explainability.xai import main as xai_main
        xai_main()
        print("\n✓ Phase 4 completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_streamlit_app():
    """Launch the Streamlit dashboard."""
    print_header("Phase 5: Launching Streamlit Dashboard")
    
    print("Starting Streamlit app...")
    print("The app will open in your browser at http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.\n")
    
    import subprocess
    subprocess.run([
        sys.executable, 
        "-m", 
        "streamlit", 
        "run", 
        "phase5_deployment/streamlit_app.py"
    ])

def main():
    """Main execution function."""
    print_header("Breast Cancer Identification - Multi-Modal Deep Learning")
    
    # Check GPU
    device = check_gpu()
    
    # Check dataset
    if not check_dataset():
        print("\n⚠ Dataset not found. Please download it first.")
        response = input("\nDo you want to download it now? (y/n): ")
        if response.lower() == 'y':
            print("\nRunning download script...")
            os.system(f"{sys.executable} download_breakhis.py")
            if not check_dataset():
                print("\n✗ Download failed. Exiting.")
                sys.exit(1)
        else:
            sys.exit(0)
    
    # Ask user which phases to run
    print("\n" + "-" * 70)
    print("Select phases to run:")
    print("-" * 70)
    print("1. Phase 1: Data Preparation")
    print("2. Phase 2: Model Training (GPU required, takes time)")
    print("3. Phase 3: Multi-Modal Fusion")
    print("4. Phase 4: Explainability (Grad-CAM, SHAP)")
    print("5. Phase 5: Launch Streamlit Dashboard")
    print("6. Run all phases")
    print("0. Exit")
    print("-" * 70)
    
    choice = input("\nEnter your choice (0-6): ").strip()
    
    if choice == '0':
        print("\nExiting...")
        sys.exit(0)
    
    elif choice == '1':
        run_phase1()
    
    elif choice == '2':
        run_phase2(device)
    
    elif choice == '3':
        run_phase3()
    
    elif choice == '4':
        run_phase4()
    
    elif choice == '5':
        run_streamlit_app()
    
    elif choice == '6':
        print("\n🚀 Running all phases sequentially...\n")
        
        success = True
        success = success and run_phase1()
        
        if success:
            success = success and run_phase2(device)
        
        if success:
            success = success and run_phase3()
        
        if success:
            success = success and run_phase4()
        
        if success:
            print_header("All Phases Completed Successfully!")
            response = input("\nWould you like to launch the dashboard? (y/n): ")
            if response.lower() == 'y':
                run_streamlit_app()
    
    else:
        print("\n✗ Invalid choice. Exiting.")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
