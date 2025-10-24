import os
from pathlib import Path
import shutil

def create_project_structure():
    """Create all project directories"""
    print("="*50)
    print("SETTING UP NETFLIX ML PROJECT")
    print("="*50)
    
    # Define project root (current directory)
    project_root = Path.cwd()
    
    # Define directories to create
    directories = [
        "data/raw",
        "data/processed",
        "src",
        "models",
        "outputs",
        "logs",
        "notebooks",
        "tests"
    ]
    
    print("\n📁 Creating directory structure...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}")
    
    return project_root

def move_data_file(project_root):
    """Move netflix_titles.csv to data/raw if it exists"""
    print("\n📊 Looking for netflix_titles.csv...")
    
    # Check if file exists in current directory
    source_file = project_root / "netflix_titles.csv"
    destination = project_root / "data" / "raw" / "netflix_titles.csv"
    
    if source_file.exists():
        if not destination.exists():
            shutil.copy(source_file, destination)
            print(f"  ✓ Copied netflix_titles.csv to data/raw/")
        else:
            print(f"  ℹ Data file already exists in data/raw/")
    else:
        # Check parent directory
        parent_file = project_root.parent / "netflix_titles.csv"
        if parent_file.exists():
            shutil.copy(parent_file, destination)
            print(f"  ✓ Copied netflix_titles.csv from parent directory to data/raw/")
        else:
            print(f"  ⚠ netflix_titles.csv not found!")
            print(f"  → Please manually copy it to: {destination}")

def create_src_files(project_root):
    """Create __init__.py in src directory"""
    print("\n📝 Creating Python package files...")
    
    src_init = project_root / "src" / "__init__.py"
    if not src_init.exists():
        src_init.write_text("# Netflix ML Analysis Package\n")
        print(f"  ✓ Created src/__init__.py")

def create_requirements_txt(project_root):
    """Create requirements.txt file"""
    print("\n📦 Creating requirements.txt...")
    
    requirements = """numpy>=2.3.4
pandas>=2.3.3
matplotlib>=3.9.0
seaborn>=0.13.0
scikit-learn>=1.7.2
torch>=2.5.1
torchvision>=0.20.1
jupyter>=1.0.0
joblib>=1.3.0
"""
    
    req_file = project_root / "requirements.txt"
    req_file.write_text(requirements)
    print(f"  ✓ Created requirements.txt")

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*50)
    print("✅ SETUP COMPLETE!")
    print("="*50)
    
    print("\n📋 NEXT STEPS:")
    print("\n1. Verify your data file is in place:")
    print("   → Check: data/raw/netflix_titles.csv")
    
    print("\n2. Create and activate virtual environment:")
    print("   py -3.12 -m venv ml_env")
    print("   ml_env\\Scripts\\activate")
    
    print("\n3. Install dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n4. Copy all the Python files to src/ directory:")
    print("   - config.py")
    print("   - data_loader.py")
    print("   - preprocessing.py")
    print("   - train.py")
    print("   - test.py")
    
    print("\n5. Run the pipeline:")
    print("   python src/data_loader.py      # Explore data")
    print("   python src/preprocessing.py    # Clean and process")
    print("   python src/train.py            # Train model")
    print("   python src/test.py             # Evaluate model")
    
    print("\n" + "="*50)
    print("Happy ML modeling! 🚀")
    print("="*50)

def main():
    """Main setup function"""
    try:
        # Create directories
        project_root = create_project_structure()
        
        # Move data file
        move_data_file(project_root)
        
        # Create package files
        create_src_files(project_root)
        
        # Create requirements.txt
        create_requirements_txt(project_root)
        
        # Print next steps
        print_next_steps()
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()