"""
Utility functions for HIVD Classifier
Maltezos, V. (2026). A framework for efficient image categorisation in social research
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiofiles


def get_app_directory() -> Path:
    """Get the application directory (works for both dev and packaged exe)"""
    if getattr(sys, 'frozen', False):
        # Running as packaged executable - use exe's directory
        return Path(os.path.dirname(sys.executable))
    else:
        # Running as script - use the HIVD_classifier folder
        return Path(__file__).parent.parent


async def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    app_dir = get_app_directory()
    
    # Try multiple locations for config file
    search_paths = [
        app_dir / config_path,                    # App directory
        Path(config_path),                         # Current directory
        Path(__file__).parent / config_path,       # Backend directory
        Path(__file__).parent.parent / config_path # Parent of backend
    ]
    
    path = None
    for p in search_paths:
        if p.exists():
            path = p
            break
    
    if path is None:
        raise FileNotFoundError(f"Config file not found. Searched: {[str(p) for p in search_paths]}")
    
    print(f"Loading config from: {path.absolute()}")
    
    # Use app_dir as the base for resolving relative paths
    config_dir = app_dir
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Resolve paths relative to app directory
    if 'paths' in config:
        for key, value in config['paths'].items():
            p = Path(value)
            if not p.is_absolute():
                resolved = (config_dir / value).resolve()
                config['paths'][key] = str(resolved)
                # Create directory if it doesn't exist (for data folders)
                if key in ['predictions_dir', 'classes_dir', 'image_dataset', 'inference_dataset']:
                    resolved.mkdir(parents=True, exist_ok=True)
                
    # Also resolve YOLO model path if it looks like a path
    if 'yolo' in config and 'model' in config['yolo']:
        model_val = config['yolo']['model']
        if '/' in model_val or '\\' in model_val or model_val.endswith('.pt'):
            p = Path(model_val)
            if not p.is_absolute():
                config_model_path = config_dir / model_val
                if config_model_path.exists():
                    config['yolo']['model'] = str(config_model_path.resolve())
            
    return config


def get_image_files(directory: Path, extensions: List[str] = None) -> List[str]:
    """
    Get all image files in a directory
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
    
    Returns:
        List of absolute paths to image files (deduplicated)
    """
    if extensions is None:
        # Use lowercase only - glob is case-insensitive on Windows
        extensions = ['jpg', 'jpeg', 'png']
    
    image_files = set()  # Use set to avoid duplicates
    for ext in extensions:
        # Search with lowercase extension
        for p in directory.glob(f'**/*.{ext}'):
            image_files.add(str(p.resolve()))
        # Also search with uppercase extension (for case-sensitive systems)
        for p in directory.glob(f'**/*.{ext.upper()}'):
            image_files.add(str(p.resolve()))
    
    # Return as sorted list
    return sorted(list(image_files))


async def save_class_config(class_def: 'ClassDefinition', classes_dir: Path):
    """
    Save class configuration to file system
    
    Creates a folder structure:
    classes/{class_name}/
        config.json
        description.txt
        references/
            ref1.jpg
            ref2.jpg
            ...
    """
    class_dir = classes_dir / class_def.name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = class_dir / "config.json"
    async with aiofiles.open(config_path, 'w') as f:
        await f.write(class_def.model_dump_json(indent=2))
    
    # Save description
    desc_path = class_dir / "description.txt"
    async with aiofiles.open(desc_path, 'w') as f:
        await f.write(class_def.description)
    
    # Copy reference images
    references_dir = class_dir / "references"
    references_dir.mkdir(exist_ok=True)
    
    # Note: Reference images are already in image_dataset,
    # so we just save the paths in config


async def load_class_configs(classes_dir: Path) -> List[Dict[str, Any]]:
    """Load all class configurations from file system"""
    configs = []
    
    if not classes_dir.exists():
        return configs
    
    for class_dir in classes_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        config_path = class_dir / "config.json"
        if config_path.exists():
            async with aiofiles.open(config_path, 'r') as f:
                content = await f.read()
                configs.append(json.loads(content))
    
    return configs


def create_prediction_structure(
    predictions_dir: Path,
    job_name: str,
    class_names: List[str]
) -> tuple[Path, Path]:
    """
    Create directory structure for predictions
    
    Returns:
        Dictionary mapping class names to their (high_prob_dir, low_prob_dir) tuple
    """
    job_dir = predictions_dir / job_name
    
    # Create a mapping for easy lookup
    class_paths = {}
    
    for class_name in class_names:
        class_dir = job_dir / class_name
        high_prob_dir = class_dir / "HighProbability"
        low_prob_dir = class_dir / "LowProbability"
        
        high_prob_dir.mkdir(parents=True, exist_ok=True)
        low_prob_dir.mkdir(parents=True, exist_ok=True)
        
        class_paths[class_name] = (high_prob_dir, low_prob_dir)
    
    return class_paths


def save_image_to_class(
    image_path: str,
    class_name: str,
    class_paths: Dict[str, tuple[Path, Path]],
    is_high_confidence: bool,
    subclass_name: Optional[str] = None
):
    """Copy image to appropriate class folder, with optional subcategorisation"""
    import shutil
    
    high_prob_dir, low_prob_dir = class_paths[class_name]
    target_dir = high_prob_dir if is_high_confidence else low_prob_dir
    
    # If there's a subclass, create a subfolder within the confidence folder
    if subclass_name:
        dest_dir = target_dir / subclass_name
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        dest_dir = target_dir
    
    dest_path = dest_dir / Path(image_path).name
    shutil.copy2(image_path, dest_path)


def format_citation() -> str:
    """Return formatted citation for the framework"""
    return ("Maltezos, V. (2026). A framework for efficient image categorisation "
            "in social research: Addressing high intra-class visual diversity. "
            "Dissertationes Universitatis Helsingiensis (27/2026). "
            "Helsingin yliopisto. http://urn.fi/URN:ISBN:978-952-84-1827-6")
