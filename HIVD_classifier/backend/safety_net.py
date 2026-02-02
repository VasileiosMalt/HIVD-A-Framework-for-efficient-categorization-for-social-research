"""
Object detection safety net for post-classification verification
Adapted from supplementary_program.py
Maltezos, V. (2026). A framework for efficient image categorisation in social research
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from ultralytics import YOLO
from PIL import Image

# Suppress YOLO verbose output
logging.getLogger('ultralytics').setLevel(logging.WARNING)


class SafetyNet:
    """
    Object detection-based post-processing to identify potential misclassifications
    """
    
    def __init__(self, model_path: str = "yolov8m.pt"):
        """Initialize YOLO model for object detection"""
        self.model = YOLO(model_path, verbose=False)
    
    def detect_objects(self, image_path: str) -> Dict[str, int]:
        """
        Detect objects in an image and return counts
        
        Returns:
            Dict mapping object names to counts
        """
        try:
            results = self.model(image_path, verbose=False)
            
            # Extract detections
            object_counts = {}
            for result in results:
                if hasattr(result, 'boxes'):
                    for box in result.boxes:
                        if hasattr(box, 'cls'):
                            class_id = int(box.cls.item())
                            class_name = result.names[class_id]
                            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            return object_counts
        except Exception as e:
            print(f"Error detecting objects in {image_path}: {e}")
            return {}
    
    def check_person_size(self, image_path: str, min_coverage: float = 0.25) -> List[bool]:
        """
        Check if detected persons cover sufficient portion of image
        
        Returns:
            List of booleans indicating which persons meet size threshold
        """
        try:
            results = self.model(image_path, verbose=False)
            img = Image.open(image_path)
            img_area = img.width * img.height
            
            large_persons = []
            for result in results:
                if hasattr(result, 'boxes'):
                    for box in result.boxes:
                        if hasattr(box, 'cls') and result.names[int(box.cls.item())] == 'person':
                            # Calculate box area
                            xyxy = box.xyxy[0].cpu().numpy()
                            box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                            coverage = box_area / img_area
                            large_persons.append(coverage > min_coverage)
            
            return large_persons
        except Exception as e:
            print(f"Error checking person size in {image_path}: {e}")
            return []
    
    def apply_safety_rules(
        self,
        predictions_dir: Path,
        high_prob_dir: Path,
        low_prob_dir: Path,
        rules: Optional[List[Dict]] = None
    ) -> Dict[str, Dict]:
        """
        Apply safety net rules to classified images
        
        Args:
            predictions_dir: Root predictions directory
            high_prob_dir: High probability directory
            low_prob_dir: Low probability directory
            rules: List of rule dictionaries
        
        Returns:
            Dict with statistics about moved images
        """
        if rules is None:
            rules = self._get_default_rules()
        
        stats = {}
        
        for prob_level, prob_dir in [("HighProbability", high_prob_dir), 
                                       ("LowProbability", low_prob_dir)]:
            
            stats[prob_level] = {}
            
            for rule in rules:
                category_name = rule.get('main_category') or rule.get('category_name')
                if not category_name:
                    continue
                
                category_path = prob_dir / category_name
                if not category_path.exists():
                    # Try partial match if name doesn't match exactly
                    for d in prob_dir.iterdir():
                        if d.is_dir() and (category_name in d.name or d.name in category_name):
                            category_path = d
                            break
                    
                    if not category_path.exists():
                        continue
                
                # Create suspected false positives folder
                suspected_fp_path = category_path / "SuspectedFalsePositives"
                suspected_fp_path.mkdir(exist_ok=True)
                
                # Get all images in category
                image_files = self._get_image_files(category_path)
                
                moved_count = 0
                for img_path in image_files:
                    action = self._check_image_against_rule(img_path, rule)
                    
                    if action:
                        # Determine destination
                        if action == "SuspectedFalsePositives":
                            dest_path = suspected_fp_path / img_path.name
                        else:
                            dest_path = category_path / action / img_path.name
                        
                        # Create destination folder if needed
                        dest_path.parent.mkdir(exist_ok=True)
                        
                        try:
                            shutil.move(str(img_path), str(dest_path))
                            moved_count += 1
                        except Exception as e:
                            logger.error(f"Error moving {img_path}: {e}")
                
                stats[prob_level][category_name] = moved_count
        
        return stats
    
    def _check_image_against_rule(self, image_path: Path, rule: Dict) -> Optional[str]:
        """
        Check if an image violates a rule
        
        Returns:
            Action to take (folder name) or None if no violation
        """
        objects = self.detect_objects(str(image_path))
        person_count = objects.get('person', 0)
        chair_count = objects.get('chair', 0)
        
        # Check person count rules
        if 'person_count_min' in rule:
            if person_count < rule['person_count_min']:
                return "SuspectedFalsePositives"
        
        if 'person_count_max' in rule:
            if person_count > rule['person_count_max']:
                return "SuspectedFalsePositives"
        
        # Check person size for selfies
        if rule.get('check_person_size'):
            large_persons = self.check_person_size(str(image_path), 
                                                   rule.get('person_size_threshold', 0.25))
            if large_persons and not any(large_persons):
                return "SuspectedFalsePositives"
        
        # Check for groupies (crowd misclassified)
        if rule.get('check_groupies'):
            groupies_range = rule.get('groupies_threshold', [3, 8])
            if groupies_range[0] < person_count < groupies_range[1]:
                return "SuspectedGroupies"
        
        # Check for meeting/deliberation (groupies misclassified)
        if rule.get('check_meeting'):
            if chair_count > rule.get('chair_count_threshold', 2):
                return "SuspectedMeeting_Deliberation"
        
        return None
    
    def _get_image_files(self, directory: Path, 
                        extensions: List[str] = None) -> List[Path]:
        """Get all image files in a directory"""
        if extensions is None:
            extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
        
        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f'*.{ext}'))
        
        return image_files
    
    def _get_default_rules(self) -> List[Dict]:
        """Get default safety net rules"""
        return [
            {
                'category_name': '1ProtestSelfies',
                'person_count_min': 1,
                'person_count_max': 6,
                'check_person_size': True,
                'person_size_threshold': 0.25
            },
            {
                'category_name': '2Crowds',
                'person_count_min': 2,
                'check_groupies': True,
                'groupies_threshold': [3, 8]
            },
            {
                'category_name': '3Groupies',
                'person_count_min': 2,
                'check_meeting': True,
                'chair_count_threshold': 2
            },
            {
                'category_name': '8Meeting_Deliberation',
                'person_count_min': 4
            }
        ]
    
    def organize_subcategories(self, predictions_dir: Path, 
                              main_categories: List[str]) -> Dict[str, int]:
        """
        Organize subcategories into main categories
        
        Args:
            predictions_dir: Root predictions directory
            main_categories: List of main category names
        
        Returns:
            Dict with counts of moved subcategories
        """
        stats = {}
        
        for prob_level in ["HighProbability", "LowProbability"]:
            prob_dir = predictions_dir / prob_level
            if not prob_dir.exists():
                continue
            
            # Get all subdirectories
            subdirs = [d for d in prob_dir.iterdir() if d.is_dir()]
            
            for subdir in subdirs:
                subdir_name = subdir.name
                
                # Skip main categories and suspected folders
                if subdir_name in main_categories or subdir_name.startswith("Suspected"):
                    continue
                
                # Determine main category based on prefix
                prefix = subdir_name[0]
                main_cat = next((cat for cat in main_categories if cat.startswith(prefix)), None)
                
                if main_cat:
                    main_cat_path = prob_dir / main_cat
                    main_cat_path.mkdir(exist_ok=True)
                    
                    # Move subcategory into main category
                    dest_path = main_cat_path / subdir_name
                    try:
                        shutil.move(str(subdir), str(dest_path))
                        stats[f"{prob_level}/{main_cat}"] = stats.get(f"{prob_level}/{main_cat}", 0) + 1
                    except Exception as e:
                        print(f"Error moving {subdir}: {e}")
        
        return stats
