"""
Classification engine using strong CLIP embeddings
Adapted from strong_embedding.py
Maltezos, V. (2026). A framework for efficient image categorisation in social research
"""

import torch
import open_clip
import albumentations as A
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Callable, Optional
from pathlib import Path
from tqdm import tqdm


# ===== Model Registry =====
# Validated model/pretrained combinations for zero-shot classification
# Format: model_name -> list of (pretrained_name, description, size_category)

CLIP_MODEL_REGISTRY = {
    # === Original OpenAI Models (Good baseline) ===
    "RN101": [
        ("openai", "Original OpenAI ResNet-101 CLIP", "small"),
    ],
    "ViT-B-16": [
        ("openai", "Original OpenAI ViT-Base", "small"),
        ("laion2b_s34b_b88k", "LAION-2B trained, excellent zero-shot", "small"),
    ],
    "ViT-B-32": [
        ("openai", "Original OpenAI ViT-Base (faster)", "small"),
        ("laion2b_s34b_b79k", "LAION-2B trained", "small"),
    ],
    "ViT-L-14": [
        ("openai", "Original OpenAI ViT-Large", "medium"),
        ("laion2b_s32b_b82k", "LAION-2B trained, strong performance", "medium"),
        ("datacomp_xl_s13b_b90k", "DataComp-XL trained", "medium"),
    ],
    
    # === State-of-the-Art Large Models ===
    "ViT-H-14": [
        ("laion2b_s32b_b79k", "ViT-Huge, excellent accuracy", "large"),
    ],
    "ViT-g-14": [
        ("laion2b_s34b_b88k", "ViT-Giant, top-tier accuracy", "xlarge"),
    ],
    "ViT-bigG-14": [
        ("laion2b_s39b_b160k", "ViT-BigG, largest CLIP model", "xlarge"),
    ],
    
    # === EVA-CLIP Models (State-of-the-art 2023-2024) ===
    "EVA02-B-16": [
        ("merged2b_s8b_b131k", "EVA02 Base - efficient & accurate", "small"),
    ],
    "EVA02-L-14": [
        ("merged2b_s4b_b131k", "EVA02 Large - excellent performance", "medium"),
    ],
    "EVA02-L-14-336": [
        ("merged2b_s6b_b61k", "EVA02 Large 336px - higher resolution", "medium"),
    ],
    "EVA02-E-14-plus": [
        ("laion2b_s9b_b144k", "EVA02 Enormous - near SOTA", "xlarge"),
    ],
    
    # === SigLIP Models (Google, 2024 - Excellent for classification) ===
    "ViT-B-16-SigLIP": [
        ("webli", "SigLIP Base - Google's improved CLIP", "small"),
    ],
    "ViT-L-16-SigLIP-256": [
        ("webli", "SigLIP Large 256px - great balance", "medium"),
    ],
    "ViT-L-16-SigLIP-384": [
        ("webli", "SigLIP Large 384px - higher resolution", "medium"),
    ],
    "ViT-SO400M-14-SigLIP": [
        ("webli", "SigLIP 400M - strong zero-shot", "large"),
    ],
    "ViT-SO400M-14-SigLIP-384": [
        ("webli", "SigLIP 400M 384px - best SigLIP", "large"),
    ],
    
    # === MetaCLIP Models (Meta, 2024) ===
    "ViT-B-16-quickgelu": [
        ("metaclip_400m", "MetaCLIP Base 400M", "small"),
        ("metaclip_fullcc", "MetaCLIP Base FullCC 2.5B", "small"),
    ],
    "ViT-L-14-quickgelu": [
        ("metaclip_400m", "MetaCLIP Large 400M", "medium"),
        ("metaclip_fullcc", "MetaCLIP Large FullCC 2.5B - excellent", "medium"),
    ],
    "ViT-H-14-quickgelu": [
        ("metaclip_fullcc", "MetaCLIP Huge FullCC 2.5B - top tier", "large"),
    ],
    
    # === DFN Models (Data Filtering Networks, 2024) ===
    "ViT-B-16": [
        ("dfn2b", "DFN Base - data-filtered training", "small"),
    ],
    "ViT-H-14-378-quickgelu": [
        ("dfn5b", "DFN Huge 378px - excellent accuracy", "large"),
    ],
}

# Recommended models for different use cases
RECOMMENDED_MODELS = {
    "fast_cpu": ("ViT-B-32", "openai", "Fast inference on CPU"),
    "balanced": ("ViT-L-14", "openai", "Good balance of speed and accuracy"),
    "accurate": ("EVA02-L-14", "merged2b_s4b_b131k", "High accuracy, reasonable speed"),
    "best_accuracy": ("ViT-SO400M-14-SigLIP-384", "webli", "State-of-the-art accuracy"),
    "best_large": ("EVA02-E-14-plus", "laion2b_s9b_b144k", "Maximum accuracy (requires good GPU)"),
}


def get_available_models() -> Dict[str, List[Dict]]:
    """Return all available model configurations for the UI"""
    result = {}
    for model_name, pretrained_list in CLIP_MODEL_REGISTRY.items():
        result[model_name] = [
            {"pretrained": p[0], "description": p[1], "size": p[2]}
            for p in pretrained_list
        ]
    return result


def validate_model_config(model_name: str, pretrained: str) -> bool:
    """Check if a model/pretrained combination is valid"""
    if model_name not in CLIP_MODEL_REGISTRY:
        return False
    valid_pretrained = [p[0] for p in CLIP_MODEL_REGISTRY[model_name]]
    return pretrained in valid_pretrained


class ImageAugmentor:
    """Handles image augmentation for creating strong embeddings"""
    
    def __init__(self, preprocess, device: str, augmentations: int = 100):
        self.preprocess = preprocess
        self.device = device
        self.augmentations = augmentations
        self.aug_embeddings = None
        
        # Define augmentation pipeline
        self.data_transform = A.Compose([
            A.OneOf([
                A.Compose([
                    A.SmallestMaxSize(max_size=300, p=1),
                    A.RandomCrop(width=224, height=224, p=1)
                ], p=1),
                A.HorizontalFlip(p=1),
                A.Rotate(limit=60, p=1),
            ], p=0.5),
            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), 
                         saturation=(0.8, 1.2), hue=(-0.1, 0.1), p=1),
            A.GaussNoise(var_limit=(0.0, 50.0), p=1),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=1),
            A.ToGray(p=0.1)
        ])
    
    def augment_images(self, image_paths: List[str], model) -> torch.Tensor:
        """
        Generate augmented embeddings for reference images
        Returns: tensor of shape (num_images, num_augmentations, embedding_dim)
        """
        all_embeddings = []
        
        for img_path in tqdm(image_paths, desc="Generating augmented embeddings"):
            image_array = np.asarray(Image.open(img_path).convert('RGB'))
            
            aug_embeddings = []
            for _ in range(self.augmentations):
                # Apply augmentation
                augmented = self.data_transform(image=image_array)['image']
                aug_image = Image.fromarray(np.uint8(augmented))
                
                # Process and encode
                with torch.no_grad():
                    image_tensor = self.preprocess(aug_image).unsqueeze(0).to(self.device)
                    embedding = model.encode_image(image_tensor)
                    aug_embeddings.append(embedding)
            
            all_embeddings.append(torch.cat(aug_embeddings))
        
        self.aug_embeddings = torch.stack(all_embeddings)
        return self.aug_embeddings
    
    def get_dimension_filter(self, random_augmentations: int = 50, 
                            std_threshold: int = 2) -> torch.Tensor:
        """
        Compute dimension filter based on embedding volatility
        Returns: boolean tensor indicating which dimensions to keep
        """
        if self.aug_embeddings is None:
            raise ValueError("Must call augment_images first")
        
        # Sample random augmentations and compute standard deviation
        num_augs = self.aug_embeddings.shape[1]
        random_indices = torch.randperm(num_augs)[:random_augmentations]
        
        deviations = self.aug_embeddings[:, random_indices, :].std(1).mean(0)
        threshold = torch.median(deviations) + torch.std(deviations) * std_threshold
        
        # Filter dimensions
        keep_mask = deviations < threshold
        num_kept = torch.sum(keep_mask).item()
        total_dims = len(keep_mask)
        
        print(f"Dimension filter: keeping {num_kept}/{total_dims} dimensions ({num_kept/total_dims:.1%})")
        
        return keep_mask


class StrongEmbeddingClassifier:
    """
    Zero-shot classifier using strong CLIP embeddings enhanced with reference images.
    
    Supports multiple CLIP model architectures including:
    - Original OpenAI models (RN101, ViT-B-16, ViT-L-14)
    - EVA-CLIP models (EVA02 series - state-of-the-art 2023-2024)
    - SigLIP models (Google's improved CLIP - excellent for classification)
    - MetaCLIP models (Meta's curated training)
    - Large models (ViT-H-14, ViT-g-14, ViT-bigG-14)
    
    See CLIP_MODEL_REGISTRY for all valid model/pretrained combinations.
    """
    
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai",
                 device: str = None):
        """
        Initialize classifier with CLIP model
        
        Args:
            model_name: CLIP model architecture (see CLIP_MODEL_REGISTRY for options)
            pretrained: Pretrained weights (must match model - see CLIP_MODEL_REGISTRY)
            device: Device to use (cuda, cpu, mps). Auto-detects if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        
        # Caches
        self.reference_embeddings = None
        self.dimension_filter = None
        self.class_names = []
    
    def prepare_reference_embeddings(
        self, 
        class_references: Dict[str, List[str]],
        class_descriptions: Optional[Dict[str, str]] = None,
        num_augmentations: int = 100,
        random_augmentations: int = 50,
        std_threshold: float = 2.0,
        progress_callback: Optional[Callable] = None
    ):
        """
        Prepare reference embeddings for classification
        
        Args:
            class_references: Dict mapping class names to list of reference image paths
            num_augmentations: Number of augmentations per reference image
            random_augmentations: Number of random augmentations for dimension filtering
            std_threshold: Standard deviation threshold for filtering
            progress_callback: Optional callback for progress updates
        """
        print("\n" + "="*70)
        print("PREPARING REFERENCE EMBEDDINGS")
        print("="*70)
        
        self.class_names = list(class_references.keys())
        all_reference_paths = []
        
        # Validate and collect reference image paths
        print(f"\nClasses to prepare: {len(self.class_names)}")
        for class_name in self.class_names:
            refs = class_references.get(class_name, [])
            desc = class_descriptions.get(class_name, class_name) if class_descriptions else class_name
            
            print(f"\n  [{class_name}]")
            print(f"    Description: {desc[:60]}...")
            print(f"    Reference images: {len(refs)}")
            
            valid_refs = []
            for ref_path in refs:
                if Path(ref_path).exists():
                    valid_refs.append(ref_path)
                    print(f"      ? {Path(ref_path).name}")
                else:
                    print(f"      ? NOT FOUND: {ref_path}")
            
            if valid_refs:
                all_reference_paths.extend(valid_refs)
                print(f"    -> Will use {len(valid_refs)} valid reference images")
            else:
                print(f"    -> TEXT-ONLY EMBEDDING (no valid images)")
        
        print(f"\nTotal reference images to process: {len(all_reference_paths)}")
        
        # Generate augmented embeddings for image-based classes
        if all_reference_paths:
            augmentor = ImageAugmentor(self.preprocess, self.device, num_augmentations)
            
            if progress_callback:
                progress_callback(f"Generating augmented embeddings for {len(all_reference_paths)} reference images...")
            
            print(f"\nGenerating {num_augmentations} augmentations per image...")
            aug_embeddings = augmentor.augment_images(all_reference_paths, self.model)
            print(f"Augmented embeddings shape: {aug_embeddings.shape}")
            
            # Calculate dimension filter (exclude divergent augmentations)
            print("\nCalculating divergent augmentation filter...")
            self.dimension_filter = augmentor.get_dimension_filter(
                random_augmentations=random_augmentations,
                std_threshold=std_threshold
            )
            
        else:
            # No image references at all - all classes are text-only
            print("\n⚠️  WARNING: No valid reference images found! All classes will be text-only.")
            self.dimension_filter = None
            aug_embeddings = None
        
        # Build final reference embeddings
        print("\n" + "-"*50)
        print("BUILDING CLASS EMBEDDINGS")
        print("-"*50)
        
        embeddings_per_class = []
        self.has_image_refs = []
        img_idx = 0
        
        for class_name in self.class_names:
            description = class_descriptions.get(class_name, class_name) if class_descriptions else class_name
            
            # 1. Get Text Embedding
            with torch.no_grad():
                text_tokenized = open_clip.tokenize([description]).to(self.device)
                text_embedding = self.model.encode_text(text_tokenized).squeeze(0)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            
            # 2. Get Image Embedding if available
            refs = class_references.get(class_name, [])
            valid_refs = [r for r in refs if Path(r).exists()]
            
            if valid_refs and aug_embeddings is not None:
                num_refs = len(valid_refs)
                class_augs = aug_embeddings[img_idx:img_idx + num_refs]
                image_embedding = class_augs.mean(dim=(0, 1))
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                img_idx += num_refs
                
                # COMBINED: 50% Text + 50% Image
                combined_embedding = (text_embedding + image_embedding) / 2.0
                combined_embedding /= combined_embedding.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                embeddings_per_class.append(combined_embedding)
                self.has_image_refs.append(True)
                print(f"  [{class_name}] -> COMBINED (text + {num_refs} images)")
            else:
                # TEXT-ONLY
                embeddings_per_class.append(text_embedding)
                self.has_image_refs.append(False)
                print(f"  [{class_name}] -> TEXT-ONLY")
        
        self.reference_embeddings = torch.stack(embeddings_per_class)
        self.reference_embeddings /= self.reference_embeddings.norm(dim=-1, keepdim=True)
        
        # Count types
        n_combined = sum(self.has_image_refs)
        n_text_only = len(self.has_image_refs) - n_combined
        
        print("\n" + "="*70)
        print(f"READY: {len(self.class_names)} class embeddings")
        print(f"  Combined (text+image): {n_combined}")
        print(f"  Text-only: {n_text_only}")
        print(f"Classes: {self.class_names}")
        print("="*70 + "\n")
        
        if progress_callback:
            progress_callback(f"Reference embeddings prepared for {len(self.class_names)} classes")
    
    def classify_images(
        self,
        image_paths: List[str],
        high_threshold: float = 0.6,
        low_threshold: float = 0.4,
        batch_size: int = 32,
        progress_callback: Optional[Callable] = None
    ) -> List[Tuple[str, str, float, bool]]:
        """
        Classify images using strong embeddings
        
        Args:
            image_paths: List of image paths to classify
            high_threshold: Threshold for high confidence
            low_threshold: Threshold for low confidence
            batch_size: Batch size for processing
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of tuples: (image_path, predicted_class, probability, is_high_confidence)
        """
        if self.reference_embeddings is None:
            raise ValueError("Must call prepare_reference_embeddings first")
        
        results = []
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            # Encode batch
            batch_embeddings = []
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.model.encode_image(image_tensor)
                        batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    # Add dummy result for failed images
                    results.append((img_path, "ERROR", 0.0, False))
                    continue
            
            if not batch_embeddings:
                continue
            
            batch_embeddings = torch.cat(batch_embeddings)
            batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True)
            
            # Calculate similarities using optimal dimensions (if filter exists)
            # This implements the "exclusion of divergent augmentations" logic
            if self.dimension_filter is not None:
                # Ensure filter is on correct device
                if self.dimension_filter.device != self.device:
                    self.dimension_filter = self.dimension_filter.to(self.device)
                
                # Apply filter to both batch and references
                # Note: We do NOT re-normalize. This effectively downweights images/classes 
                # that relied heavily on unstable features.
                filtered_batch = batch_embeddings[:, self.dimension_filter]
                filtered_refs = self.reference_embeddings[:, self.dimension_filter]
                
                raw_similarities = (filtered_batch @ filtered_refs.T)
            else:
                raw_similarities = (batch_embeddings @ self.reference_embeddings.T)
            
            # CRITICAL: Normalize similarities to make text-only classes competitive
            # Text-only classes systematically get lower cosine similarity in CLIP
            # We normalize by boosting text-only scores to match the typical range of combined scores
            similarities = raw_similarities.clone()
            
            # Compute mean similarity for combined vs text-only classes
            combined_indices = [i for i, has_img in enumerate(self.has_image_refs) if has_img]
            text_only_indices = [i for i, has_img in enumerate(self.has_image_refs) if not has_img]
            
            if combined_indices and text_only_indices:
                # Calculate the gap between combined and text-only similarities
                combined_mean = raw_similarities[:, combined_indices].mean().item()
                text_only_mean = raw_similarities[:, text_only_indices].mean().item()
                gap = combined_mean - text_only_mean
                
                # Boost text-only classes to close the gap (make them competitive)
                if gap > 0.05:  # Only boost if there's a meaningful gap
                    boost = gap * 0.8  # Close 80% of the gap
                    for idx in text_only_indices:
                        similarities[:, idx] += boost
            
            # Log similarities for first batch
            if batch_idx == 0 and len(batch_paths) > 0:
                print("\n" + "="*70)
                print("SIMILARITY SCORES (first image):")
                print("="*70)
                for cls_idx, cls_name in enumerate(self.class_names):
                    raw_sim = raw_similarities[0, cls_idx].item()
                    adj_sim = similarities[0, cls_idx].item()
                    marker = "??" if self.has_image_refs[cls_idx] else "??"
                    if raw_sim != adj_sim:
                        print(f"  {marker} {cls_name}: {raw_sim:.4f} ? {adj_sim:.4f} (boosted)")
                    else:
                        print(f"  {marker} {cls_name}: {raw_sim:.4f}")
                print("="*70 + "\n")
            
            # Convert to probabilities with moderate temperature
            # 50.0 gives good discrimination without being too extreme
            temperature = 50.0
            probabilities = torch.softmax(similarities * temperature, dim=1)
            
            # Get predictions
            max_probs, predicted_indices = probabilities.max(dim=1)
            
            for i, img_path in enumerate(batch_paths):
                if i < len(predicted_indices):
                    idx = predicted_indices[i]
                    predicted_class = self.class_names[idx]
                    prob_val = max_probs[i].item()
                    is_high_confidence = prob_val >= high_threshold
                    
                    # Log classification result
                    conf_marker = "?" if is_high_confidence else "?"
                    print(f"\n  {conf_marker} [{Path(img_path).name}] -> {predicted_class} ({prob_val:.1%})")
                    sorted_probs = sorted(
                        zip(self.class_names, probabilities[i].tolist()),
                        key=lambda x: -x[1]
                    )
                    for cls_name, cls_prob in sorted_probs[:4]:  # Top 4 only
                        marker = "?" if cls_name == predicted_class else " "
                        type_marker = "??" if self.has_image_refs[self.class_names.index(cls_name)] else "??"
                        print(f"    {marker} {type_marker} {cls_name}: {cls_prob:.1%}")
                    
                    results.append((img_path, predicted_class, prob_val, is_high_confidence))
            
            if progress_callback:
                progress = (end_idx / len(image_paths)) * 100
                progress_callback(f"Processed {end_idx}/{len(image_paths)} images ({progress:.1f}%)")
        
        return results
    
    def encode_single_image(self, image_path: str) -> torch.Tensor:
        """Encode a single image to embedding"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding /= embedding.norm(dim=-1, keepdim=True)
        
        return embedding
