# HIVD: A Framework for Efficient Image Categorization in Social Research

[![DOI](https://img.shields.io/badge/DOI-10.31885/9789528418276-blue)](http://urn.fi/URN:ISBN:978-952-84-1827-6)
[![License](https://img.shields.io/badge/License-Research-green.svg)](LICENSE)
[![University of Helsinki](https://img.shields.io/badge/University-Helsinki-blue.svg)](https://www.helsinki.fi/)

<p align="center">
  <img src="https://raw.githubusercontent.com/VasileiosMalt/HIVD-A-Framework-for-efficient-image-categorization-for-social-research/refs/heads/main/hivd_logo.png" width="200">
</p>

## Overview

This repository is a technical implementation (HIVD Classifier Webapp) of the methodological framework and computational approaches developed for completing the dissertation **"A Framework for Efficient Image Categorisation in Social Research: Addressing High Intra-Class Visual Diversity"** (University of Helsinki, 2026).

**NOTE: IT CONTAINS THE IN-CODE APPLICATION OF THE FRAMEWORK (ALONG WITH A USER INTERFACE), AND NOT THE TOTALITY OF ITS CONTRIBUTIONS WHICH WERE ALSO THEORETICAL** 

The framework attempts to address a fundamental challenge in computational social science: **how to accurately classify political imagery when images within the same theoretical category which exhibit substantially high visual diversity between each other and altogether** - a phenomenon identified as **High Intra-Class Visual Diversity (HIVD)**.

## The Challenge: High Intra-Class Visual Diversity

In political and social research, images that belong to the same conceptual category (e.g., "environmental activism", "protest imagery", "political memes") often display significant visual differences.

## The Framework In A NutShell

![The Framework's logic](https://raw.githubusercontent.com/VasileiosMalt/HIVD-A-Framework-for-efficient-image-categorization-for-social-research/refs/heads/main/framework2.png)

## Key Contributions

### Methodological Steps

1. **Class Solidification**: Forces explicit definition of visual characteristics that qualify images for specific categories
2. **Category Refinement**: Breaks broad theoretical constructs into visually coherent subgroups
3. **Subcategorization**: Addresses visual diversity while enriching theoretical understanding
4. **Object Detection Verification**: Post-classification system identifying potential misclassifications based on compositional rules
5. **Enhanced "Strong" Embedding-based Zero-Shot Classification**: Incorporates visual information from reference images to improve performance without extensive training datasets

### Research Applications

The framework has been successfully applied to:

- **Hong Kong Anti-Extradition Bill Movement**: Analysis of Pepe the Frog meme adaptations
- **European Environmental Activism**: Classification of activist imagery from Finland, France, Germany, and Portugal
- **Instagram Political Visibility**: Mapping regimes of online political visibility of activists
- **Cross-cultural Visual Activism**: Understanding how visual political participation varies across contexts

## Core Principles

1. **Ethnographic-Computational Dialogue**: Sustained interaction between computational methods and ethnographic understanding
2. **Visual-Theoretical Balance**: Computational requirements enhance theoretical clarity while ethnographic insights guide algorithmic development
3. **Contextual Sensitivity**: Maintains political and cultural context without sacrificing computational efficiency
4. **Scalability**: Applicable to large-scale visual datasets across diverse political contexts

## Use Cases

This framework is designed for researchers working on:

- Political communication and visual rhetoric
- Social movement analysis
- Meme culture and online activism
- Computational social science
- Digital ethnography
- Visual sociology and political science

## Technical Approach

### Classification Pipeline

- **Supervised Learning**: Custom trained models with ethnographically-informed categories
- **Zero-Shot Learning**: CLIP-based approaches enhanced with reference images
- **Object Detection**: YOLO-based verification systems
- **Hybrid Methods**: Combining multiple approaches for robust classification

### Key Advantages

✓ Handles high visual diversity within categories  
✓ Maintains interpretability and theoretical grounding  
✓ Computationally efficient for large datasets  
✓ Adaptable across political and cultural contexts  
✓ Combines quantitative rigor with qualitative depth  

---

# 📖 User Guide: HIVD Classifier Application

This section provides step-by-step instructions for using the HIVD Classifier software. No programming experience required.

## 📥 Installation

1. **Download** the latest release from the [Releases page](https://github.com/VasileiosMalt/HIVD-A-Framework-for-efficient-image-categorization-for-social-research/releases)
2. **Extract** the downloaded ZIP file to any location (e.g., your Desktop or Documents folder)
3. **Done!** No installation wizard needed—the application is self-contained

---

## 🚀 Quick Start (5-Minute Guide)

### Step 1: Prepare Your Images

Before launching the software, organize your images:

```
HIVD_Classifier/
├── image_dataset/        ← Put EXAMPLE images here (to teach the classifier)
├── inference_dataset/    ← Put ALL images you want to classify here
```

- **`image_dataset/`**: Add 3-5 example images for each category you want to create. These "teach" the AI what each category looks like.
- **`inference_dataset/`**: Add all the images you want the software to automatically sort.

### Step 2: Launch the Application

1. Double-click **`HIVD_Classifier.exe`**
2. A console window opens (keep it open!)
3. Your web browser automatically opens to `http://127.0.0.1:8000`

> ⏳ **First launch takes 2-5 minutes** as AI models are downloaded (~2GB). Subsequent launches are much faster.

### Step 3: Create Your Categories

1. Click the **"Class Management"** tab
2. Click **"+ New Class"**
3. Fill in the form:
   - **Class Name**: Give it a simple name (e.g., "ProtestSigns", "Crowds", "Selfies")
   - **Description**: Describe what this category looks like in plain language
   - **Reference Images**: Click "Browse Dataset" and select 3-5 example images

4. Click **"Save Class"**
5. Repeat for each category you need

### Step 4: Run the Classification

1. Click the **"Classification"** tab
2. Enter a **Job Name** (e.g., "my_first_run")
3. Check the boxes next to the categories you want to use
4. Click **"🚀 Start Classification"**
5. Watch the progress bar—classification typically takes a few seconds per image

### Step 5: View Your Results

1. Click the **"Results"** tab
2. Click **"View Details"** on your completed job
3. Browse through classified images:
   - **High Confidence**: Images the AI is confident about
   - **Low Confidence**: Images that need your manual review

### Step 6: Find Your Sorted Images

Open the **`predictions/`** folder in the HIVD_Classifier directory. Your images are organized like this:

```
predictions/
└── my_first_run/
    ├── ProtestSigns/
    │   ├── HighProbability/     ← Confidently classified images
    │   └── LowProbability/      ← Needs manual review
    ├── Crowds/
    │   ├── HighProbability/
    │   └── LowProbability/
    └── ...
```

---

## 📁 Folder Structure Explained

| Folder | What to Put Here | What Happens |
|--------|------------------|--------------|
| `image_dataset/` | Example images for each category (3-5 per category) | Used to "teach" the AI |
| `inference_dataset/` | All images you want to classify | These get sorted automatically |
| `predictions/` | *Don't touch* | Results appear here after classification |
| `classes/` | *Don't touch* | Stores your category definitions |

---

## ⚙️ Understanding the Settings

### Classification Tab Parameters

When you run a classification, you'll see several options. Here's what each one means:

#### 🏷️ **Job Name**
A unique name for this classification run (e.g., "protest_photos_march2024"). Helps you find results later.

#### ☑️ **Select Classes**
Choose which categories to use. Only checked categories will be considered during classification.

#### 🔀 **Subcategorisation Strategy**

If you've created subcategories (e.g., "Protest" → "Protest Signs", "Protest Crowd"), this controls how images are sorted:

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **POST** (Default) | First sorts into main category, then into subcategory | When subcategories are refinements |
| **PRE** | All subcategories compete equally from the start | When subcategories are distinct |

> 💡 **Not sure?** Leave it on POST—it works well for most research projects.


#### 🔄 **Augmentations per Reference**

How many variations of your example images the AI creates to learn from. 

| Value | Effect |
|-------|--------|
| **around 20** | found best for highly diverse images within the class (HIVD) |
| **more** | Overfitting possible, therefore avoid |
| **less** | Maybe not as helpful |

> 💡 **Default (20)** works well for most projects.

#### 📊 **Confidence Thresholds**

These control how the software sorts results:

```
                    Low Threshold    High Threshold
                         ↓                ↓
   0% ──────────────── 40% ──────────── 60% ──────────────── 100%
        │                    │                    │
    Rejected          Low Confidence       High Confidence
   (not sorted)      (needs review)       (auto-sorted)
```

| Threshold | Default | What It Does |
|-----------|---------|--------------|
| **High Confidence** | 60% | Images above this are "safely" classified |
| **Low Confidence** | 40% | Images between low and high need manual review |

> 💡 **Recommended**: Keep defaults (40% / 60%). Adjust only if you're getting too many false positives or negatives.

#### 🛡️ **Safety Net (Object Detection)**

An optional second check using object detection. Example: If an image is classified as "Group Selfie" but no people are detected, it's flagged for review.

> 💡 **Recommended**: Keep this ON for better accuracy.

---

## 🏷️ Creating Effective Categories

### Writing Good Descriptions

The **Description** field helps the AI understand your category. Write it as if explaining to a human assistant:

| ❌ Poor Description | ✅ Good Description |
|---------------------|---------------------|
| "Protest" | "A photograph showing people participating in a street protest, holding signs or banners, often in a crowd setting outdoors" |
| "Violence" | "Images depicting physical confrontation between people, including pushing, fighting, or people being restrained" |
| "Selfie" | "A self-portrait photograph where the person's face is prominent, often taken at arm's length or with a selfie stick" |

### Choosing Good Reference Images

Select example images that represent the **diversity** within your category:

✅ **Do include**:
- Different lighting conditions (day, night, indoor, outdoor)
- Different angles and perspectives
- Different contexts within the same category

❌ **Don't include**:
- Only perfect/ideal examples
- Blurry or unclear images
- Images that could fit multiple categories

---

## 📊 Understanding Results

### The Results Tab

After classification, click **"View Details"** to see:

| Section | What It Shows |
|---------|---------------|
| **High Confidence** | Images the AI is confident about (above your high threshold) |
| **Low Confidence** | Images that scored between your low and high thresholds—review these manually |

### Clicking on Images

- **Click any thumbnail** → See the full-size image with its confidence score
- **Click a class name** → Filter to show only that category

### Output Folder Structure

Your sorted images appear in `predictions/[job_name]/`:

```
predictions/
└── my_job_name/
    ├── CategoryA/
    │   ├── HighProbability/           ← ✅ Confident classifications
    │   ├── LowProbability/            ← ⚠️ Needs review
    │   └── SuspectedFalsePositives/   ← 🛡️ Safety net flagged these
    └── CategoryB/
        └── ...
```

---

## 🔧 Advanced: Editing config.yaml

For power users, the `config.yaml` file allows fine-tuning:

```yaml
# Change the AI model (see Technical Approach section for options)
clip:
  model_name: "ViT-L-14"    # More accurate but slower
  pretrained: "openai"
  device: "cuda"             # Use "cpu" if you don't have an NVIDIA GPU

# Adjust classification behavior
classification:
  augmentations: 50                    # Increase for better accuracy
  high_probability_threshold: 0.7      # Stricter high confidence
  low_probability_threshold: 0.3       # More permissive low confidence

# Change server port if 8000 is already in use
server:
  port: 8080
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| **"First launch is very slow"** | Normal! AI models (~2GB) are downloading. Wait 2-5 minutes. |
| **"Browser doesn't open automatically"** | Manually go to `http://127.0.0.1:8000` in your browser |
| **"Port 8000 is already in use"** | Edit `config.yaml` and change `server.port` to 8080 or another number |
| **"Out of memory error"** | Close other applications, or reduce `augmentations` in settings |
| **"Classification is very slow"** | Use a smaller CLIP model (RN101), reduce augmentations, or process fewer images |
| **"Poor accuracy"** | Add more diverse reference images, improve class descriptions, or increase augmentations |
| **"GPU not being used"** | Ensure you have an NVIDIA GPU with CUDA installed. The app auto-falls back to CPU otherwise. |

---

## 🎓 Methodological Best Practices

Based on the research framework, follow these steps for best results:

### 1. Class Solidification
Don't assume your categories are obvious to the AI. Explicitly define what **visual features** make an image belong to a category.

### 2. Category Refinement  
Test your definitions with a small sample. If a category like "Violence" produces inconsistent results, refine it to something more visually concrete like "Physical Confrontation."

### 3. Subcategorisation
Break broad categories into visually coherent subgroups. Instead of one "Protest" category, create:
- "Protest Signs" (focus on banners/signs)
- "Protest Crowds" (focus on large groups)
- "Protest Selfies" (focus on individuals at protests)

This transforms one difficult classification task into several easier ones.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@phdthesis{maltezos2026hivd,
  author = {Maltezos, Vasileios},
  title = {A Framework for Efficient Image Categorisation in Social Research: Addressing High Intra-Class Visual Diversity},
  school = {University of Helsinki},
  year = {2026},
  series = {Dissertationes Universitatis Helsingiensis},
  number = {27/2026},
  isbn = {978-952-84-1827-6},
  url = {http://urn.fi/URN:ISBN:978-952-84-1827-6}
}
```

## Related Publications

This dissertation is article-based and draws from multiple peer-reviewed publications addressing different aspects of the HIVD challenge.

## Author

**Vasileios Maltezos**  
Faculty of Social Sciences  
University of Helsinki  

## Acknowledgments

This research was conducted at the Faculty of Social Sciences, University of Helsinki, as part of the Dissertationes Universitatis Helsingiensis series (27/2026).

## Links

- [Full Dissertation (HELDA Repository)](https://helda.helsinki.fi/items/9acc572a-1028-49a2-b7c0-091577d148c3)
