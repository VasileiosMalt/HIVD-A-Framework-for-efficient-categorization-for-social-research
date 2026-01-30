# HIVD: A Framework for Efficient Image Categorization in Social Research

[![DOI](https://img.shields.io/badge/DOI-10.31885/9789528418276-blue)](http://urn.fi/URN:ISBN:978-952-84-1827-6)
[![License](https://img.shields.io/badge/License-Research-green.svg)](LICENSE)
[![University of Helsinki](https://img.shields.io/badge/University-Helsinki-blue.svg)](https://www.helsinki.fi/)

## Overview

This repository contains the methodological framework and computational approaches developed in the dissertation **"A Framework for Efficient Image Categorisation in Social Research: Addressing High Intra-Class Visual Diversity"** (University of Helsinki, 2026).

The framework attempts to address a fundamental challenge in computational social science: **how to accurately classify political imagery when images within the same theoretical category which exhibit substantially high visual diversity between each other and altogether** - a phenomenon identified as **High Intra-Class Visual Diversity (HIVD)**.

## The Challenge: High Intra-Class Visual Diversity

In political and social research, images that belong to the same conceptual category (e.g., "environmental activism", "protest imagery", "political memes") often display significant visual differences.

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

## Framework In A NutShell

![The Framework's logic](https://raw.githubusercontent.com/VasileiosMalt/HIVD-A-Framework-for-efficient-image-categorization-for-social-research/refs/heads/main/framework2.png)

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

## License

This publication is copyrighted. You may download, display and print it for your own personal use. Commercial use is prohibited.

## Acknowledgments

This research was conducted at the Faculty of Social Sciences, University of Helsinki, as part of the Dissertationes Universitatis Helsingiensis series (27/2026).

## Links

- [Full Dissertation (HELDA Repository)](https://helda.helsinki.fi/items/9acc572a-1028-49a2-b7c0-091577d148c3)
