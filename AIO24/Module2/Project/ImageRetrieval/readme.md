# Project: Image Retrieval

## Overview
This is a small project focusing on **image retrieval**, a fundamental task in **Information Retrieval (IR)**. The goal is to build a system that returns images similar to a given query image from a predefined dataset. Additionally, the project includes **web scraping** to collect images from **flickr.com** for dataset expansion.

## Features
- **Basic Image Retrieval**: Uses distance metrics such as **L1, L2, Cosine Similarity, and Correlation Coefficient**.
- **Advanced Image Retrieval**: Implements **CLIP (Contrastive Language-Image Pretraining) model** for feature extraction.
- **Vector Database Integration**: Utilizes **ChromaDB** for optimized image indexing and retrieval.
- **Dataset Handling**: Supports structured dataset organization and preprocessing.
- **Image Crawling**: Uses **web scraping** to collect images from **flickr.com** to expand the dataset.

## Dataset
- The dataset consists of training and test images stored in structured directories (`data/train/` and `data/test/`).
- Preprocessing includes resizing images and converting them to feature vectors.
- Additional images are collected from **flickr.com** using web scraping techniques.

## Results
- Compares different retrieval methods and displays the top similar images.
- Optimizations were made by using CLIP and ChromaDB
- Scrape images from flickr.com


