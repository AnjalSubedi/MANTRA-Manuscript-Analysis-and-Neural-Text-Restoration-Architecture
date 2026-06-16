# MANTRA-Synth100k

This folder contains the Stage 1 synthetic Devanagari dataset for MANTRA.

## Purpose

The dataset is used to pretrain the HTR model before transfer learning on printed and historical Devanagari manuscript data.

## Structure

data/stage1_synthetic/
├── text/
│   ├── raw/
│   ├── cleaned/
│   └── generated/
├── fonts/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
├── metadata/
├── reports/
└── corpus_manifest.csv

## Dataset Target

Total images: 100,000

Train: 80,000  
Validation: 10,000  
Test: 10,000  

## Status

Phase 1: dataset specification and setup.
