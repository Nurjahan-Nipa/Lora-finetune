# LoRA Fine-tuning of Dolly-3B for Instruction Following

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Parameter-efficient fine-tuning of Dolly-3B using LoRA (Low-Rank Adaptation) to improve instruction-following capabilities while maintaining computational efficiency.

## 🎯 Overview

This project demonstrates how to fine-tune a large language model (Dolly-3B, 2.8B parameters) using LoRA technique, training only **2.93% of parameters** while achieving significant performance improvements in instruction-following tasks.

### Key Results
- **Training Time**: ~6 minutes (vs hours for full fine-tuning)
- **Training Loss**: 65% reduction (0.59 → 0.21)
- **Parameters Trained**: Only 83.8M out of 2.8B parameters
- **Memory Efficient**: 8-bit quantization + LoRA adapters

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
datasets>=2.12.0
bitsandbytes>=0.39.0
accelerate>=0.20.0
```

### Usage
1. **Data Loading & Preparation**
   ```bash
   python load_data.py
   ```

2. **Model & Tokenizer Setup**
   ```bash
   python prepare_model.py
   ```

3. **Data Preprocessing**
   ```bash
   python preprocess_data.py
   ```

4. **LoRA Training**
   ```bash
   python train_lora.py
   ```

5. **Inference & Evaluation**
   ```bash
   python run_inference.py
   ```

## 📁 Project Structure

```
├── load_data.py          # Data loading and preparation
├── prepare_model.py      # Model and tokenizer setup
├── preprocess_data.py    # Data preprocessing and tokenization
├── train_lora.py         # LoRA fine-tuning script
├── run_inference.py      # Inference and evaluation
├── requirements.txt      # Python dependencies
├── dolly-3b-lora/       # Saved LoRA adapters (generated)
└── README.md
```

## 🔬 Technical Details

### LoRA Configuration
- **Rank (r)**: 256
- **Alpha**: 512
- **Target Modules**: query_key_value layers
- **Dropout**: 0.1

### Training Configuration
- **Batch Size**: 1 (memory efficient)
- **Learning Rate**: 1e-4
- **Epochs**: 3
- **Precision**: FP16
- **Optimizer**: AdamW

### Model Details
- **Base Model**: `databricks/dolly-v2-3b`
- **Parameters**: 2.8B total, 83.8M trainable
- **Quantization**: 8-bit loading
- **Dataset**: MBZUAI/LaMini-instruction (200 samples)
link- https://huggingface.co/datasets/MBZUAI/LaMini-instruction

## 📊 Results

### Training Metrics
| Metric | Value |
|--------|-------|
| Initial Loss | 0.59 |
| Final Loss | 0.21 |
| Training Speed | 1.55 samples/sec |
| GPU Memory | ~8-12GB |

### Example Outputs

**Input**: "List 5 reasons why someone should learn to cook"

**Output**:
```
1. Health benefits - Control ingredients and nutrition
2. Budget savings - More cost-effective than dining out
3. Career opportunities - Culinary skills open job prospects
4. Time management - Meal prep saves daily time
5. Creative expression - Cooking as an artistic outlet
```

## 🛠️ Hardware Requirements

- **GPU**: CUDA-enabled GPU (8GB+ VRAM recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space for models and data

## 💡 Key Features

- **Parameter Efficiency**: Train only 2.93% of model parameters
- **Memory Optimization**: 8-bit quantization reduces memory usage by ~50%
- **Fast Training**: Complete fine-tuning in ~6 minutes
- **Modular Design**: LoRA adapters can be easily swapped or combined
- **High Quality**: Maintains performance comparable to full fine-tuning

## 🚀 Future Enhancements

- [ ] Scale to larger datasets (1000+ examples)
- [ ] Add quantitative evaluation metrics (BLEU, ROUGE)
- [ ] Implement multiple task-specific LoRA adapters
- [ ] Hyperparameter optimization
- [ ] API wrapper for production deployment
- [ ] Cross-domain transfer learning experiments

## 📈 Performance Analysis

### Cost-Benefit Comparison
| Method | Training Time | Parameters | GPU Memory | Cost |
|--------|---------------|------------|------------|------|
| Full Fine-tuning | 2-4 hours | 2.8B | 24GB+ | $50+ |
| LoRA Fine-tuning | 6 minutes | 83.8M | 8-12GB | ~$0.10 |

## 📚 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT Library](https://github.com/huggingface/peft)
- [Dolly-v2 Model](https://huggingface.co/databricks/dolly-v2-3b)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 🙏 Acknowledgments

- Hugging Face for the transformers and PEFT libraries
- Databricks for the Dolly-v2 model
-huggingface for datasets

