# MedicalGPT: Transformer-based Medical Language Model

## Overview

MedicalGPT is a Natural Language Processing (NLP) project that develops a GPT-style transformer language model for medical text. The project includes data preprocessing, custom tokenizer training, transformer model development, experimentation with different model configurations, training, evaluation, and deployment through an interactive Streamlit web application.

The objective is to demonstrate the complete workflow of building a domain-specific language model capable of understanding and generating medical text.

---

## Features

- Custom medical text preprocessing pipeline
- SentencePiece tokenizer for medical vocabulary
- GPT-style Transformer architecture
- Multiple training experiments with different hyperparameters
- Model checkpoint management
- Performance evaluation and training visualization
- Interactive Streamlit-based web interface
- Configurable model architecture

---

## Technologies Used

- Python
- PyTorch
- SentencePiece
- Streamlit
- NumPy
- JSON
- Jupyter Notebook

---

## Project Structure

```
NLP-main/
│── experiments/
│   ├── exp1_heads4_layers6/
│   ├── exp2_heads4_layers8/
│   ├── exp3_d512_heads4_layers8/
│   └── exp4_d512_heads2_layers8/
│
│── notebooks/
│   ├── step1_data_cleaning.ipynb
│   ├── step2_tokenizer.ipynb
│   ├── step3_prepare_sequences.ipynb
│   ├── step4_model.ipynb
│   ├── step5_train.ipynb
│   └── step6.ipynb
│
│── tokenizer/
│   ├── medical_bpe.model
│   ├── medical_bpe.vocab
│   └── config.json
│
│── ui/
│   └── app.py
```

---

## Workflow

1. Clean and preprocess medical text.
2. Train a custom SentencePiece tokenizer.
3. Prepare token sequences for model training.
4. Build the GPT-style transformer architecture.
5. Train the model using different hyperparameter settings.
6. Evaluate model performance.
7. Deploy the trained model using a Streamlit interface.

---

## Experimental Configurations

The project includes multiple experiments by varying:

- Number of Transformer layers
- Number of attention heads
- Embedding dimensions
- Training configurations

Each experiment stores:

- Configuration files
- Training logs
- Evaluation metrics
- Loss curves
- Model checkpoints

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sohanreddy1717-gif/NLP-Project.git
```

Move into the project directory:

```bash
cd NLP-Project
```

Install dependencies:

```bash
pip install torch streamlit sentencepiece numpy
```

Run the application:

```bash
streamlit run ui/app.py
```

---

## Results

The project evaluates different transformer architectures by comparing training performance, loss curves, and evaluation metrics across multiple experimental settings to identify an effective configuration for medical text generation.

---

## Future Improvements

- Fine-tune on larger medical datasets
- Implement Beam Search and Top-k Sampling
- Support conversational medical question answering
- Deploy using Docker
- Integrate Retrieval-Augmented Generation (RAG)
- Improve inference speed using model optimization

---

## Author

**Sohan Reddy Vadiyala**

GitHub: https://github.com/sohanreddy1717-gif
