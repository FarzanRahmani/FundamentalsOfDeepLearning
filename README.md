# Fundamentals of Deep Learning

Homework, tutorial material, and sentiment-analysis experiments from my seventh-semester B.Sc. course at Iran University of Science and Technology, taught by Dr. Marzieh Davoodabadi Farahani. Author: Farzan Rahmani.

## Repository Guide

| Path | Contents |
| --- | --- |
| `HWs/HW1/` | Python/numerical foundations, activation functions, NumPy, Keras, and PyTorch notebooks |
| `HWs/HW2/`–`HWs/HW6/` | Subsequent assignments, practical notebooks, and written material |
| `HWs/HW Notes.pdf` | Supporting homework notes |
| `TA/` | PyTorch tutorial notebook |
| `project/SA.ipynb` | Sentiment-analysis workflow with preprocessing, transformer training, prediction, and evaluation |
| `project/Final files Quera/` | Submitted notebook and Python files (`SA.py`, `Predict.py`, and `Evaluate.py`) |
| Other folders under `project/` | Experiment versions exploring early stopping, regularization, validation splits, and batch sizes |

## Sentiment-Analysis Project

The project uses PyTorch and Hugging Face Transformers for sequence classification, including BERT and XLM-RoBERTa experiments. Text preparation uses libraries such as Hazm and NLTK, and training/evaluation reads TSV datasets with sentence and label columns.

Start with the project notebook to understand the complete sequence. The submitted prediction helper expects a model/tokenizer checkpoint, device, label mapping, and base path; its file alone is not a complete command-line application and relies on imports supplied by the surrounding workflow.

## Environment and Execution

Use Python with Jupyter or Google Colab. Install dependencies for the selected exercise rather than assuming one environment fits the entire course. Homework uses both TensorFlow/Keras and PyTorch; the final NLP workflow additionally imports Transformers, pandas, scikit-learn, tqdm, Hazm, NLTK, and a `cleantext` module.

Before running a notebook:

1. Review setup/import cells and choose compatible library versions.
2. Configure dataset paths, model identifiers, cache directories, and output locations.
3. Obtain required datasets, pretrained weights, and any NLTK resources separately where they are not included.
4. Run preprocessing, training, and evaluation cells in order. A GPU is recommended for transformer fine-tuning.

Some notebooks assume hosted-runtime or Google Drive paths. Model downloads require network access, and selected model sources may require authentication. Saved outputs and historical experiment folders are records of earlier runs, not a reproducibility guarantee.

## License

See [LICENSE](LICENSE) for license terms. External models and datasets retain their own terms.
