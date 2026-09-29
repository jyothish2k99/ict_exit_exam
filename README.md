# Bharatanatyam Mudra Recognition

A student-level machine learning project that recognizes five Bharatanatyam hand mudras from images using a small custom CNN and a Streamlit application.

Streamlit app link: https://ictexitexam-project.streamlit.app/

## Project structure

```text
Bharatanatyam-Mudra-Recognition/
├── Bharatanatyam_Mudra_Recognition.ipynb
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── data/
│   └── mudra.zip
├── model/
│   ├── mudra_cnn.pth
│   └── class_names.json
├── assets/
│   ├── class_distribution.png
│   ├── dataset_samples.png
│   ├── augmented_samples.png
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   └── misclassified_15.png
├── train_split.csv
├── val_split.csv
├── test_split.csv
├── training_history.csv
├── tuning_results.csv
├── stress_test_misclassified.csv
└── metrics.json
```

## Dataset

The supplied dataset contains **2,000 RGB images** across five classes:

| Mudra | Images |
|---|---:|
| Musti | 400 |
| Pataka | 400 |
| Sikhara | 400 |
| Simhamukha | 400 |
| Trisula | 400 |

The original images are 300×300. The notebook resizes them to 64×64 for the custom CNN. No class is downsampled because the supplied dataset is small enough to use in full.

## Model workflow

The notebook follows the assignment in one place:

1. Load and inspect the image dataset.
2. Check class distribution and image characteristics.
3. Create a stratified 70/15/15 train-validation-test split.
4. Resize and normalize images.
5. Create mild image augmentations and visualize them.
6. Build a custom CNN.
7. Tune learning rate and dropout.
8. Plot training and validation curves.
9. Evaluate accuracy, precision, recall, F1-score and confusion matrix.
10. Perform failure analysis.
11. Save the trained PyTorch model used by Streamlit.

## Results from the supplied dataset

The selected setting was:

- Learning rate: **0.001**
- Dropout: **0.30**

On the held-out clean test split:

- Accuracy: **100%**
- Precision: **1.00** for every class
- Recall: **1.00** for every class
- F1-score: **1.00** for every class
- Confusion matrix: no clean-test errors

Because the clean test set has zero errors, the notebook does not pretend that there are 15 clean misclassified images. Instead, it performs a separate **occlusion robustness stress test** on the same held-out test images. The stress test reached about **67% accuracy** and produced more than 15 incorrect predictions. The notebook displays 15 of those failure cases with actual label, predicted label and confidence.

This is an important limitation of the dataset: the supplied images are visually consistent, and the sample images contain visible hand-landmark markings. A real application should be tested on new student images collected with different cameras, lighting conditions, backgrounds and hand positions.

## Run the notebook locally

### 1. Create an environment

Python 3.11 is a good choice for this project.

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\\Scripts\\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open:

```text
Bharatanatyam_Mudra_Recognition.ipynb
```

The notebook defaults to `TRAIN_FROM_SCRATCH = False`, so it loads the included trained model and stored training history. This makes it quick to review.

To retrain the model, change:

```python
TRAIN_FROM_SCRATCH = True
```

and run the notebook from the top.

## Run the Streamlit app

From the repository root:

```bash
streamlit run app.py
```

Then open the local address shown by Streamlit in the terminal.

The app has three pages:

### Home

Shows the project purpose, the five classes and a short note about the dataset limitation.

### Predict

Upload a `.jpg`, `.jpeg` or `.png` image. The app displays:

- predicted mudra
- confidence score
- probability distribution across all five classes
- a warning when the confidence is below 70%

### About

Shows dataset details, model details and the evaluation summary.

## Testing the app

For the clearest prediction, use an image where the whole hand is visible and roughly framed like the training examples. A practice image from a different camera or environment may be less reliable than the supplied dataset.

## Public GitHub submission

The repository is already structured for GitHub. From the repository root:

```bash
git init
git add .
git commit -m "Bharatanatyam mudra recognition project"
git branch -M main
git remote add origin YOUR_PUBLIC_GITHUB_REPOSITORY_URL
git push -u origin main
```

The dataset ZIP is about 54 MB. It is below GitHub's individual-file limit, although GitHub may show a large-file warning. Git LFS can be used instead if your course repository has a size policy.

## Notes for presentation

The most important discussion point is not the 100% clean score by itself. Explain that the supplied dataset is highly consistent and that the separate occlusion test demonstrates how quickly performance can fall when key finger information is hidden. That gives a realistic error-analysis story for the project.
