# User Guide

## 1. Get the project

Clone the public GitHub repository after you create/push it:

```bash
git clone YOUR_PUBLIC_REPOSITORY_URL
cd Bharatanatyam-Mudra-Recognition
```

## 2. Create a Python environment

Use Python 3.11 for the simplest setup.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## 3. Install packages

```bash
pip install -r requirements.txt
```

## 4. Check the dataset

The supplied dataset is already included as:

```text
data/mudra.zip
```

The notebook extracts it automatically into:

```text
data/Bharatanatyam_Mudras/
```

There is no need to manually rename the class folders.

## 5. Run the notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
Bharatanatyam_Mudra_Recognition.ipynb
```

### Fast review mode

The default setting is:

```python
TRAIN_FROM_SCRATCH = False
```

This loads the already trained model and saved results, so you can review the complete analysis without retraining.

### Retraining mode

Change it to:

```python
TRAIN_FROM_SCRATCH = True
```

Then run the notebook from top to bottom. The notebook will rebuild the split, preprocessing, augmentation, CNN and hyperparameter experiments, and save the new model to:

```text
model/mudra_cnn.pth
```

## 6. Run the Streamlit application

From the project root:

```bash
streamlit run app.py
```

The browser will open the application.

### Home

Shows the project overview and the five supported mudras.

### Predict

1. Click the **Predict** page.
2. Upload a JPG, JPEG or PNG image.
3. The app displays the uploaded image.
4. The app shows the predicted mudra.
5. It also shows the confidence percentage and probabilities for all five classes.
6. When confidence is below 70%, the app asks for a clearer image.

### About

Shows dataset details, model settings and the evaluation summary.

## 7. Recommended test images

Start with one of the original dataset images to confirm the installation works. For example, any image under:

```text
Bharatanatyam_Mudras/Musti/
Bharatanatyam_Mudras/Pataka/
Bharatanatyam_Mudras/Sikhara/
Bharatanatyam_Mudras/Simhamukha/
Bharatanatyam_Mudras/Trisula/
```

Then try a new photograph from your phone. The second test is more useful because it shows whether the model generalizes outside the very consistent training dataset.

## 8. What to explain during evaluation/viva

The key points are:

- The dataset is balanced: 400 images per class.
- The split is stratified: 70% train, 15% validation, 15% test.
- Mild augmentation is used to make training images more varied.
- The custom CNN has three convolution blocks.
- Learning rate and dropout were tuned.
- Clean test accuracy is 100% on the supplied split.
- The clean confusion matrix has no errors.
- An additional occlusion stress test gives about 67% accuracy and exposes real failure cases.
- High clean accuracy is partly explained by the very consistent dataset appearance.

## 9. GitHub submission

From the project folder:

```bash
git init
git add .
git commit -m "Add Bharatanatyam mudra recognition project"
git branch -M main
git remote add origin YOUR_PUBLIC_GITHUB_REPOSITORY_URL
git push -u origin main
```

After pushing, make the repository **Public** and submit only that GitHub repository URL according to the assignment instructions.
