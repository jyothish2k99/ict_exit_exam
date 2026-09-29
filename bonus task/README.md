# Bharatanatyam Mudra Recognition — Transfer Learning Bonus

Separate bonus project comparing a simple custom CNN with MobileNetV2 transfer learning.

## Run
1. Open `Bharatanatyam_Mudra_Transfer_Learning.ipynb`.
2. Put `mudra.zip` in the same working directory, or upload it in Google Colab.
3. Run all cells from the beginning.
4. For the final submission, keep the generated comparison table and deployment recommendation.

The notebook measures:
- Accuracy
- Model size
- Training time
- Single-image inference speed

The transfer-learning model uses ImageNet-pretrained MobileNetV2 weights. The first run needs internet access to download those weights.

Google Colab with a GPU is recommended for the final run.

## Install
```bash
pip install torch torchvision scikit-learn pandas matplotlib pillow jupyter
```
