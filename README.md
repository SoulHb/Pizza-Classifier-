## Pizza Classifier 

### Overview
This documentation provides information about the Pizza Classifier project, including the data used, the methods and ideas employed, and the accuracy achieved. It also includes usage instructions and author information.


### Data
The dataset used for training and scoring is loaded with pytorch and consists images with pizza and other dishes.

[Link to the dataset on Kaggle](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-tiktok-dancing-dataset
)
## Model Architecture
The Pizza Classifier neural network model is built using the [VGG-19](https://arxiv.org/pdf/1409.1556v6.pdf)
## Accuracy
After training, the model achieved an Accuracy of 95% on the validation set.
## Usage
### Requirements

- Python 3.10

### Getting Started
Clone repository
```bash
git clone https://github.com/SoulHb/Pizza-Classifier-.git
```
Move to project folder
```bash
cd Pizza-Classifier-
```
Install dependencies
```bash
pip install -r requirements.txt
```
### Training
The model is trained on the provided dataset using the following configuration:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Number of epochs: 10

Move to src folder
```bash
cd src
```
Run train.py
```bash
python your_script.py \
    --data_path /path/to/your/dataset \
    --saved_model_path /path/to/save/models \
    --epoch 10 \
    --batch_size 32 \
    --lr 0.001 \
    --image_height 224 \
    --image_width 224
```

## Inference
To use the trained model for dishes classification , follow the instructions below:

Move to src folder
```bash
cd src
```
Run Flask api
```bash
python inference.py --saved_model_path /path/to/your/saved/model
```

Run streamlit ui
```bash
python ui.py
```

Open streamlit ui in browser
```bash
streamlit run /your_path/Pizza-Classifier-/src/ui.py
```

## Author
This Pizza Classifier project was developed by Namchuk Maksym. If you have any questions, please contact with me: namchuk.maksym@gmail.com
