# Traffic Police Gesture Recognition with Enhanced Pose Processing (Convolution Pose Machine)

## Overview
This project focuses on recognizing traffic police gestures using pose estimation techniques. A Convolution Pose Machine (CPM)-inspired architecture is implemented to enhance gesture recognition performance. The project also compares CPM-based models with vanilla architectures like VGG16 and ResNet50 to evaluate performance differences.

## Project Structure
|-- cpm_inspired_vgg16_model.py # Implementation of CPM-inspired VGG16 model |-- cpm_cm_VGG16.png # Confusion matrix for CPM-inspired VGG16 |-- cpm_graph_VGG16.png # Training/validation accuracy graph for CPM-inspired VGG16 |-- testing.py # Script to test trained models |-- vanilla_vgg16_and_resnet50.py # Script to train vanilla VGG16 and ResNet50 models |-- vanilla_resnet50_4_layer_train.png # Accuracy plot (4 layers) for vanilla ResNet50 |-- vanilla_resnet50_8_layer_train.png # Accuracy plot (8 layers) for vanilla ResNet50 |-- vanilla_vgg16_4_layer_train.png # Accuracy plot (4 layers) for vanilla VGG16 |-- vanilla_vgg16_8_layer_train.png # Accuracy plot (8 layers) for vanilla VGG16


## Results

### Vanilla VGG16
- **4 Layers Trained**  
  - Training Accuracy: **87%**  
  - Test Accuracy: **60%**
- **8 Layers Trained**  
  - Training Accuracy: **82%**  
  - Test Accuracy: **51%**

### Vanilla ResNet50
- **4 Layers Trained**  
  - Training Accuracy: **41%**  
  - Test Accuracy: **37%**
- **8 Layers Trained**  
  - Training Accuracy: **45%**  
  - Test Accuracy: **37%**

### CPM-Inspired VGG16
- **Training Accuracy**: **67.6%**  
- **Test Accuracy**: **58.5%**  
- **Advantages**: Improved generalization and reduced overfitting compared to vanilla architectures.

## Key Insights
1. Vanilla VGG16 achieves better accuracy compared to ResNet50 for traffic gesture recognition.
2. CPM-inspired VGG16 effectively balances training and test accuracy, reducing overfitting while maintaining competitive performance.

## Visualizations
- **Confusion Matrix**: ![cpm_cm_VGG16.png](cpm_cm_VGG16.png)
- **Training Graph**: ![cpm_graph_VGG16.png](cpm_graph_VGG16.png)

## How to Run
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Train models using the respective scripts:
   - Vanilla Models: `vanilla_vgg16_and_resnet50.py`
   - CPM-Inspired Model: `cpm_inspired_vgg16_model.py`
4. Test models with the `testing.py` script.

## Future Work
- Explore additional pose estimation models to improve accuracy.
- Fine-tune CPM architecture for real-time, resource-constrained environments.



