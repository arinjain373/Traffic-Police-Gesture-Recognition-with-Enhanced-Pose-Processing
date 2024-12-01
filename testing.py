import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import os
from tqdm import tqdm

def load_test_data(test_dir, target_size=(640, 640)):
    """
    Load images from test directory structure
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(test_dir))  # Get sorted class names
    
    print("Loading test images...")
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Processing class: {class_name}")
        for img_name in tqdm(os.listdir(class_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                try:
                    # Load and preprocess image
                    img = load_img(img_path, target_size=target_size)
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0  # Normalize to [0,1]
                    
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
    
    return np.array(images), np.array(labels), class_names

def evaluate_model(model_path, test_dir, target_size=(640, 640)):
    """
    Evaluate the model on test data
    """
    # Load the model
    print("Loading model...")
    model = load_model(model_path)
    
    # Load test data
    X_test, y_test, class_names = load_test_data(test_dir, target_size)
    
    # Get predictions
    print("Making predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print summary metrics
    print("\nOverall Metrics:")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_precision = precision_score(y_test == i, y_pred == i, average='binary')
        class_recall = recall_score(y_test == i, y_pred == i, average='binary')
        class_f1 = f1_score(y_test == i, y_pred == i, average='binary')
        
        per_class_metrics[class_name] = {
            'precision': class_precision,
            'recall': class_recall,
            'f1_score': class_f1
        }
    
    return {
        'overall_metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'per_class_metrics': per_class_metrics,
        'class_names': class_names
    }

# Example usage
if __name__ == "__main__":
    test_dir = 'Traffic police gesture/Test'  # Path to your test directory
    model_path = "traffic_gesture_classifier_claude_VGG16.h5"  # Path to your saved model
    

    target_size = (640, 640)  
    # Evaluate the model
    results = evaluate_model(model_path, test_dir, target_size)