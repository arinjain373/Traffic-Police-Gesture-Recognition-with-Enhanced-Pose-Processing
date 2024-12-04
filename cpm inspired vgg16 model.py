import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class TrafficGestureClassifier:
    def __init__(self, input_shape=(640, 640, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_convolutional_pose_machine(self, x):
        """Build CPM stages"""
        # Stage 1
        x = Conv2D(128, 9, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, 9, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, 9, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Stage 2
        x = Conv2D(256, 7, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, 7, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Stage 3
        x = Conv2D(512, 5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, 5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        return x
        
    def _build_model(self):
        """Build the complete model architecture"""
        # Load VGG16 as feature extractor
        base_model = VGG16(weights='imagenet', include_top=False, 
                          input_shape=self.input_shape)
        
        # Freeze early layers
        for layer in base_model.layers[:15]:
            layer.trainable = False
            
        # Add CPM layers
        x = base_model.output
        x = self._build_convolutional_pose_machine(x)
        
        # Add classification layers
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model

def create_data_generators():
    """Create train and validation data generators with augmentation"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, test_datagen

def train_model(model, train_dir, test_dir, epochs=50, batch_size=32):
    """Train the model with early stopping and learning rate reduction"""
    train_datagen, test_datagen = create_data_generators()
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(640, 640),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(640, 640),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(640, 640),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    return history, test_generator

def plot_training_history(history):
    """Plot training history with loss and accuracy graphs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('cpm_vgg16.png')

def plot_confusion_matrix(model, test_generator, class_names):
    """Plot confusion matrix"""
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cpm_cm_vgg16.png')

def main():
    # Define paths
    train_dir = 'Traffic police gesture/Train'
    test_dir = 'Traffic police gesture/Test'
    
    # Define class names
    class_names = [
        'stop', 'right', 'right_turn', 'right_over',
        'move_straight', 'left', 'left_turn_1', 'left_over',
        'lane_right', 'lane_left'
    ]
    
    # Create and train model
    classifier = TrafficGestureClassifier()
    history, test_generator = train_model(
        classifier.model,
        train_dir,
        test_dir,
        epochs=50,
        batch_size=32
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(classifier.model, test_generator, class_names)
    
    # Save model
    classifier.model.save('traffic_gesture_classifier_cpm_vgg16.h5')

if __name__ == "__main__":
    main()