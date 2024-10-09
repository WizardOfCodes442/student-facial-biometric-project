# utils/cnn_model.py
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model

class CNNModel:
    _instance = None
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mobilenet_v2_model.h5')

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if CNNModel._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            CNNModel._instance = self
            self.model = self._build_model()
    
    def _build_model(self):
        try:
            # First, try to load a local saved model if it exists
            if os.path.exists(self.MODEL_PATH):
                print(f"Loading model from {self.MODEL_PATH}")
                return load_model(self.MODEL_PATH)
        except Exception as e:
            print(f"Error loading local model: {str(e)}")

        try:
            print("Building new model...")
            # Build the model without pre-trained weights first
            base_model = MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights=None
            )
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(128, activation='softmax')
            ])

            # Try to load pre-trained weights
            try:
                print("Loading pre-trained weights...")
                base_model.load_weights(tf.keras.applications.mobilenet_v2.WEIGHTS_PATH_NO_TOP)
                print("Pre-trained weights loaded successfully")
            except Exception as e:
                print(f"Could not load pre-trained weights: {str(e)}")
                print("Continuing with randomly initialized weights")

            # Save the model for future use
            try:
                model.save(self.MODEL_PATH)
                print(f"Model saved to {self.MODEL_PATH}")
            except Exception as e:
                print(f"Could not save model: {str(e)}")

            return model

        except Exception as e:
            print(f"Error building model: {str(e)}")
            # Fallback to a simpler model if everything else fails
            return self._build_fallback_model()

    def _build_fallback_model(self):
        print("Building fallback model...")
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='softmax')
        ])
        return model