import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def build_lstm_model(
    input_shape: tuple,
    lstm_units: int = 50,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    Builds and compiles a simple LSTM model for multi-class classification.

    The architecture consists of an LSTM layer followed by Dropout for regularization,
    a hidden Dense layer, and a final Dense output layer with softmax activation
    for three classes (-1, 0, 1).

    :param input_shape: The shape of the input data (timesteps, num_features).
    :param lstm_units: The number of units in the LSTM layer.
    :param dropout_rate: The dropout rate for regularization.
    :return: A compiled Keras model.
    """
    print(f"Building LSTM model with input shape {input_shape}...")

    model = Sequential([
        Input(shape=input_shape, name="input_layer"),
        LSTM(units=lstm_units, return_sequences=False, name="lstm_layer"),
        Dropout(dropout_rate, name="dropout_layer"),
        Dense(units=lstm_units // 2, activation='relu', name="hidden_dense_layer"),
        # 3 output units for 3 classes: Loss (-1), Timeout (0), Profit (1)
        # These labels will need to be one-hot encoded before training.
        Dense(units=3, activation='softmax', name="output_layer")
    ])

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Model built and compiled successfully.")
    return model

if __name__ == '__main__':
    print("--- Testing Model Building ---")

    # Define a sample input shape (e.g., 10 timesteps, 20 features)
    sample_input_shape = (10, 20)

    # Build the model
    try:
        test_model = build_lstm_model(input_shape=sample_input_shape)

        # Print the model summary to verify the architecture
        print("\nModel Summary:")
        test_model.summary()
    except Exception as e:
        print(f"An error occurred during model building: {e}")
        print("Please ensure TensorFlow is installed correctly.")
