import sys

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

def predict_genres(model, file):
    # Load the trained model

    # Load and preprocess the images
    img = image.load_img(file, target_size=(100, 100),  color_mode='rgb')
    img = image.img_to_array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model.predict(img, batch_size=1, verbose=1)

    # Get the expected genres
    genres = ["action", "adventure", "animation", "comedy", "crime", "drama", "fantasy", "horror", "mystery", "romance", "sci-fi", "short"]
    expected_genres = []
    most = 0
    for prediction in predictions:

        expected_genre_indices = np.where(prediction >= 0.30)[0]  # Threshold the predictions at 0.5
        expected_genre_labels = [genres[index] for index in expected_genre_indices]
        expected_genres.append(expected_genre_labels)

    # Print the expected genres for each image
    print(f"Image: {file}")
    print(f"Expected Genres: {expected_genres}")
    print("")

# Example usage
model_name = "movieclassifyer"
model = load_model(model_name,compile=False)
model.compile( optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6),
              loss = ["binary_crossentropy","binary_crossentropy","binary_crossentropy",
                      "binary_crossentropy","binary_crossentropy","binary_crossentropy",
                      "binary_crossentropy","binary_crossentropy","binary_crossentropy",
                      "binary_crossentropy","binary_crossentropy","binary_crossentropy"],
              metrics=['accuracy'])

sys.argv.pop(0)
spot = 0
for val in sys.argv:
    #predict the outcome with the model
    if (os.path.exists(val)):
        predict_genres(model, val)

#for im in image_files:
 #   predict_genres(model, im)
    #print the prediction result