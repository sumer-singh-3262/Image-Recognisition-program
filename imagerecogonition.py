import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.applications.MobileNetV2()

# Load the image
img = tf.keras.preprocessing.image.load_img('image.jpeg', target_size=(224, 224))

# Convert the image to a numpy array
img_array = tf.keras.preprocessing.image.img_to_array(img)

# Expand the dimensions to match the input shape of the model
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image
img_processed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# Make a prediction
prediction = model.predict(img_processed)

# Decode the prediction
predicted_label = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=1)[0][0]

# Print the predicted label
print("The image is most likely a " + predicted_label[1] + " with a confidence of " + str(predicted_label[2]))

# Display the image
plt.imshow(img)
plt.show()
