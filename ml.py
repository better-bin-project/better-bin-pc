import tensorflow as tf
import numpy as np

models_dict = {
    'resnet50': tf.keras.applications.resnet50.ResNet50(weights='imagenet'),
    'vgg16': tf.keras.applications.vgg16.VGG16(weights='imagenet'),
    'mobilenet': tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet'),
}

def recognize(modelname, imgpath):
    image = tf.keras.preprocessing.image.load_img(imgpath, 
        target_size=(224, 224))
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.imagenet_utils.preprocess_input(image)
    model = models_dict[modelname]
    predictions = model.predict(image)
    processed_preds = tf.keras.applications.imagenet_utils.decode_predictions(
        preds=predictions
    )

    print(f"Processed predictions: {processed_preds}")
    print('-' * 50)

    print("Prediction: ")
    print(f"  {imgpath}: {processed_preds[0][0][1]} to {processed_preds[0][0][2]}")
    return (processed_preds[0][0][1], processed_preds[0][0][2])
