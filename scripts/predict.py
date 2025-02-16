import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Carregar o modelo a partir do caminho correto
model_path = 'models/cats_vs_dogs_mobilenetv2.keras'
model = tf.keras.models.load_model(model_path)

def predict_image(image_path):
    # Carregar imagem
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Realizar predição
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Calcular a confiança e classificar
    confidence = prediction if prediction > 0.5 else 1 - prediction
    class_name = "Cachorro" if prediction > 0.5 else "Gato"
    
    # Mostrar imagem e resultado da predição
    plt.imshow(img)
    plt.title(f"Predição: {class_name}\nConfiança: {confidence:.2%}")
    plt.axis('off')
    plt.show()
    
    return class_name, confidence

# Exemplo de uso - Caminho das imagens na pasta 'c'
image_path = os.path.join('c', 'sua_imagem.jpg')  # Substitua 'sua_imagem.jpg' pelo nome da imagem
class_name, confidence = predict_image(image_path)
print(f"Classe prevista: {class_name}, Confiança: {confidence:.2%}")

