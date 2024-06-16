
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import os
import tempfile
from predict_class import predict_class

app = Flask(__name__)

# Carregue o modelo na inicialização do servidor
model = load_model('vgg16_fine_tuned.h5')

@app.route('/', methods=['GET'])
def get():
    return "Backend rodando"

@app.route('/api/test', methods=['GET'])
def get_img_class_test():
    query_image_path = 'images/test/class1-deforestation/test_22.jpg'
    predicted_class = predict_class(model, query_image_path)
    return jsonify({'predicted_class': predicted_class})

@app.route('/api/prediction', methods=['POST'])
def get_img_class():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    try:
        img = Image.open(BytesIO(file.read()))  # Lê a imagem

        # Salva a imagem em um arquivo temporário
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        img.save(temp_file, format='JPEG')
        temp_file_path = temp_file.name
        temp_file.close()

        predicted_class = predict_class(model, temp_file_path)

        # Remove o arquivo temporário após a predição
        os.remove(temp_file_path)

        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)