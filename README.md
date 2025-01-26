
# 🐱 vs 🐶 Classifier - Transfer Learning com MobileNetV2

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Classificador de imagens para gatos e cachorros usando Transfer Learning com MobileNetV2.

## 📋 Sumário
- [Instalação](#instalação)
- [Uso](#uso)
- [Treinamento](#treinamento)
- [Contribuição](#contribuição)
- [Licença](#licença)

## 🛠️ Instalação
```bash
git clone https://github.com/RailsonDataExplorer/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
pip install -r requirements.txt
```

## 🚀 Uso
```python
from src.model_training import predict_image

result = predict_image("path/to/image.jpg")
print(f"Predição: {result['class']} (Confiança: {result['confidence']:.2%})")
```

## 🧠 Treinamento
```bash
python src/model_training.py \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0001
```

## 🤝 Contribuição
1. Faça um fork do projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença
Este projeto está licenciado sob a Licença MIT - veja [LICENSE](LICENSE) para detalhes.

## 🙏 Reconhecimentos
- Dataset: [Microsoft Cats vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- Modelo Base: [MobileNetV2](https://arxiv.org/abs/1801.04381)
