# 🐱 vs 🐶 Classifier - Transfer Learning com MobileNetV2 🐾

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Classificador de imagens de gatos e cachorros utilizando **Transfer Learning** com MobileNetV2. Este projeto demonstra como aproveitar modelos pré-treinados para alcançar alta precisão mesmo com datasets limitados, além de demonstrar técnicas como aumento de dados, fine-tuning e avaliação em um conjunto de testes externo.

## 📋 Índice

1. [Visão Geral](#visao-geral)
2. [Dataset](#dataset)
3. [Requisitos](#requisitos)
4. [Instalação](#instalacao)
5. [Uso](#uso)
6. [Treinamento](#treinamento)
7. [Métricas e Resultados](#metricas-e-resultados)
8. [Estrutura do Projeto](#estrutura-do-projeto)
9. [Contribuições](#contribuicoes)
10. [Licença](#licenca)
11. [Reconhecimentos](#reconhecimentos)

---
## 🌟 Visão Geral

O objetivo deste projeto é construir um modelo robusto para distinguir entre imagens de **gatos** e **cachorros**. Utilizamos:
- **MobileNetV2**: Um modelo leve e eficiente pré-treinado no ImageNet.
- **Fine-tuning**: Ajuste fino das camadas superiores para melhorar a precisão.
- **Aumento de Dados**: Técnicas como flips, rotações e zoom para aumentar a diversidade do dataset.

---
## 📊 Dataset

O dataset utilizado é o [**Cats vs Dogs**](https://www.microsoft.com/en-us/research/project/cats-vs-dogs/), disponível no [TensorFlow Datasets](https://www.tensorflow.org/datasets). Ele contém aproximadamente **25.000 imagens rotuladas** de gatos e cachorros. O dataset foi dividido da seguinte forma:

- **Treinamento**: 80% (usado para treinar o modelo).
- **Validação**: 10% (usado para ajustar hiperparâmetros e monitorar o desempenho).
- **Teste**: 10% (usado para avaliar o desempenho final do modelo).

As imagens são redimensionadas para **224x224 pixels** e normalizadas com `mobilenet_v2.preprocess_input`. Além disso, técnicas de aumento de dados (flips, rotações, zoom) são aplicadas ao conjunto de treinamento.

---
## 💻 Requisitos

Para executar este projeto, você precisa dos seguintes pacotes:

- Python 3.8+
- TensorFlow 2.x
- TensorFlow Datasets
- Matplotlib
- NumPy

Instale as dependências com o comando abaixo:
`pip install tensorflow tensorflow_datasets matplotlib numpy`
## 🔧 Instalação
Clone o Repositório
`git clone https://github.com/seu-usuario/cats-vs-dogs-mobilenetv2.git
cd cats-vs-dogs-mobilenetv2`
Instale as Dependências
`pip install -r requirements.txt`
## 🚀 Uso
Treinamento e Avaliação
Abra o notebook notebooks/cats_vs_dogs_mobilenetv2.ipynb em seu ambiente Jupyter ou Google Colab para treinar e avaliar o modelo.

Teste com Imagens Externas
Você pode usar a função predict_image para testar o modelo com suas próprias imagens:

from predict import predict_image
predict_image('caminho/para/sua/imagem.jpg')
## 🏋️‍♂️ Treinamento
O treinamento é dividido em duas fases:

Treinamento Inicial: As camadas do MobileNetV2 são congeladas, e apenas as camadas superiores são treinadas.
Fine-Tuning: Parte das camadas do MobileNetV2 é descongelada para ajuste fino.
Os gráficos de acurácia e loss estão disponíveis no notebook.

## 📈 Métricas e Resultados
Após o treinamento inicial e o fine-tuning, o modelo alcançou os seguintes resultados:

Treinamento Inicial:

Acurácia no Treino: ~85-90%
Acurácia na Validação: ~80-85%
Fine-Tuning:

Acurácia no Treino: ~90-95%
Acurácia na Validação: ~85-90%
Avaliação Final:

Acurácia no Teste: ~85-90%
Gráficos detalhados de acurácia e loss estão disponíveis no notebook.

## 📂 Estrutura do Projeto


```plaintext

cats-vs-dogs-mobilenetv2/
├── notebooks/                  # Notebooks Jupyter com código detalhado
│   └── cats_vs_dogs_mobilenetv2.ipynb
├── data/                       # Dados baixados automaticamente pelo TensorFlow Datasets
├── models/                     # Modelos salvos (opcional)
│   └── cats_vs_dogs_mobilenetv2.keras
├── scripts/                    # Scripts utilitários
│   └── predict.py              # Função de predição para imagens externas
├── assets/                     # Imagens usadas nos tutoriais
├── requirements.txt            # Lista de dependências
└── README.md                   # Este arquivo
  ```


## 🤝 Contribuições
Contribuições são bem-vindas! Siga os passos abaixo:

Faça um fork deste repositório.
Crie uma branch para sua contribuição:
`git checkout -b feature/nova-funcionalidade`
Envie um pull request detalhando suas alterações.
## 📜 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
## 🙏 Reconhecimentos
TensorFlow Team: Pela disponibilização do dataset cats_vs_dogs e do framework TensorFlow.
Autores do MobileNetV2: Pelo excelente modelo pré-treinado.
Comunidade Open Source: Por fornecer ferramentas e recursos que facilitam o desenvolvimento de projetos de IA.
## 📸 Exemplo de Predição
Exemplo de Predição
Predição: Gato
Confiança: 98.76%
## 🌐 Links Úteis
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Dataset Cats vs Dogs](https://www.microsoft.com/en-us/research/project/cats-vs-dogs/)


