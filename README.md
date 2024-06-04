### Creating a Machine Learning Model using Pytorch and CV2 to predict Brazilian coin images 

This project is part of my journey to learn more about PyTorch, a powerful open-source machine learning library, and OpenCV (cv2), a library of programming functions mainly aimed at real-time computer vision. By building a model to predict Brazilian coin denominations from images, I aim to gain hands-on experience and deepen my understanding of how to use PyTorch and cv2 for real-world machine learning tasks.

In this notebook, I will be using the br-coins(https://www.kaggle.com/datasets/lgmoneda/br-coins) dataset to create a Convolutional Neural Network (CNN) model for image classification. The goal is to accurately predict the denomination of Brazilian coins based on images. Additionally, I will be using cv2 to identify circles in the images, which represent the coins.

The steps we will follow are:

*  **Data Preprocessing**: We will use the OpenCV library (cv2) to preprocess the images. This includes identifying circles (coins) in the images and resizing the images to a uniform size.
*  **Model Building**: We will use PyTorch to build our CNN. The model will consist of several convolutional layers for feature extraction, followed by fully connected layers for classification.
*  **Training**: We will train our model on the training set using a suitable loss function and optimizer. We will also use a validation set to tune the hyperparameters and prevent overfitting.
*  **Evaluation**: We will evaluate the performance of our model on the test set. We will use metrics such as accuracy, precision, recall, and F1 score to measure the performance of our model.
*  **Prediction**: Finally, we will use our trained model to predict the denomination of Brazilian coins from images.

By the end of this project, I hope to have a working model and a better understanding of how to use PyTorch and cv2 for image classification tasks.


### Criando um Modelo de Machine Learning usando Pytorch e CV2 para prever imagens de moedas brasileiras

Este projeto faz parte da minha jornada para aprender mais sobre PyTorch, uma poderosa biblioteca de aprendizado de máquina de código aberto, e OpenCV (cv2), uma biblioteca de funções de programação voltada principalmente para visão computacional em tempo real. Ao construir um modelo para prever as denominações de moedas brasileiras a partir de imagens, pretendo ganhar experiência prática e aprofundar minha compreensão de como usar o PyTorch e o cv2 para tarefas de aprendizado de máquina do mundo real.

Neste notebook, estarei usando o conjunto de dados br-coins(https://www.kaggle.com/datasets/lgmoneda/br-coins) para criar um modelo de Rede Neural Convolucional (CNN) para classificação de imagens. O objetivo é prever com precisão a denominação das moedas brasileiras com base nas imagens. Além disso, estarei usando o cv2 para identificar círculos nas imagens, que representam as moedas.

As etapas que seguiremos são:

*  **Pré-processamento de Dados**: Usaremos a biblioteca OpenCV (cv2) para pré-processar as imagens. Isso inclui identificar círculos (moedas) nas imagens e redimensionar as imagens para um tamanho uniforme.
*  **Construção do Modelo**: Usaremos o PyTorch para construir nossa CNN. O modelo consistirá em várias camadas convolucionais para extração de características, seguidas por camadas totalmente conectadas para classificação.
*  **Treinamento**: Treinaremos nosso modelo no conjunto de treinamento usando uma função de perda adequada e um otimizador. Também usaremos um conjunto de validação para ajustar os hiperparâmetros e prevenir o sobreajuste.
*  **Avaliação**: Avaliaremos o desempenho do nosso modelo no conjunto de testes. Usaremos métricas como acurácia, precisão, recall e pontuação F1 para medir o desempenho do nosso modelo.
*  **Predição**: Finalmente, usaremos nosso modelo treinado para prever a denominação das moedas brasileiras a partir de imagens.

Ao final deste projeto, espero ter um modelo funcional e uma melhor compreensão de como usar o PyTorch e o cv2 para tarefas de classificação de imagens.
