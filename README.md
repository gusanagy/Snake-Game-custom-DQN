# Snake Game with Custom DQN model
The implementation of the snake game using deep q leaning methods to implement an AI that can play. 

A rede neural utilizada foi testada e modificada de forma que pudesse ser utilizada para o treinamento de um modelo computacional hábil a jogar o jogo SNAKE. Este modelo apresenta camadas lineares, camadas de dropout e funções de ativação rotacionais e leaky relu.

A maior inspiração deste projeto teve como referência a aula dada pela FreeCodeCamp no youtube, Python + PyTorch + Pygame Reinforcement Learning – Train an AI to Play Snake. Que implementa o ambiente e o algoritmo da DQN. Mas não otimiza a rede utilizada nem os parâmetros.
Nesse sentido, e tomando como ideia inicial o artigo Adaptive Rational Activations to Boost Deep Reinforcement Learning, foram implemtadas funções de ativação leaky relu e rotacionais. E as camadas da rede foram modificadas para funcionarem de acordo. 
O desempenho da rede ainda não mostrou comportamento otimo. Poŕem pode aprender a jogar e pontuar ate os 50 pontos. 

### Rodando o codigo e dependencias 
Este codigo foi desnvolvido num ambiente conda utilizando como principais pacotes: Pygame 2.5.0, python 3.7.16, torch 1.13.1 e torchvision 0.14.1. Para mais informações sobre o ambiente por favor visitar o inicio do video da FreeCodeCamp nas referências.
Para instalação siga: 
```
conda create -n pygame_env python=3.7
conda activate pygame_env
pip install pygame
pip install torch torchvision
pip install matplotlib ipython
```
Apos instalado o pacote rode e entrar no diretorio dos arquivos rode no terminal
```
Python3 agent.py
```

## Referências
Python + PyTorch + Pygame Reinforcement Learning – Train an AI to Play Snake
https://youtu.be/L8ypSXwyBds

Adaptive Rational Activations to Boost Deep Reinforcement Learning
https://arxiv.org/pdf/2102.09407.pdf

