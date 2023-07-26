# Snake Game with Custom DQN model
The implementation of the snake game using deep q leaning methods to implement an AI that can play. 

A rede neural utilizada foi testada e modificada de forma que pudesse ser utilizada para o treinamento de um modelo computacional hábil a jogar o jogo SNAKE. Este modelo apresenta camadas lineares, camadas de dropout e funções de ativação rotacionais e leaky relu.

A maior inspiração deste projeto teve como referência a aula dada pela FreeCodeCamp no youtube, Python + PyTorch + Pygame Reinforcement Learning – Train an AI to Play Snake. Que implementa o ambiente e o algoritmo da DQN. Mas não otimiza a rede utilizada nem os parâmetros.
Nesse sentido, e tomando como ideia inicial o artigo Adaptive Rational Activations to Boost Deep Reinforcement Learning, foram implemtadas funções de ativação leaky relu e rotacionais. E as camadas da rede foram modificadas para funcionarem de acordo. 
O desempenho da rede ainda não mostrou comportamento otimo. Poŕem pode aprender a jogar e pontuar ate os 50 pontos. 

## Referências
Python + PyTorch + Pygame Reinforcement Learning – Train an AI to Play Snake
https://youtu.be/L8ypSXwyBds

Adaptive Rational Activations to Boost Deep Reinforcement Learning
https://arxiv.org/pdf/2102.09407.pdf

