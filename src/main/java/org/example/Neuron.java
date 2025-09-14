package org.example;

import java.util.Random;

public class Neuron {
    public double[] weights; // 'weights' (pesos): Um monte de números que indicam a 'importância' de cada entrada para este neurônio.
    // É um array (uma lista) porque o neurônio pode ter várias entradas.
    public double bias;      // 'bias' (viés): Um número extra que sempre adicionamos para 'ajustar' a decisão do neurônio.

    public double input;     // 'input' (entrada bruta): É a soma das entradas * pesos + viés. Antes de passar pela 'porta' Sigmoide.
    public double output;    // 'output' (saída): É o resultado final depois de passar o 'input' pela 'porta' Sigmoide.
    public double delta;     // 'delta': Usado APENAS durante o aprendizado (backpropagation). Indica o "quanto" o neurônio errou e como ele precisa ajustar seus pesos/viés.

    // Construtor 1: Para neurônios que recebem entradas de outros neurônios (camada oculta e de saída)
    public Neuron(int numInputs) {
        weights = new double[numInputs]; // Cria um array de pesos do tamanho do número de entradas que este neurônio vai receber
        Random rand = new Random();      // Cria um objeto para gerar números aleatórios

        // Inicializa os pesos e o viés com valores aleatórios entre -1 e 1.
        // Isso é importante para a rede começar "adivinhando" e não ter todos os neurônios fazendo a mesma coisa.
        for (int i = 0; i < numInputs; i++) {
            weights[i] = rand.nextDouble() * 2 - 1; // rand.nextDouble() dá um número entre 0 e 1. Multiplicamos por 2 e subtraímos 1 para ter entre -1 e 1.
        }
        bias = rand.nextDouble() * 2 - 1; // O viés também é aleatório.
    }

    // Construtor 2: Para neurônios da camada de entrada. Eles são mais simples.
    // Eles não precisam de pesos nem viés porque apenas repassam o valor que recebem de fora da rede.
    public Neuron() {
        // Não faz nada, a saída será diretamente o valor de entrada que definiremos depois.
    }
}