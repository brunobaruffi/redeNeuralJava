package org.example;

public class Layer {
    public Neuron[] neurons; // 'neurons': Uma lista (array) de objetos Neuron. Esta camada é feita de vários neurônios.
    private int numInputsPerNeuron; // Quantas entradas CADA neurônio nesta camada vai receber da camada ANTERIOR.

    // Construtor: Quando criamos uma camada, precisamos dizer quantos neurônios ela terá e de quantos lugares ela recebe informação.
    public Layer(int numNeurons, int numInputsPerNeuron) {
        this.numInputsPerNeuron = numInputsPerNeuron;
        neurons = new Neuron[numNeurons]; // Cria um array para guardar os neurônios da camada

        // Se 'numInputsPerNeuron' for 0, significa que é a camada de entrada (ela não recebe de outra camada, mas de fora).
        if (numInputsPerNeuron == 0) {
            for (int i = 0; i < numNeurons; i++) {
                neurons[i] = new Neuron(); // Cria neurônios simples (sem pesos/viés) para a camada de entrada.
            }
        } else { // Se não for 0, é uma camada oculta ou de saída.
            for (int i = 0; i < numNeurons; i++) {
                neurons[i] = new Neuron(numInputsPerNeuron); // Cria neurônios completos (com pesos/viés) que recebem informações da camada anterior.
            }
        }
    }

    // Este método está vazio porque os pesos e vieses já são inicializados no construtor do Neuron.
    // Ele existe aqui mais como um lembrete se você quisesse mudar a forma de inicialização depois.
    public void initializeWeightsAndBiases() {
        // Já feito no construtor do Neuron.
    }
}