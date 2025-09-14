package org.example;

public class NeuralNetwork {

    private Layer entradaLayer;  // A camada que recebe os dados (Peso, Textura)
    private Layer ocultaLayer; // A camada do meio que processa
    private Layer saidaLayer; // A camada que dá a resposta final (Maçã ou Laranja)

    private double learningRate = 0.5; // 'learningRate' (Taxa de Aprendizado): QUÃO RÁPIDO a rede vai ajustar seus pesos e vieses a cada erro.
    // Um valor alto pode fazer a rede "pular" a resposta certa; um valor baixo pode torná-la muito lenta para aprender.

    // Construtor da Rede Neural: Monta toda a estrutura.
    public NeuralNetwork(int numInput, int numHidden, int numOutput) {
        // numInput: Quantos neurônios na camada de entrada (2: Peso, Textura)
        // numHidden: Quantos neurônios na camada oculta (3)
        // numOutput: Quantos neurônios na camada de saída (1: Fruta)

        entradaLayer = new Layer(numInput, 0); // Cria a camada de entrada. O '0' indica que ela não recebe de outra camada.
        ocultaLayer = new Layer(numHidden, numInput); // Cria a camada oculta. Cada neurônio oculto recebe de 'numInput' neurônios.
        saidaLayer = new Layer(numOutput, numHidden); // Cria a camada de saída. Cada neurônio de saída recebe de 'numHidden' neurônios.

        // Os pesos e vieses já são inicializados aleatoriamente quando os neurônios são criados nas camadas ocultas e de saída.
        // hiddenLayer.initializeWeightsAndBiases(); // Este método está vazio, como vimos em Layer.java
        // outputLayer.initializeWeightsAndBiases(); // Este método também
    }

    // --- Funções Auxiliares para os Neurônios ---

    // Função de Ativação Sigmoide: A "porta" que transforma a soma ponderada em um valor entre 0 e 1.
    public double calcularSigmoide(double x) {
        return 1.0 / (1.0 + Math.exp(-x)); // 'Math.exp(-x)' calcula 'e' elevado a '-x'. É uma fórmula matemática padrão para a Sigmoide.
    }

    // Derivada da Função de Ativação Sigmoide: ESSENCIAL para o aprendizado!
    // Esta fórmula nos diz o quão "sensível" a saída do neurônio é a uma pequena mudança em sua entrada.
    // Precisamos disso para saber o quanto ajustar os pesos e vieses durante o backpropagation.
    public double calcularDerivadaSigmoide(double x) {
        // A derivada da sigmoide pode ser calculada facilmente usando o próprio valor da sigmoide (x * (1 - x)).
        // 'x' aqui já é o valor de 'output' (a saída) do neurônio, que já passou pela sigmoid.
        return x * (1.0 - x);
    }

    // --- Propagação Direta (Feedforward) ---
    // Faz os dados fluírem da entrada até a saída para obter uma "adivinhação".
    public double[] propagar_FeedForward(double[] inputs) {
        // 1. Passa as entradas para a Camada de Entrada
        for (int i = 0; i < entradaLayer.neurons.length; i++) {
            entradaLayer.neurons[i].output = inputs[i]; // Cada neurônio de entrada apenas recebe e armazena o valor de entrada.
        }

        // 2. Calcula as saídas da Camada Oculta
        for (int i = 0; i < ocultaLayer.neurons.length; i++) {
            Neuron neuron = ocultaLayer.neurons[i]; // Pega um neurônio da camada oculta
            neuron.input = neuron.bias;             // Começa somando o viés do neurônio

            // Soma todas as entradas da camada anterior (inputLayer) multiplicadas pelos pesos deste neurônio
            for (int j = 0; j < entradaLayer.neurons.length; j++) {
                neuron.input += entradaLayer.neurons[j].output * neuron.weights[j]; // (Saída do neurônio de entrada * Peso da conexão)
            }
            neuron.output = calcularSigmoide(neuron.input); // Aplica a função de ativação Sigmoide ao resultado final
        }

        // 3. Calcula as saídas da Camada de Saída
        // É o mesmo processo da camada oculta, mas agora usando as saídas da camada oculta como entradas.
        for (int i = 0; i < saidaLayer.neurons.length; i++) {
            Neuron neuron = saidaLayer.neurons[i]; // Pega um neurônio da camada de saída
            neuron.input = neuron.bias;             // Começa somando o viés

            // Soma todas as entradas da camada anterior (hiddenLayer) multiplicadas pelos pesos deste neurônio
            for (int j = 0; j < ocultaLayer.neurons.length; j++) {
                neuron.input += ocultaLayer.neurons[j].output * neuron.weights[j]; // (Saída do neurônio oculto * Peso da conexão)
            }
            neuron.output = calcularSigmoide(neuron.input); // Aplica a função de ativação Sigmoide
        }

        // Retorna as saídas finais da rede (a "adivinhação")
        double[] outputs = new double[saidaLayer.neurons.length];
        for (int i = 0; i < saidaLayer.neurons.length; i++) {
            outputs[i] = saidaLayer.neurons[i].output;
        }
        return outputs;
    }

    // --- Retropropagação (Backpropagation) - O Aprendizado ---
    // Ajusta os pesos e vieses da rede com base no erro.
    public void treinoUnitario(double[] inputs, double[] targets) {
        // 'inputs': Os dados que estamos mostrando para a rede (ex: Peso, Textura)
        // 'targets': A resposta CORRETA que a rede deveria dar (ex: 0.0 para maçã)

        // 1. Primeiro, fazemos a Propagação Direta para ver o que a rede "adivinha" AGORA.
        double[] outputs = propagar_FeedForward(inputs);

        // 2. Cálculo do Erro e do 'Delta' para a Camada de Saída
        // O 'delta' indica o erro e a direção que precisamos ajustar.
        for (int i = 0; i < saidaLayer.neurons.length; i++) {
            Neuron neuron = saidaLayer.neurons[i];
            // Fórmula do Delta: (Alvo_Correto - Saída_Atual_da_Rede) * Derivada_da_Sigmoide_da_Saída_Atual
            // Isso nos diz: "Quanto eu errei?" e "Para que lado eu devo ajustar (para mais ou para menos)?"
            neuron.delta = (targets[i] - neuron.output) * calcularDerivadaSigmoide(neuron.output);
        }

        // 3. Cálculo do 'Delta' para a Camada Oculta (Retropropagação do Erro)
        // Agora, o erro da camada de saída é "distribuído" de volta para a camada oculta.
        // Um neurônio oculto que tem uma conexão forte com um neurônio de saída que errou muito, vai receber um 'delta' maior.
        for (int i = 0; i < ocultaLayer.neurons.length; i++) {
            Neuron neuron = ocultaLayer.neurons[i];
            double sumOfWeightsTimesDeltas = 0; // Soma dos erros propagados
            for (int j = 0; j < saidaLayer.neurons.length; j++) {
                // (Peso da conexão do neurônio oculto com o neurônio de saída * Delta do neurônio de saída)
                sumOfWeightsTimesDeltas += saidaLayer.neurons[j].weights[i] * saidaLayer.neurons[j].delta;
            }
            // Delta da camada oculta: (Soma dos erros propagados) * Derivada_da_Sigmoide_da_Saída_Atual_do_Oculto
            neuron.delta = sumOfWeightsTimesDeltas * calcularDerivadaSigmoide(neuron.output);
        }




        // 4. Ajuste dos Pesos e Vieses da Camada de Saída
        // Finalmente, usamos os 'deltas' para mudar os pesos e vieses.
        // Novo Peso = Peso Atual + (Taxa de Aprendizado * Delta do Neurônio de Saída * Saída do Neurônio da Camada ANTERIOR)
        for (int i = 0; i < saidaLayer.neurons.length; i++) {
            Neuron neuron = saidaLayer.neurons[i];
            for (int j = 0; j < ocultaLayer.neurons.length; j++) {
                neuron.weights[j] += learningRate * neuron.delta * ocultaLayer.neurons[j].output;
            }
            // Novo Viés = Viés Atual + (Taxa de Aprendizado * Delta do Neurônio de Saída)
            neuron.bias += learningRate * neuron.delta;
        }

        // 5. Ajuste dos Pesos e Vieses da Camada Oculta
        // O mesmo processo, mas para os pesos e vieses que ligam a camada de entrada à oculta.
        for (int i = 0; i < ocultaLayer.neurons.length; i++) {
            Neuron neuron = ocultaLayer.neurons[i];
            for (int j = 0; j < entradaLayer.neurons.length; j++) {
                neuron.weights[j] += learningRate * neuron.delta * entradaLayer.neurons[j].output;
            }
            neuron.bias += learningRate * neuron.delta;
        }
    }

    // --- Loop de treinamento. Chamada em loop do treinoUnitario para executar o trinamento
    public void treinar(NeuralNetwork rede, double[][] inputs, double[][] targets){
        int epochs = 100000; // 'epochs' (Épocas): Quantas vezes a rede vai "ver" TODO o conjunto de dados de treinamento.
        // Quanto mais épocas, mais ela tem chance de aprender, mas também pode "decorar" demais.

        System.out.println("Iniciando treinamento...");
        // --- 3. Loop de Treinamento ---
        // A rede vai repetir o processo de feedforward e backpropagation muitas vezes.
        for (int i = 0; i < epochs; i++) { // Loop para cada época
            for (int j = 0; j < inputs.length; j++) { // Loop para cada exemplo de treinamento
                rede.treinoUnitario(inputs[j], targets[j]); // Chama o método 'train' para ensinar a rede com um exemplo
            }
            // A cada 10.000 épocas, ele imprime o erro total para vermos se a rede está aprendendo.
            if (i % 10000 == 0) { // O '%' é o operador de resto da divisão. Se 'i' dividido por 10000 tiver resto 0, então imprime.
                double totalError = 0;
                for (int j = 0; j < inputs.length; j++) {
                    double[] output = rede.propagar_FeedForward(inputs[j]); // Vê o que a rede adivinha para CADA exemplo
                    totalError += Math.pow(targets[j][0] - output[0], 2); // Calcula o erro (diferença ao quadrado)
                }
                System.out.printf("Época %d, Erro: %.4f%n", i, totalError); // Imprime a época e o erro
            }
        }
        System.out.println("Treinamento concluído.");
    }
}