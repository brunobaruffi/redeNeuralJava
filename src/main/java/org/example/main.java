package org.example;

public class main {

    public static void main(String[] args) {
        // --- 1. Dados de Treinamento ---
        // Aqui estão os exemplos que a rede vai usar para aprender.
        // O problema:
        // Peso (g)	Textura (0=lisa,1=rugosa)	Fruta (0=maçã,1=laranja)
        // 150	0	0
        // 170	0	0
        // 140	0	0
        // 130	1	1
        // 160	1	1
        // 180	1	1

        // MUITO IMPORTANTE: NORMALIZAÇÃO DOS DADOS!
        // A função Sigmoide funciona melhor com números entre 0 e 1.
        // Se passarmos pesos como 150, 170, a rede pode ter dificuldade para aprender.
        // Então, transformamos os pesos para o intervalo de 0 a 1.
        // Como fazer: (Valor_Atual - Menor_Peso_Total) / (Maior_Peso_Total - Menor_Peso_Total)
        // Nosso menor peso é 130, o maior é 180.
        // Exemplo: Para 150g -> (150 - 130) / (180 - 130) = 20 / 50 = 0.4

        double[][] inputs = {
                { (150.0 - 130.0) / (180.0 - 130.0), 0.0 }, // Maçã (0.4, 0.0)
                { (170.0 - 130.0) / (180.0 - 130.0), 0.0 }, // Maçã (0.8, 0.0)
                { (140.0 - 130.0) / (180.0 - 130.0), 0.0 }, // Maçã (0.2, 0.0)
                { (130.0 - 130.0) / (180.0 - 130.0), 1.0 }, // Laranja (0.0, 1.0)
                { (160.0 - 130.0) / (180.0 - 130.0), 1.0 }, // Laranja (0.6, 1.0)
                { (180.0 - 130.0) / (180.0 - 130.0), 1.0 }  // Laranja (1.0, 1.0)
        };

        // As saídas esperadas (0.0 para maçã, 1.0 para laranja)
        double[][] targets = {
                { 0.0 }, // Maçã
                { 0.0 }, // Maçã
                { 0.0 }, // Maçã
                { 1.0 }, // Laranja
                { 1.0 }, // Laranja
                { 1.0 }  // Laranja
        };

        // --- 2. Cria a Rede Neural ---
        // 2 entradas (Peso, Textura), 3 neurônios ocultos (você pode experimentar outros números), 1 saída (Fruta)
        NeuralNetwork rede = new NeuralNetwork(2, 3, 1);

        rede.treinar(rede,inputs,targets);

        // --- 4. Teste da Rede Treinada ---
        // Depois de aprender, vamos ver se ela consegue classificar novos exemplos.
        System.out.println("\nTestando a rede:");

        // Exemplo 1: Maçã (peso 155g, textura lisa)
        // Lembre-se de normalizar as entradas de teste também!
        double[] testInput1 = { (155.0 - 130.0) / (180.0 - 130.0), 0.0 }; // Peso 155g -> (25/50)=0.5, textura 0.0
        double[] output1 = rede.propagar_FeedForward(testInput1); // Pede para a rede "adivinhar"
        System.out.printf("Peso: 155g, Textura: Lisa (0) -> Saída: %.4f (Esperado: Maçã - 0)%n", output1[0]);

        // Exemplo 2: Laranja (peso 175g, textura rugosa)
        double[] testInput2 = { (175.0 - 130.0) / (180.0 - 130.0), 1.0 }; // Peso 175g -> (45/50)=0.9, textura 1.0
        double[] output2 = rede.propagar_FeedForward(testInput2);
        System.out.printf("Peso: 175g, Textura: Rugosa (1) -> Saída: %.4f (Esperado: Laranja - 1)%n", output2[0]);

        // Mais exemplos para testar
        double[] testInput3 = { (145.0 - 130.0) / (180.0 - 130.0), 0.0 }; // Peso 145g -> (15/50)=0.3, textura 0.0
        double[] output3 = rede.propagar_FeedForward(testInput3);
        System.out.printf("Peso: 145g, Textura: Lisa (0) -> Saída: %.4f (Esperado: Maçã - 0)%n", output3[0]);

        double[] testInput4 = { (135.0 - 130.0) / (180.0 - 130.0), 1.0 }; // Peso 135g -> (5/50)=0.1, textura 1.0
        double[] output4 = rede.propagar_FeedForward(testInput4);
        System.out.printf("Peso: 135g, Textura: Rugosa (1) -> Saída: %.4f (Esperado: Laranja - 1)%n", output4[0]);
    }
}
