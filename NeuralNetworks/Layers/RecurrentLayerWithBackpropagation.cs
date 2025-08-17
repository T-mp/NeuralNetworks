using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Layers;

public class RecurrentLayerWithBackpropagation : RecurrentLayer, IModelLayerWithBackpropagation
{
    protected float[] lastInput = default!;
    protected float[] lastPreAct = default!;
    protected float[] lastNodeValuesBeforeUpdate = default!;

    public RecurrentLayerWithBackpropagation(
        int nodeCount,
        IActivationWithDerivat activation,
        bool useBias,
        IInitializer kernelInitializer,
        IInitializer biasInitializer,
        IInitializer recurrentInitializer)
        : base(nodeCount, activation, useBias, kernelInitializer, biasInitializer, recurrentInitializer)
    { }

    public override float[] Update(float[] inputValues)
    {
        lastInput = (float[])inputValues.Clone();
        lastNodeValuesBeforeUpdate = (float[])nodeValues.Clone();

        lastPreAct = new float[OutputSize.TotalSize];
        for (int nodeIndex = 0; nodeIndex < OutputSize.TotalSize; nodeIndex++)
        {
            var nodeValue = recurrentWeights[nodeIndex] * lastNodeValuesBeforeUpdate[nodeIndex];
            for (int inputIndex = 0; inputIndex < inputValues.Length; inputIndex++)
            {
                nodeValue += inputValues[inputIndex] * weights[nodeIndex, inputIndex];
            }
            if (useBias)
            {
                nodeValue += biases[nodeIndex];
            }
            lastPreAct[nodeIndex] = nodeValue;
            nodeValues[nodeIndex] = activation.Apply(nodeValue);
        }
        return nodeValues;
    }

    public float[] Backward(float[] outputError, float learningRate)
    {
        int nodes = OutputSize.TotalSize;
        int inputs = lastInput.Length;
        float[] inputError = new float[inputs];

        float[,] weightsGrad = new float[nodes, inputs];
        float[] recurrentWeightsGrad = new float[nodes];
        float[]? biasesGrad = useBias ? new float[nodes] : null;

        float[] prevStateError = new float[nodes]; // Fehler rückwärts auf "letzten" State

        for (int nodeIndex = 0; nodeIndex < nodes; nodeIndex++)
        {
            // Ableitung der Aktivierung, am Voraktivierungswert!
            float actPrime = ((IActivationWithDerivat)activation).Derivat(lastPreAct[nodeIndex]);

            float delta = outputError[nodeIndex] * actPrime;

            // Gradienten für Input-Gewichte und Fehler auf Inputs
            for (int inputIndex = 0; inputIndex < inputs; inputIndex++)
            {
                weightsGrad[nodeIndex, inputIndex] += delta * lastInput[inputIndex];
                inputError[inputIndex] += weights[nodeIndex, inputIndex] * delta;
            }

            // Gradienten der rekurrenten Gewichte
            recurrentWeightsGrad[nodeIndex] += delta * lastNodeValuesBeforeUpdate[nodeIndex];

            // Fehler auf vorherigen hidden state (wird ggf. in Backpropagation Through Time verwendet)
            prevStateError[nodeIndex] = recurrentWeights[nodeIndex] * delta;

            if (useBias)
            {
                biasesGrad![nodeIndex] += delta;
            }
        }

        // Parameter-Updates
        for (int nodeIndex = 0; nodeIndex < nodes; nodeIndex++)
        {
            for (int inputIndex = 0; inputIndex < inputs; inputIndex++)
            {
                weights[nodeIndex, inputIndex] -= learningRate * weightsGrad[nodeIndex, inputIndex];
            }
            recurrentWeights[nodeIndex] -= learningRate * recurrentWeightsGrad[nodeIndex];
            if (useBias)
            {
                biases[nodeIndex] -= learningRate * biasesGrad![nodeIndex];
            }
        }

        // Rückgabe: Fehler auf Inputs; Option: prevStateError für Backpropagation Through Time weiterreichen
        return inputError;
    }
}