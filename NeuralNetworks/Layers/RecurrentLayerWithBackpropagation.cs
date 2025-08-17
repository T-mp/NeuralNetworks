using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers;

public class RecurrentLayerWithBackpropagation : RecurrentLayer, IModelLayerWithBackpropagation
{
    public RecurrentLayerWithBackpropagation(
        int nodeCount,
        IActivationWithDerivat activation,
        bool useBias,
        IInitializer kernelInitializer,
        IInitializer biasInitializer,
        IInitializer recurrentInitializer)
        : base(nodeCount, activation, useBias, kernelInitializer, biasInitializer, recurrentInitializer)
    { }

    public NamedVectors<float> BackpropagationState { get; } = new NamedVectors<float>();
    protected override void BackpropagationStateSet(string name, int index, float value)
    {
        BackpropagationState.Get1dVector(name)[index] = value;
    }

    public override float[] Update(float[] inputValues)
    {
        if (!IsBildet) throw new InvalidOperationException("Layer must be built before updating");

        BackpropagationState.Clear();
        BackpropagationState.Add("inputs", inputValues);
        BackpropagationState.Add("preNodeValues", (float[])nodeValues.Clone());
        BackpropagationState.Add("lastPreAct", new float[OutputSize.TotalSize]);

        return base.Update(inputValues);
    }

    public float[] Backward(float[] outputError, float learningRate)
    {
        if (!IsBildet) throw new InvalidOperationException("Layer must be built before updating");
        var derivative = activation as IActivationWithDerivat
            ?? throw new InvalidOperationException("Activation function must implement IActivationWithRevert for backpropagation.");

        var lastInput = (ReadOnlySpan<float>)BackpropagationState.Get1dVector("inputs");
        var preNodeValues = (ReadOnlySpan<float>)BackpropagationState.Get1dVector("preNodeValues");
        var lastPreAct = (ReadOnlySpan<float>)BackpropagationState.Get1dVector("lastPreAct");

        int nodes = OutputSize.TotalSize;
        int inputLength = lastInput.Length;
        float[] inputError = new float[inputLength];

        float[,] weightsGrad = new float[nodes, inputLength];
        float[] recurrentWeightsGrad = new float[nodes];
        float[]? biasesGrad = useBias ? new float[nodes] : null;

        float[] prevStateError = new float[nodes]; // Fehler rückwärts auf "letzten" State

        for (int nodeIndex = 0; nodeIndex < nodes; nodeIndex++)
        {
            // Ableitung der Aktivierung, am Voraktivierungswert!
            float actPrime = ((IActivationWithDerivat)activation).Derivat(lastPreAct[nodeIndex]);

            float delta = outputError[nodeIndex] * actPrime;

            // Gradienten für Input-Gewichte und Fehler auf Inputs
            for (int inputIndex = 0; inputIndex < inputLength; inputIndex++)
            {
                weightsGrad[nodeIndex, inputIndex] += delta * lastInput[inputIndex];
                inputError[inputIndex] += weights[nodeIndex, inputIndex] * delta;
            }

            // Gradienten der rekurrenten Gewichte
            recurrentWeightsGrad[nodeIndex] += delta * preNodeValues[nodeIndex];

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
            for (int inputIndex = 0; inputIndex < inputLength; inputIndex++)
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