using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers;

public class GruLayerWithBackpropagation : GruLayer, IModelLayerWithBackpropagation
{
    public GruLayerWithBackpropagation(
        Size1D nodeCount,
        IActivationWithDerivat activation,
        IActivationWithDerivat recurrentActivation,
        bool useBias,
        IInitializer kernelInitializer,
        IInitializer recurrentInitializer,
        IInitializer biasInitializer)
        : base(nodeCount, activation, recurrentActivation, useBias, kernelInitializer, recurrentInitializer, biasInitializer)
    { }

    public NamedVectors<float> BackpropagationState { get; } = new NamedVectors<float>();

    /// <inheritdoc/>>
    public override float[] Update(float[] inputValues)
    {
        if (!IsBildet) throw new InvalidOperationException("Layer must be built before updating");

        BackpropagationState.Clear();
        BackpropagationState.Add("inputs", inputValues);
        BackpropagationState.Add("VorLastUpdateNodeValues", (float[])NodeValues.Clone());

        BackpropagationState.Add("forgetGateInputs", new float[OutputSize.TotalSize]);
        BackpropagationState.Add("forgetGates", new float[OutputSize.TotalSize]);
        BackpropagationState.Add("candidateInputs", new float[OutputSize.TotalSize]);
        BackpropagationState.Add("candidates", new float[OutputSize.TotalSize]);

        return base.Update(inputValues);
    }

    protected override void BackpropagationStateSet(string name, int index, float value)
    {
        BackpropagationState.Get1dVector(name)[index] = value;
    }

    /// <summary>
    /// Backpropagation für einen GRU-Layer (vereinfachte Variante analog zu deinem Aufbau).
    /// </summary>
    public float[] Backward(float[] outputError, float learningRate)
    {
        if (!IsBildet)
            throw new InvalidOperationException("Layer must be built before Backward can be called.");

        // Fehler-Arrays
        float[] inputError = new float[ForgetGateWeights.GetLength(1)];

        // Gradienten für Gewichte/Rekurrenz
        float[,] forgetGateWeightsGrad = new float[ForgetGateWeights.GetLength(0), ForgetGateWeights.GetLength(1)];
        float[] forgetRecurrentWeightsGrad = new float[ForgetRecurrentWeights.Length];

        float[,] candidateWeightsGrad = new float[CandidateWeights.GetLength(0), CandidateWeights.GetLength(1)];
        float[] candidateRecurrentWeightsGrad = new float[CandidateRecurrentWeights.Length];

        float[]? forgetBiasGrad = UseBias ? new float[ForgetBiases.Length] : null;
        float[]? candidateBiasGrad = UseBias ? new float[CandidateBiases.Length] : null;

        var inputs = (ReadOnlySpan<float>)BackpropagationState.Get1dVector("inputs");
        var VorLastUpdateNodeValues = (ReadOnlySpan<float>)BackpropagationState.Get1dVector("VorLastUpdateNodeValues");

        var forgetGateInputs = (ReadOnlySpan<float>)BackpropagationState.Get1dVector("forgetGateInputs");
        var forgetGates = (ReadOnlySpan<float>)BackpropagationState.Get1dVector("forgetGates");
        var candidateInputs = (ReadOnlySpan<float>)BackpropagationState.Get1dVector("candidateInputs");
        var candidates = (ReadOnlySpan<float>)BackpropagationState.Get1dVector("candidates");

        // Für jede Zelle rückwärts (analog zu UpdateCell)
        for (int nodeIndex = 0; nodeIndex < OutputSize.TotalSize; nodeIndex++)
        {
            // Extrahiere gespeicherte Werte
            float prevState = VorLastUpdateNodeValues[nodeIndex];

            float forgetGateInput = forgetGateInputs[nodeIndex];
            float forgetGate = forgetGates[nodeIndex];

            float candidateInput = candidateInputs[nodeIndex];
            float candidate = candidates[nodeIndex];

            // Ableitungen der Aktivierungen
            float dForgetGate = ((IActivationWithDerivat)RecurrentActivation).Derivat(forgetGateInput);
            float dCandidate = ((IActivationWithDerivat)Activation).Derivat(candidateInput);

            // --- Schritt 2: Fehler auf forgetGate und candidate aufteilen ---
            float err = outputError[nodeIndex];
            float dNewState_dForget = -prevState + candidate;
            float dNewState_dCandidate = forgetGate;

            float errorForgetGate = err * dNewState_dForget * dForgetGate;
            float errorCandidate = err * dNewState_dCandidate * dCandidate;

            // --- Schritt 3: Gradienten berechnen und Fehler in die Inputs propagieren ---
            // Für ForgetGateWeights/CandidateWeights
            for (int inputIndex = 0; inputIndex < ForgetGateWeights.GetLength(1); inputIndex++)
            {
                // TODO: Echte Inputs zwischenspeichern, im Forward!
                float previousInput = inputs[inputIndex]; // Hole aus gespeicherten Inputs des letzten Forward!
                forgetGateWeightsGrad[nodeIndex, inputIndex] += previousInput * errorForgetGate;
                candidateWeightsGrad[nodeIndex, inputIndex] += previousInput * errorCandidate;

                // Fehler rückpropagieren auf Input
                inputError[inputIndex] +=
                    ForgetGateWeights[nodeIndex, inputIndex] * errorForgetGate +
                    CandidateWeights[nodeIndex, inputIndex] * errorCandidate;
            }
            // Rekurrente Gewichte & Bias
            forgetRecurrentWeightsGrad[nodeIndex] += prevState * errorForgetGate;
            candidateRecurrentWeightsGrad[nodeIndex] += prevState * forgetGate * errorCandidate;

            if (UseBias)
            {
                forgetBiasGrad![nodeIndex] += errorForgetGate;
                candidateBiasGrad![nodeIndex] += errorCandidate;
            }
        }

        // --- Schritt 4: Parameter-Update ---
        for (int i = 0; i < ForgetGateWeights.GetLength(0); i++)
            for (int j = 0; j < ForgetGateWeights.GetLength(1); j++)
                ForgetGateWeights[i, j] -= learningRate * forgetGateWeightsGrad[i, j];

        for (int i = 0; i < CandidateWeights.GetLength(0); i++)
            for (int j = 0; j < CandidateWeights.GetLength(1); j++)
                CandidateWeights[i, j] -= learningRate * candidateWeightsGrad[i, j];

        for (int i = 0; i < ForgetRecurrentWeights.Length; i++)
            ForgetRecurrentWeights[i] -= learningRate * forgetRecurrentWeightsGrad[i];
        for (int i = 0; i < CandidateRecurrentWeights.Length; i++)
            CandidateRecurrentWeights[i] -= learningRate * candidateRecurrentWeightsGrad[i];

        if (UseBias)
        {
            for (int i = 0; i < ForgetBiases.Length; i++)
                ForgetBiases[i] -= learningRate * forgetBiasGrad![i];
            for (int i = 0; i < CandidateBiases.Length; i++)
                CandidateBiases[i] -= learningRate * candidateBiasGrad![i];
        }

        BackpropagationState.Clear();
        return inputError;
    }
}