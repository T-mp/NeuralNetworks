using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers;

public class PoolingLayerWithBackpropagation : PoolingLayer, IModelLayerWithBackpropagation
{
    public PoolingLayerWithBackpropagation(int window, int stride, PoolingType type)
        : base(window, stride, type)
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
        BackpropagationState.Add("minMaxIds", new float[inputValues.Length]);

        return base.Update(inputValues);
    }

    public float[] Backward(float[] outputError, float learningRate)
    {
        if (outputError.Length != nodeValues.Length)
            throw new ArgumentException("Output error length must match the number of nodes in the pooling layer.", nameof(outputError));

        var inputValues = (ReadOnlySpan<float>)BackpropagationState.Get1dVector("inputs");
        var minMaxIds = (ReadOnlySpan<float>)BackpropagationState.Get1dVector("minMaxIds");

        var inputError = new float[inputValues.Length];
        for (int nodeIndex = 0; nodeIndex < nodeValues.Length; nodeIndex++)
        {
            var startIndex = nodeIndex * Stride;
            var windowEnd = Math.Min(startIndex + Window, inputError.Length);

            if (Type == PoolingType.Max)
            {
                // Fehler nur am maximalen Wertposition weiterreichen
                int maxIdx = (int)minMaxIds[startIndex];
                inputError[maxIdx] += outputError[nodeIndex];
            }
            else if (Type == PoolingType.Min)
            {
                int minIdx = (int)minMaxIds[startIndex];
                inputError[minIdx] += outputError[nodeIndex];
            }
            else if (Type == PoolingType.Average)
            {
                var share = outputError[nodeIndex] / (windowEnd - startIndex);
                for (int i = startIndex; i < windowEnd; i++)
                    inputError[i] += share;
            }
            else if (Type == PoolingType.Sum)
            {
                for (int i = startIndex; i < windowEnd; i++)
                    inputError[i] += outputError[nodeIndex];
            }
        }
        return inputError;
    }

    private int FindMaxIndex(ReadOnlySpan<float> inputValues, int start, int end)
    {
        var max = float.NegativeInfinity;
        int maxIdx = start;
        for (int i = start; i < end; i++)
        {
            var val = inputValues[i];
            if (val > max)
            {
                max = val;
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    private int FindMinIndex(ReadOnlySpan<float> inputValues, int start, int end)
    {
        var min = float.PositiveInfinity;
        int minIdx = start;
        for (int i = start; i < end; i++)
        {
            var val = inputValues[i];
            if (val < min)
            {
                min = val;
                minIdx = i;
            }
        }
        return minIdx;
    }
}