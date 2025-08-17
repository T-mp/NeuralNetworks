using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers;

public class Pooling2dLayerWithBackpropagation : Pooling2dLayer, IModelLayerWithBackpropagation
{
    public Pooling2dLayerWithBackpropagation(Size2D windowSize, Stride2D stride, PoolingType poolingType)
        : base(windowSize, stride, poolingType)
    { }

    public NamedVectors<float> BackpropagationState { get; } = new NamedVectors<float>();
    protected override void BackpropagationStateSet(string name, int index, float value)
    {
        BackpropagationState.Get1dVector(name)[index] = value;
    }
    protected override void BackpropagationStateSet(string name, int index1, int index2 ,float value)
    {
        BackpropagationState.Get2dVector(name)[index1, index2] = value;
    }

    // Im Forward werden die relevanten Indizes gespeichert
    public override float[] Update(float[] inputValues)
    {
        if (!IsBildet) throw new InvalidOperationException("Layer must be built before updating");

        BackpropagationState.Clear();
        BackpropagationState.Add("inputs", inputValues);
        BackpropagationState.Add("minMaxIds", new float[OutputSize[0], OutputSize[1]]);

        return base.Update(inputValues);
    }

    public float[] Backward(float[] outputError, float learningRate)
    {
        var minMaxIds = BackpropagationState.Get2dVector("minMaxIds");

        // Fehler für Input-Shape
        float[] inputError = new float[InputSize.TotalSize];

        for (int nodeX = 0; nodeX < nodeValuesWidth; nodeX++)
        {
            for (int nodeY = 0; nodeY < nodeValuesHeight; nodeY++)
            {
                int nodeIndex = nodeX * nodeValuesHeight + nodeY;

                if (PoolingType == PoolingType.Max)
                {
                    int inIdx = (int)minMaxIds[nodeX, nodeY];
                    inputError[inIdx] += outputError[nodeIndex];
                }
                else if (PoolingType == PoolingType.Min)
                {
                    int inIdx = (int)minMaxIds[nodeX, nodeY];
                    inputError[inIdx] += outputError[nodeIndex];
                }
                else if (PoolingType == PoolingType.Sum)
                {
                    for (int fx = 0; fx < WindowSize.Width; fx++)
                    {
                        for (int fy = 0; fy < WindowSize.Height; fy++)
                        {
                            int inputX = nodeX * Stride.Horizontal + fx;
                            int inputY = nodeY * Stride.Vertical + fy;
                            int inputIdxFlat = inputX * InputSize.Width + inputY;
                            inputError[inputIdxFlat] += outputError[nodeIndex];
                        }
                    }
                }
                else if (PoolingType == PoolingType.Average)
                {
                    float share = outputError[nodeIndex] / (WindowSize.Width * WindowSize.Height);
                    for (int fx = 0; fx < WindowSize.Width; fx++)
                    {
                        for (int fy = 0; fy < WindowSize.Height; fy++)
                        {
                            int inputX = nodeX * Stride.Horizontal + fx;
                            int inputY = nodeY * Stride.Vertical + fy;
                            int inputIdxFlat = inputX * InputSize.Width + inputY;
                            inputError[inputIdxFlat] += share;
                        }
                    }
                }
            }
        }
        return inputError;
    }
}