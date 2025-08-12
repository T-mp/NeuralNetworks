using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DropoutLayer : IModelLayer
    {
        public ISize OutputSize { get; private set; } = default!;
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }
        public float DropoutRate { get; }
        public IRandomProvider RandomProvider { get; }

        public bool IsBildet { get; private set; } = false;

        protected float[] nodeValues = default!;

        public DropoutLayer(float dropoutRate, IRandomProvider randomProvider)
        {
            DropoutRate = dropoutRate;
            RandomProvider = randomProvider;
            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public virtual void Build(ISize inputSize)
        {
            IsBildet = true;
            OutputSize = inputSize;
            nodeValues = new float[OutputSize.TotalSize];
            State.Add("nodeValues", nodeValues);
        }

        public virtual float[] Update(float[] inputValues)
        {
            if (!IsBildet) throw new InvalidOperationException("Layer must be built before updating");

            for (int i = 0; i < OutputSize.TotalSize; i++)
            {
                nodeValues[i] =  RandomProvider.NextBool(DropoutRate) ? inputValues[i] : 0;
            }

            return nodeValues;
        }
    }
}
