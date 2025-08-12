﻿using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class GruLayer : IModelLayer
    {
        public ISize OutputSize { get; }
        public IActivation Activation { get; }
        public IActivation RecurrentActivation { get; }
        public bool UseBias { get; }
        public IInitializer KernelInitializer { get; }
        public IInitializer RecurrentInitializer { get; }
        public IInitializer BiasInitializer { get; }

        public NamedVectors<float> Parameters { get; } = new NamedVectors<float>();
        public NamedVectors<float> State { get; } = new NamedVectors<float>();

        public float[,] ForgetGateWeights { get; private set; } = default!;
        public float[,] CandidateWeights { get; private set; } = default!;
        public float[] NodeValues { get; private set; } = default!;
        public float[] ForgetRecurrentWeights { get; private set; } = default!;
        public float[] CandidateRecurrentWeights { get; private set; } = default!;
        public float[] ForgetBiases { get; private set; } = default!;
        public float[] CandidateBiases { get; private set; } = default!;

        public GruLayer(Size1D nodeCount,
            IActivation activation,
            IActivation recurrentActivation,
            bool useBias,
            IInitializer kernelInitializer,
            IInitializer recurrentInitializer,
            IInitializer biasInitializer)
        {
            if (nodeCount == null) throw new ArgumentNullException(nameof(nodeCount));

            OutputSize = nodeCount ?? throw new ArgumentNullException(nameof(nodeCount));
            Activation = activation ?? throw new ArgumentNullException(nameof(activation));
            RecurrentActivation = recurrentActivation ?? throw new ArgumentNullException(nameof(recurrentActivation));
            UseBias = useBias;
            KernelInitializer = kernelInitializer ?? throw new ArgumentNullException(nameof(kernelInitializer));
            RecurrentInitializer = recurrentInitializer ?? throw new ArgumentNullException(nameof(recurrentInitializer));
            BiasInitializer = biasInitializer ?? throw new ArgumentNullException(nameof(biasInitializer));
        }

        public bool IsBildet { get; private set; } = false;
        public void Build(ISize inputSize)
        {
            if (inputSize == null) throw new ArgumentNullException(nameof(inputSize));

            var inputs = inputSize.TotalSize;
            var nodes = OutputSize.TotalSize;

            ForgetGateWeights = KernelInitializer.GenerateValues2d(inputs, nodes, nodes, inputs);
            ForgetRecurrentWeights = RecurrentInitializer.GenerateValues(inputs, nodes, nodes);

            CandidateWeights = KernelInitializer.GenerateValues2d(inputs, nodes, nodes, inputs);
            CandidateRecurrentWeights = RecurrentInitializer.GenerateValues(inputs, nodes, nodes);

            NodeValues = new float[nodes];

            Parameters.Add("forgetGateWeights", ForgetGateWeights);
            Parameters.Add("forgetRecurrentWeights", ForgetRecurrentWeights);
            Parameters.Add("candidateWeights", CandidateWeights);
            Parameters.Add("candidateRecurrentWeights", CandidateRecurrentWeights);
            State.Add("nodeValues", NodeValues);

            if (UseBias)
            {
                CandidateBiases = BiasInitializer.GenerateValues(inputs, nodes, nodes);
                ForgetBiases = BiasInitializer.GenerateValues(inputs, nodes, nodes);
                Parameters.Add("forgetBiases", ForgetBiases);
                Parameters.Add("candidateBiases", CandidateBiases);
            }

            IsBildet = true;
        }

        public virtual float[] Update(float[] inputValues)
        {
            if (!IsBildet) throw new InvalidOperationException("Layer must be built before updating");

            for (int nodeIndex = 0; nodeIndex < OutputSize.TotalSize; nodeIndex++)
            {
                UpdateCell(nodeIndex, inputValues);
            }

            return NodeValues;
        }

        private void UpdateCell(int index, float[] inputs)
        {
            var forgetGateInput = Mutliply(ForgetGateWeights, inputs, index) + (NodeValues[index] * ForgetRecurrentWeights[index]);
            if (UseBias)
            {
                forgetGateInput += ForgetBiases[index];
            }
            var forgetGate = RecurrentActivation.Apply(forgetGateInput);

            var candidateInput = Mutliply(CandidateWeights, inputs, index) + (NodeValues[index] * CandidateRecurrentWeights[index] * forgetGate);
            if (UseBias)
            {
                candidateInput += CandidateBiases[index];
            }
            var candidate = Activation.Apply(candidateInput);
            NodeValues[index] = (1 - forgetGate) * NodeValues[index] + forgetGate * candidate;
        }

        private float Mutliply(float[,] weights, float[] inputs, int weightsIndex)
        {
            var result = 0f;
            for (int inputIndex = 0; inputIndex < inputs.Length; inputIndex++)
            {
                result += weights[weightsIndex, inputIndex] * inputs[inputIndex];
            }

            return result;
        }
    }
}
