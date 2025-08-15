using AwesomeAssertions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration.Initializers;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test.Layers;

public class GruLayerWithBackpropagationTests
    {
        private GruLayerWithBackpropagation layer;
        private float[] input;
        private int inputSize;
        private int nodeCount;

        [SetUp]
        public void SetUp()
        {
            inputSize = 3;
            nodeCount = 2;

            input = new float[inputSize];
            for (int i = 0; i < input.Length; i++)
                input[i] = i + 1;

            var activation = new SigmoidActivationWithDerivat(); // Oder gewünschte Aktivierungen
            var recurrentActivation = new SigmoidActivationWithDerivat();
            var kernelInitializer = new ConstantInitializer(0.1f);
            var recurrentInitializer = new ConstantInitializer(0.1f);
            var biasInitializer = new ConstantInitializer(0.0f);

            layer = new GruLayerWithBackpropagation(
                new Size1D(nodeCount),
                activation,
                recurrentActivation,
                useBias: true,
                kernelInitializer,
                recurrentInitializer,
                biasInitializer
            );

            layer.Build(new Size1D(inputSize));
        }

        [Test]
        public void Update_ShouldProduceExpectedOutputLength()
        {
            var output = layer.Update(input);
            output.Length.Should().Be(nodeCount);
        }

        [Test]
        public void Update_ShouldStoreBackpropagationState()
        {
            layer.Update(input);

            layer.BackpropagationState.Should().NotBeNull();
            layer.BackpropagationState.ContainsKey1D("inputs").Should().BeTrue();
            layer.BackpropagationState.ContainsKey1D("VorLastUpdateNodeValues").Should().BeTrue();
            layer.BackpropagationState.ContainsKey1D("forgetGateInputs").Should().BeTrue();
            layer.BackpropagationState.ContainsKey1D("forgetGates").Should().BeTrue();
            layer.BackpropagationState.ContainsKey1D("candidateInputs").Should().BeTrue();
            layer.BackpropagationState.ContainsKey1D("candidates").Should().BeTrue();

            var forgetGates = layer.BackpropagationState.Get1dVector("forgetGates");
            forgetGates.Should().HaveCount(nodeCount);
        }

        [Test]
        public void Backward_ShouldReturnInputErrorOfCorrectLength()
        {
            layer.Update(input);
            var outputError = new float[nodeCount];
            for (int i = 0; i < outputError.Length; i++)
                outputError[i] = 0.5f;

            var inputError = layer.Backward(outputError, learningRate: 0.1f);
            inputError.Should().HaveCount(inputSize);
        }

        [Test]
        public void Backward_ShouldUpdateWeightsAndBiases()
        {
            layer.Update(input);
            var outputError = new float[nodeCount];
            for (int i = 0; i < outputError.Length; i++)
                outputError[i] = 1f;

            var forgetGateWeightsBefore = layer.Parameters.Get2dVectorCopy("forgetGateWeights");
            var candidateWeightsBefore = layer.Parameters.Get2dVectorCopy("candidateWeights");
            var forgetBiasesBefore = (float[])layer.Parameters.Get1dVector("forgetBiases").Clone();
            var candidateBiasesBefore = (float[])layer.Parameters.Get1dVector("candidateBiases").Clone();

            layer.Backward(outputError, learningRate: 0.1f);

            var forgetGateWeightsAfter = layer.Parameters.Get2dVector("forgetGateWeights");
            var candidateWeightsAfter = layer.Parameters.Get2dVector("candidateWeights");
            var forgetBiasesAfter = layer.Parameters.Get1dVector("forgetBiases");
            var candidateBiasesAfter = layer.Parameters.Get1dVector("candidateBiases");

            // Prüfe, dass sich Gewichte verändert haben
            forgetGateWeightsBefore.Should().NotBeEquivalentTo(forgetGateWeightsAfter);
            candidateWeightsBefore.Should().NotBeEquivalentTo(candidateWeightsAfter);
            forgetBiasesBefore.Should().NotEqual(forgetBiasesAfter);
            candidateBiasesBefore.Should().NotEqual(candidateBiasesAfter);
        }
    }