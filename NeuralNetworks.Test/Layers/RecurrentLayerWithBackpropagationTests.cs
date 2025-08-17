using AwesomeAssertions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration.Initializers;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Layers;

public class RecurrentLayerWithBackpropagationTests
    {
        private RecurrentLayerWithBackpropagation layer;
        private readonly int nodeCount = 2;
        private readonly int inputSize = 3;
        private float[] input;

        [SetUp]
        public void SetUp()
        {
            input = new float[inputSize];
            for (int i = 0; i < inputSize; i++)
                input[i] = 0.1f + i;

            var activation = new SigmoidActivationWithDerivat(); // z.B. eigene Klasse
            var kernelInitializer = new ConstantInitializer(0.2f);
            var recurrentInitializer = new ConstantInitializer(0.1f);
            var biasInitializer = new ConstantInitializer(0.0f);

            layer = new RecurrentLayerWithBackpropagation(
                nodeCount, activation, true, kernelInitializer, biasInitializer, recurrentInitializer
            );

            layer.Build(new Size1D(inputSize));
        }

        [Test]
        public void Update_ReturnsCorrectOutputLength()
        {
            var output = layer.Update(input);
            output.Should().HaveCount(nodeCount);
        }

        [Test]
        public void Backward_ReturnsCorrectInputErrorLength()
        {
            layer.Update(input);
            var error = new float[nodeCount];
            for (int i = 0; i < nodeCount; i++) error[i] = 1.0f;
            var inputError = layer.Backward(error, learningRate: 0.01f);
            inputError.Should().HaveCount(inputSize);
        }

        [Test]
        public void Backward_UpdatesWeightsAndBiases()
        {
            layer.Update(input);
            var error = new float[nodeCount];
            for (int i = 0; i < nodeCount; i++) error[i] = 0.5f;

            var weightsBefore = (float[,])layer.Parameters.Get2dVector("weights");
            var recurrentBefore = (float[])layer.Parameters.Get1dVector("recurrentWeights");
            var biasesBefore = (float[])layer.Parameters.Get1dVector("biases");

            layer.Backward(error, 0.01f);

            var weightsAfter = (float[,])layer.Parameters.Get2dVector("weights");
            var recurrentAfter = (float[])layer.Parameters.Get1dVector("recurrentWeights");
            var biasesAfter = (float[])layer.Parameters.Get1dVector("biases");

            // Die Werte sollten sich ändern
            Assert.That(weightsBefore, Is.Not.EqualTo(weightsAfter));
            Assert.That(recurrentBefore, Is.Not.EqualTo(recurrentAfter));
            Assert.That(biasesBefore, Is.Not.EqualTo(biasesAfter));
        }
    }