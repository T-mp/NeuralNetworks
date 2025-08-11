using AwesomeAssertions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class DenseLayerWithBackpropagationTests
    {
        [Test]
        public void XORPattern_ShouldLearn()
        {
            var inputs = new[]
            {
                [0, 0],
                [0, 1],
                [1, 0],
                new float[] { 1, 1 }
            };
            var targets = new[] { 0f, 1f, 1f, 0f };

            var layer = new DenseLayerWithBackpropagation(2, new LinearActivationWithDerivat(), false, NN.Initializers.GlorotUniform(), NN.Initializers.Zeros());
            layer.Build(NN.Size.Of(inputs.Length));


            float initialLoss = 0f;

            for (int epoch = 0; epoch < 200; epoch++)
            {
                float loss = 0f;
                for (int i = 0; i < inputs.Length; i++)
                {
                    var output = layer.Update(inputs[i]);
                    float error = output[0] - targets[i];
                    loss += error * error;

                    layer.Backward([error], learningRate: 0.1f);
                }
                loss /= inputs.Length;
                if (epoch == 0) initialLoss = loss;
            }

            var testOut = layer.Update(new float[] { 0, 1 })[0];
            testOut.Should().BeGreaterThan(0.5f);
            initialLoss.Should().BeGreaterThan(0.5f);
        }
    }
}
