using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class SigmoidActivationWithRevert : SigmoidActivation, IActivationWithRevert
    {
        public float Revert(float input)
        {
            return input * (1f - input);
        }
    }
}
