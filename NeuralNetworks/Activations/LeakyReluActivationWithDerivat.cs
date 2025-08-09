using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class LeakyReluActivationWithDerivat : LeakyReluActivation, IActivationWithDerivat
    {
        public LeakyReluActivationWithDerivat(float alpha)
            : base(alpha)
        { }

        public float Derivat(float input)
        {
            return input > 0f 
                ? 1f 
                : alpha;
        }
    }
}
