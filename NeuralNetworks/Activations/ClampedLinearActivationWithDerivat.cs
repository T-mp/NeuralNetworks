using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class ClampedLinearActivationWithDerivat : ClampedLinearActivation, IActivationWithDerivat
    {
        public ClampedLinearActivationWithDerivat(float min, float max)
            : base(min, max)
        { }

        public float Derivat(float input)
        {
            if (input <= Min || input >= Max)
            {
                return 0.0f;
            }
            return 1.0f;
        }
    }
}
