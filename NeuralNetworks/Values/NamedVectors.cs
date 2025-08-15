using System.Collections.Generic;

namespace Ivankarez.NeuralNetworks.Values
{
    public class NamedVectors<T>
    {
        private readonly IDictionary<string, T[]> vectors1d;
        private readonly IDictionary<string, T[,]> vectors2d;

        public NamedVectors()
        {
            vectors1d = new Dictionary<string, T[]>();
            vectors2d = new Dictionary<string, T[,]>();
        }

        public void Add(string name, T[] vector)
        {
            vectors1d.Add(name, vector);
        }

        public T[] Get1dVector(string name)
        {
            return vectors1d[name];
        }

        public void Add(string name, T[,] vector)
        {
            vectors2d.Add(name, vector);
        }

        public T[,] Get2dVector(string name)
        {
            return vectors2d[name];
        }
        public T[,] Get2dVectorCopy(string name)
        {
            var source = vectors2d[name];
            int dim0 = source.GetLength(0);
            int dim1 = source.GetLength(1);
            var copy = new T[dim0, dim1];

            for (int i = 0; i < dim0; i++)
                for (int j = 0; j < dim1; j++)
                    copy[i, j] = source[i, j];

            return copy;
        }

        public ICollection<string> Get1dVectorNames()
        {
            return vectors1d.Keys;
        }

        public ICollection<string> Get2dVectorNames()
        {
            return vectors2d.Keys;
        }
    }
}
