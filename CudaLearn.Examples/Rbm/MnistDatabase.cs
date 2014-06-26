using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace CudaLearn.Examples.Rbm
{

    public class MnistDataset
    {
        private readonly FileInfo imagesFile;
        private readonly FileInfo labelsFile;

        public MnistDataset( FileInfo images, FileInfo labels )
        {
            // Uses the files from http://yann.lecun.com/exdb/mnist/
            Contract.Requires<ArgumentNullException>(images != null);
            Contract.Requires<FileNotFoundException>(images.Exists);
            Contract.Requires<ArgumentNullException>(labels != null);
            Contract.Requires<FileNotFoundException>(labels.Exists);

            this.imagesFile = images;
            this.labelsFile = labels;

            using (var imagesStream = new BinaryReader(imagesFile.OpenRead()))
            {
                Console.WriteLine(BitConverter.IsLittleEndian);

                imagesStream.ReadInt32(); // discard

                Samples = IPAddress.NetworkToHostOrder(imagesStream.ReadInt32());

                int numRows = IPAddress.NetworkToHostOrder(imagesStream.ReadInt32());
                int numCols = IPAddress.NetworkToHostOrder(imagesStream.ReadInt32());

                SampleSize = numRows * numCols;
            }
        }

        public int Samples { get; private set; }

        public int SampleSize { get; private set; }

        public IEnumerable<Tuple<Matrix<float>, Matrix<int>>> Batches(int batchSize)
        {
            using (var imagesStream = new BinaryReader(imagesFile.OpenRead(), Encoding.BigEndianUnicode))
            using (var labelsStream = new BinaryReader(labelsFile.OpenRead(), Encoding.BigEndianUnicode))
            {
                imagesStream.ReadInt32(); // discard
                imagesStream.ReadInt32(); // discard
                imagesStream.ReadInt32(); // discard
                imagesStream.ReadInt32();// discard

                int magic2 = labelsStream.ReadInt32();
                int numLabels = labelsStream.ReadInt32();

                int totalBatches = Samples / batchSize;                
                for (int batches = 0; batches < totalBatches; batches++)
                {
                    var images = new Matrix<float>(SampleSize, batchSize);
                    var labels = new Matrix<int>(1, batchSize);

                    for (int current = 0; current < batchSize; current++ )
                    {
                        for (int i = 0; i < SampleSize; i++)
                        {
                            float v = (float) imagesStream.ReadByte();
                            images[i, current] = v / 255.0f;
                        }
                            

                        labels[0, current] = (int)labelsStream.ReadByte();
                    }

                    yield return new Tuple<Matrix<float>, Matrix<int>>(images, labels);
                }
            }
        }

        [ContractInvariantMethod]
        private void ObjectInvariants()
        {
            Contract.Invariant(this.imagesFile != null);
            Contract.Invariant(this.imagesFile.Exists);
            Contract.Invariant(this.labelsFile != null);
            Contract.Invariant(this.labelsFile.Exists);
        }
    }

    public class MnistDatabase
    {
        private readonly DirectoryInfo directory;

        public MnistDatabase( DirectoryInfo directory )
        {
            Contract.Requires<ArgumentNullException>(directory != null);
            Contract.Requires<DirectoryNotFoundException>( directory.Exists );

            this.directory = directory;
        }

        public MnistDataset GetTrainingSet()
        {
            return new MnistDataset(
                        new FileInfo(Path.Combine(directory.FullName, "train-images.idx3-ubyte")),
                        new FileInfo(Path.Combine(directory.FullName, "train-labels.idx1-ubyte")));
        }

        public MnistDataset GetValidationSet()
        {
            return new MnistDataset(
                        new FileInfo(Path.Combine(directory.FullName, "t10k-images.idx3-ubyte")),
                        new FileInfo(Path.Combine(directory.FullName, "t10k-labels.idx1-ubyte")));
        }


        [ContractInvariantMethod]
        private void ObjectInvariants()
        {
            Contract.Invariant(this.directory != null);
            Contract.Invariant(this.directory.Exists);
        }
    }
}
