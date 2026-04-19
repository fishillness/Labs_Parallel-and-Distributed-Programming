using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Lab3_ProgramB;

internal class Program
{
    enum PixelType
    {
        Float32,
        Float16,
        Int32,
        Int16,
        UInt8
    }

    static void Main(string[] args)
    {
        string path = "..\\..\\..\\img\\";
        string[] images = { "img1.jpg", "img2.jpg"};
        int repeats = 3;

        var gpu = InitializeCUDA();
        if (gpu == null)
            return;

        var (context, accelerator,
             intensityKernelFloat, sobelXKernelFloat, sobelYKernelFloat,
             magnitudeKernelFloat, normalizeKernelFloat) = gpu.Value;

        foreach (string imgName in images)
        {
            Console.WriteLine($"\nИзображение {imgName}:");

            string inputPath = path + imgName;
            var (imageBytes, width, height) = LoadImage(inputPath);

            int pixelCount = width * height;
            int stride = width * 3;
            int byteCount = stride * height;

            PixelType[] types =
            {
                PixelType.Float32,
                PixelType.Float16,
                PixelType.Int32,
                PixelType.Int16,
                PixelType.UInt8
            };

            foreach (var type in types)
            {
                Console.WriteLine($"\n  Тип данных: {type}");
                List<double> times = new();

                using var gpuImageBytes = accelerator.Allocate1D<byte>(byteCount);
                gpuImageBytes.CopyFromCPU(imageBytes);

                switch (type)
                {
                    case PixelType.Float32:
                        {
                            using var gpuIntensity = accelerator.Allocate1D<float>(pixelCount);
                            using var gpuMX = accelerator.Allocate1D<float>(pixelCount);
                            using var gpuMY = accelerator.Allocate1D<float>(pixelCount);
                            using var gpuMR = accelerator.Allocate1D<float>(pixelCount);
                            using var gpuOutput = accelerator.Allocate1D<byte>(pixelCount);

                            for (int i = 0; i < repeats; i++)
                            {
                                var sw = Stopwatch.StartNew();

                                intensityKernelFloat(pixelCount, gpuImageBytes.View, gpuIntensity.View, width, stride);
                                accelerator.Synchronize();

                                sobelXKernelFloat(pixelCount, gpuIntensity.View, gpuMX.View, width, height);
                                accelerator.Synchronize();

                                sobelYKernelFloat(pixelCount, gpuIntensity.View, gpuMY.View, width, height);
                                accelerator.Synchronize();

                                magnitudeKernelFloat(pixelCount, gpuMX.View, gpuMY.View, gpuMR.View);
                                accelerator.Synchronize();

                                float[] mrCpu = new float[pixelCount];
                                gpuMR.CopyToCPU(mrCpu);
                                float maxMR = mrCpu.Max();

                                normalizeKernelFloat(pixelCount, gpuMR.View, gpuOutput.View, maxMR);
                                accelerator.Synchronize();

                                sw.Stop();
                                double time = sw.Elapsed.TotalSeconds;
                                times.Add(time);
                                Console.WriteLine($"    [{i}] float32: {Math.Round(time, 4)} c");
                            }

                            byte[] outputBytes = new byte[pixelCount];
                            gpuOutput.CopyToCPU(outputBytes);
                            SaveImage(outputBytes, width, height, path, $"{type}_{imgName}");

                            Console.WriteLine($"  Среднее время для {type}: {Math.Round(times.Average(), 4)} c");
                            break;
                        }

                    case PixelType.Int32:
                        {
                            using var gpuIntensity = accelerator.Allocate1D<int>(pixelCount);
                            using var gpuMX = accelerator.Allocate1D<int>(pixelCount);
                            using var gpuMY = accelerator.Allocate1D<int>(pixelCount);
                            using var gpuMR = accelerator.Allocate1D<int>(pixelCount);
                            using var gpuOutput = accelerator.Allocate1D<byte>(pixelCount);

                            var intensityKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<byte>, ArrayView<int>, int, int>(IntensityKernelInt);
                            var sobelXKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<int>, ArrayView<int>, int, int>(SobelXKernelInt);
                            var sobelYKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<int>, ArrayView<int>, int, int>(SobelYKernelInt);
                            var magnitudeKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(MagnitudeKernelInt);
                            var normalizeKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<int>, ArrayView<byte>, int>(NormalizeKernelInt);

                            for (int i = 0; i < repeats; i++)
                            {
                                var sw = Stopwatch.StartNew();

                                intensityKernel(pixelCount, gpuImageBytes.View, gpuIntensity.View, width, stride);
                                accelerator.Synchronize();

                                sobelXKernel(pixelCount, gpuIntensity.View, gpuMX.View, width, height);
                                accelerator.Synchronize();

                                sobelYKernel(pixelCount, gpuIntensity.View, gpuMY.View, width, height);
                                accelerator.Synchronize();

                                magnitudeKernel(pixelCount, gpuMX.View, gpuMY.View, gpuMR.View);
                                accelerator.Synchronize();

                                int[] mrCpu = new int[pixelCount];
                                gpuMR.CopyToCPU(mrCpu);
                                int maxMR = mrCpu.Max();

                                normalizeKernel(pixelCount, gpuMR.View, gpuOutput.View, maxMR);
                                accelerator.Synchronize();

                                sw.Stop();
                                double time = sw.Elapsed.TotalSeconds;
                                times.Add(time);
                                Console.WriteLine($"    [{i}] int32: {Math.Round(time, 4)} c");
                            }

                            byte[] outputBytes = new byte[pixelCount];
                            gpuOutput.CopyToCPU(outputBytes);
                            SaveImage(outputBytes, width, height, path, $"{type}_{imgName}");

                            Console.WriteLine($"  Среднее время для {type}: {Math.Round(times.Average(), 4)} c");
                            break;
                        }

                    case PixelType.Int16:
                        {
                            using var gpuIntensity = accelerator.Allocate1D<short>(pixelCount);
                            using var gpuMX = accelerator.Allocate1D<short>(pixelCount);
                            using var gpuMY = accelerator.Allocate1D<short>(pixelCount);
                            using var gpuMR = accelerator.Allocate1D<short>(pixelCount);
                            using var gpuOutput = accelerator.Allocate1D<byte>(pixelCount);

                            var intensityKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<byte>, ArrayView<short>, int, int>(IntensityKernelInt16);
                            var sobelXKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<short>, ArrayView<short>, int, int>(SobelXKernelInt16);
                            var sobelYKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<short>, ArrayView<short>, int, int>(SobelYKernelInt16);
                            var magnitudeKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<short>, ArrayView<short>, ArrayView<short>>(MagnitudeKernelInt16);
                            var normalizeKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<short>, ArrayView<byte>, short>(NormalizeKernelInt16);

                            for (int i = 0; i < repeats; i++)
                            {
                                var sw = Stopwatch.StartNew();

                                intensityKernel(pixelCount, gpuImageBytes.View, gpuIntensity.View, width, stride);
                                accelerator.Synchronize();

                                sobelXKernel(pixelCount, gpuIntensity.View, gpuMX.View, width, height);
                                accelerator.Synchronize();

                                sobelYKernel(pixelCount, gpuIntensity.View, gpuMY.View, width, height);
                                accelerator.Synchronize();

                                magnitudeKernel(pixelCount, gpuMX.View, gpuMY.View, gpuMR.View);
                                accelerator.Synchronize();

                                short[] mrCpu = new short[pixelCount];
                                gpuMR.CopyToCPU(mrCpu);
                                short maxMR = mrCpu.Max();

                                normalizeKernel(pixelCount, gpuMR.View, gpuOutput.View, maxMR);
                                accelerator.Synchronize();

                                sw.Stop();
                                double time = sw.Elapsed.TotalSeconds;
                                times.Add(time);
                                Console.WriteLine($"    [{i}] int16: {Math.Round(time, 4)} c");
                            }

                            byte[] outputBytes = new byte[pixelCount];
                            gpuOutput.CopyToCPU(outputBytes);
                            SaveImage(outputBytes, width, height, path, $"{type}_{imgName}");

                            Console.WriteLine($"  Среднее время для {type}: {Math.Round(times.Average(), 4)} c");
                            break;
                        }

                    case PixelType.UInt8:
                        {
                            using var gpuIntensity = accelerator.Allocate1D<byte>(pixelCount);
                            using var gpuMX = accelerator.Allocate1D<byte>(pixelCount);
                            using var gpuMY = accelerator.Allocate1D<byte>(pixelCount);
                            using var gpuMR = accelerator.Allocate1D<byte>(pixelCount);
                            using var gpuOutput = accelerator.Allocate1D<byte>(pixelCount);

                            var intensityKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<byte>, ArrayView<byte>, int, int>(IntensityKernelByte);
                            var sobelXKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<byte>, ArrayView<byte>, int, int>(SobelXKernelByte);
                            var sobelYKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<byte>, ArrayView<byte>, int, int>(SobelYKernelByte);
                            var magnitudeKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>>(MagnitudeKernelByte);
                            var normalizeKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<byte>, ArrayView<byte>, byte>(NormalizeKernelByte);

                            for (int i = 0; i < repeats; i++)
                            {
                                var sw = Stopwatch.StartNew();

                                intensityKernel(pixelCount, gpuImageBytes.View, gpuIntensity.View, width, stride);
                                accelerator.Synchronize();

                                sobelXKernel(pixelCount, gpuIntensity.View, gpuMX.View, width, height);
                                accelerator.Synchronize();

                                sobelYKernel(pixelCount, gpuIntensity.View, gpuMY.View, width, height);
                                accelerator.Synchronize();

                                magnitudeKernel(pixelCount, gpuMX.View, gpuMY.View, gpuMR.View);
                                accelerator.Synchronize();

                                byte[] mrCpu = new byte[pixelCount];
                                gpuMR.CopyToCPU(mrCpu);
                                byte maxMR = mrCpu.Max();

                                normalizeKernel(pixelCount, gpuMR.View, gpuOutput.View, maxMR);
                                accelerator.Synchronize();

                                sw.Stop();
                                double time = sw.Elapsed.TotalSeconds;
                                times.Add(time);
                                Console.WriteLine($"    [{i}] uint8: {Math.Round(time, 4)} c");
                            }

                            byte[] outputBytes = new byte[pixelCount];
                            gpuOutput.CopyToCPU(outputBytes);
                            SaveImage(outputBytes, width, height, path, $"{type}_{imgName}");

                            Console.WriteLine($"  Среднее время для {type}: {Math.Round(times.Average(), 4)} c");
                            break;
                        }

                    case PixelType.Float16:
                        {
                            using var gpuIntensity = accelerator.Allocate1D<ushort>(pixelCount);
                            using var gpuMX = accelerator.Allocate1D<ushort>(pixelCount);
                            using var gpuMY = accelerator.Allocate1D<ushort>(pixelCount);
                            using var gpuMR = accelerator.Allocate1D<ushort>(pixelCount);
                            using var gpuOutput = accelerator.Allocate1D<byte>(pixelCount);

                            var intensityKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<byte>, ArrayView<ushort>, int, int>(IntensityKernelUShort);
                            var sobelXKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<ushort>, ArrayView<ushort>, int, int>(SobelXKernelUShort);
                            var sobelYKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<ushort>, ArrayView<ushort>, int, int>(SobelYKernelUShort);
                            var magnitudeKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<ushort>, ArrayView<ushort>, ArrayView<ushort>>(MagnitudeKernelUShort);
                            var normalizeKernel = accelerator.LoadAutoGroupedStreamKernel<
                                Index1D, ArrayView<ushort>, ArrayView<byte>, ushort>(NormalizeKernelUShort);

                            for (int i = 0; i < repeats; i++)
                            {
                                var sw = Stopwatch.StartNew();

                                intensityKernel(pixelCount, gpuImageBytes.View, gpuIntensity.View, width, stride);
                                accelerator.Synchronize();

                                sobelXKernel(pixelCount, gpuIntensity.View, gpuMX.View, width, height);
                                accelerator.Synchronize();

                                sobelYKernel(pixelCount, gpuIntensity.View, gpuMY.View, width, height);
                                accelerator.Synchronize();

                                magnitudeKernel(pixelCount, gpuMX.View, gpuMY.View, gpuMR.View);
                                accelerator.Synchronize();

                                ushort[] mrCpu = new ushort[pixelCount];
                                gpuMR.CopyToCPU(mrCpu);
                                ushort maxMR = mrCpu.Max();

                                normalizeKernel(pixelCount, gpuMR.View, gpuOutput.View, maxMR);
                                accelerator.Synchronize();

                                sw.Stop();
                                double time = sw.Elapsed.TotalSeconds;
                                times.Add(time);
                                Console.WriteLine($"    [{i}] float16(ushort): {Math.Round(time, 4)} c");
                            }

                            byte[] outputBytes = new byte[pixelCount];
                            gpuOutput.CopyToCPU(outputBytes);
                            SaveImage(outputBytes, width, height, path, $"{type}_{imgName}");

                            Console.WriteLine($"  Среднее время для {type}: {Math.Round(times.Average(), 4)} c");
                            break;
                        }
                }

                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        accelerator.Dispose();
        context.Dispose();
    }

    private static (Context context, Accelerator accelerator,
            Action<Index1D, ArrayView<byte>, ArrayView<float>, int, int> intensityKernel,
            Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> sobelXKernel,
            Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> sobelYKernel,
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> magnitudeKernel,
            Action<Index1D, ArrayView<float>, ArrayView<byte>, float> normalizeKernel
    )? InitializeCUDA()
    {
        Context context = Context.CreateDefault();

        Device? cudaDevice = context.Devices
            .Where(d => d.AcceleratorType == AcceleratorType.Cuda)
            .OrderByDescending(d => d.MaxNumThreads)
            .FirstOrDefault();

        if (cudaDevice == null)
        {
            Console.WriteLine("Устройство CUDA не найдено");
            context.Dispose();
            return null;
        }

        Console.WriteLine($"CUDA device: {cudaDevice.Name}");

        Accelerator accelerator = cudaDevice.CreateAccelerator(context);

        var intensityKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<float>, int, int>(IntensityKernel);
        var sobelXKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(SobelXKernel);
        var sobelYKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(SobelYKernel);
        var magnitudeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(MagnitudeKernel);
        var normalizeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<byte>, float>(NormalizeKernel);

        return (context, accelerator, intensityKernel, sobelXKernel, sobelYKernel, magnitudeKernel, normalizeKernel);
    }

    private static (byte[] imageBytes, int width, int height) LoadImage(string inputPath)
    {
        using var image = Image.Load<Rgb24>(inputPath);
        int width = image.Width;
        int height = image.Height;

        byte[] imageBytes = new byte[width * 3 * height];
        image.CopyPixelDataTo(imageBytes);

        return (imageBytes, width, height);
    }

    private static void SaveImage(byte[] outputBytes, int width, int height, string path, string imgName)
    {
        using var outputImage = Image.LoadPixelData<L8>(outputBytes, width, height);
        outputImage.Save(path + "new_" + imgName);
    }

    #region  IntensityKernel

    private static void IntensityKernel(Index1D pixelIndex, ArrayView<byte> imageBytes, ArrayView<float> intensity, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int byteIdx = y * stride + x * 3;

        float r = imageBytes[byteIdx];
        float g = imageBytes[byteIdx + 1];
        float b = imageBytes[byteIdx + 2];

        intensity[pixelIndex] = (r + g + b) / 3f;
    }
    private static void IntensityKernelInt(Index1D pixelIndex, ArrayView<byte> imageBytes, ArrayView<int> intensity, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int byteIdx = y * stride + x * 3;

        int r = imageBytes[byteIdx];
        int g = imageBytes[byteIdx + 1];
        int b = imageBytes[byteIdx + 2];

        intensity[pixelIndex] = (r + g + b) / 3;
    }
    private static void IntensityKernelInt16(Index1D pixelIndex, ArrayView<byte> imageBytes, ArrayView<short> intensity, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int byteIdx = y * stride + x * 3;

        short r = imageBytes[byteIdx];
        short g = imageBytes[byteIdx + 1];
        short b = imageBytes[byteIdx + 2];

        intensity[pixelIndex] = (short)((r + g + b) / 3);
    }
    private static void IntensityKernelByte(Index1D pixelIndex, ArrayView<byte> imageBytes, ArrayView<byte> intensity, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int byteIdx = y * stride + x * 3;

        int r = imageBytes[byteIdx];
        int g = imageBytes[byteIdx + 1];
        int b = imageBytes[byteIdx + 2];

        intensity[pixelIndex] = (byte)((r + g + b) / 3);
    }
    private static void IntensityKernelUShort(Index1D pixelIndex, ArrayView<byte> imageBytes, ArrayView<ushort> intensity, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int byteIdx = y * stride + x * 3;

        float r = imageBytes[byteIdx];
        float g = imageBytes[byteIdx + 1];
        float b = imageBytes[byteIdx + 2];

        float val = (r + g + b) / 3f;
        intensity[pixelIndex] = (ushort)val;
    }
    #endregion

    #region  SobelXKernel
    private static void SobelXKernel(Index1D pixelIndex, ArrayView<float> intensity, ArrayView<float> mx, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        float sum =
            -1f * intensity[yU * width + xL] + 1f * intensity[yU * width + xR] +
            -2f * intensity[y * width + xL] + 2f * intensity[y * width + xR] +
            -1f * intensity[yD * width + xL] + 1f * intensity[yD * width + xR];

        mx[pixelIndex] = sum;
    }
    private static void SobelXKernelInt(Index1D pixelIndex, ArrayView<int> intensity, ArrayView<int> mx, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        int sum =
            -1 * intensity[yU * width + xL] + 1 * intensity[yU * width + xR] +
            -2 * intensity[y * width + xL] + 2 * intensity[y * width + xR] +
            -1 * intensity[yD * width + xL] + 1 * intensity[yD * width + xR];

        mx[pixelIndex] = sum;
    }
    private static void SobelXKernelInt16(Index1D pixelIndex, ArrayView<short> intensity, ArrayView<short> mx, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        int sum =
            -1 * intensity[yU * width + xL] + 1 * intensity[yU * width + xR] +
            -2 * intensity[y * width + xL] + 2 * intensity[y * width + xR] +
            -1 * intensity[yD * width + xL] + 1 * intensity[yD * width + xR];

        mx[pixelIndex] = (short)sum;
    }
    private static void SobelXKernelByte(Index1D pixelIndex, ArrayView<byte> intensity, ArrayView<byte> mx, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        int sum =
            -1 * intensity[yU * width + xL] + 1 * intensity[yU * width + xR] +
            -2 * intensity[y * width + xL] + 2 * intensity[y * width + xR] +
            -1 * intensity[yD * width + xL] + 1 * intensity[yD * width + xR];

        sum = IntrinsicMath.Clamp(sum, 0, 255);
        mx[pixelIndex] = (byte)sum;
    }
    private static void SobelXKernelUShort(Index1D pixelIndex, ArrayView<ushort> intensity, ArrayView<ushort> mx, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        float sum =
            -1f * intensity[yU * width + xL] + 1f * intensity[yU * width + xR] +
            -2f * intensity[y * width + xL] + 2f * intensity[y * width + xR] +
            -1f * intensity[yD * width + xL] + 1f * intensity[yD * width + xR];

        sum = IntrinsicMath.Clamp(sum, 0f, 65535f);
        mx[pixelIndex] = (ushort)sum;
    }
    #endregion

    #region SobelYKernel
    private static void SobelYKernel(Index1D pixelIndex, ArrayView<float> intensity, ArrayView<float> my, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        float sum =
            -1f * intensity[yU * width + xL] + -2f * intensity[yU * width + x] + -1f * intensity[yU * width + xR] +
             1f * intensity[yD * width + xL] + 2f * intensity[yD * width + x] + 1f * intensity[yD * width + xR];

        my[pixelIndex] = sum;
    }

    private static void SobelYKernelInt(Index1D pixelIndex, ArrayView<int> intensity, ArrayView<int> my, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        int sum =
            -1 * intensity[yU * width + xL] + -2 * intensity[yU * width + x] + -1 * intensity[yU * width + xR] +
             1 * intensity[yD * width + xL] + 2 * intensity[yD * width + x] + 1 * intensity[yD * width + xR];

        my[pixelIndex] = sum;
    }
    private static void SobelYKernelInt16(Index1D pixelIndex, ArrayView<short> intensity, ArrayView<short> my, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        int sum =
            -1 * intensity[yU * width + xL] + -2 * intensity[yU * width + x] + -1 * intensity[yU * width + xR] +
             1 * intensity[yD * width + xL] + 2 * intensity[yD * width + x] + 1 * intensity[yD * width + xR];

        my[pixelIndex] = (short)sum;
    }

    private static void SobelYKernelByte(Index1D pixelIndex, ArrayView<byte> intensity, ArrayView<byte> my, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        int sum =
            -1 * intensity[yU * width + xL] + -2 * intensity[yU * width + x] + -1 * intensity[yU * width + xR] +
             1 * intensity[yD * width + xL] + 2 * intensity[yD * width + x] + 1 * intensity[yD * width + xR];

        sum = IntrinsicMath.Clamp(sum, 0, 255);
        my[pixelIndex] = (byte)sum;
    }
    private static void SobelYKernelUShort(Index1D pixelIndex, ArrayView<ushort> intensity, ArrayView<ushort> my, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        float sum =
            -1f * intensity[yU * width + xL] + -2f * intensity[yU * width + x] + -1f * intensity[yU * width + xR] +
             1f * intensity[yD * width + xL] + 2f * intensity[yD * width + x] + 1f * intensity[yD * width + xR];

        sum = IntrinsicMath.Clamp(sum, 0f, 65535f);
        my[pixelIndex] = (ushort)sum;
    }
    #endregion

    #region MagnitudeKernel
    private static void MagnitudeKernel(Index1D pixelIndex, ArrayView<float> mx, ArrayView<float> my, ArrayView<float> mr)
    {
        float mxVal = mx[pixelIndex];
        float myVal = my[pixelIndex];
        mr[pixelIndex] = MathF.Sqrt(mxVal * mxVal + myVal * myVal);
    }
    private static void MagnitudeKernelInt(Index1D pixelIndex, ArrayView<int> mx, ArrayView<int> my, ArrayView<int> mr)
    {
        int mxVal = mx[pixelIndex];
        int myVal = my[pixelIndex];
        int mag = (int)MathF.Sqrt(mxVal * mxVal + myVal * myVal);
        mr[pixelIndex] = mag;
    }

    private static void MagnitudeKernelInt16(Index1D pixelIndex, ArrayView<short> mx, ArrayView<short> my, ArrayView<short> mr)
    {
        int mxVal = mx[pixelIndex];
        int myVal = my[pixelIndex];
        int mag = (int)MathF.Sqrt(mxVal * mxVal + myVal * myVal);
        mag = IntrinsicMath.Clamp(mag, 0, 65535);
        mr[pixelIndex] = (short)mag;
    }

    private static void MagnitudeKernelByte(Index1D pixelIndex, ArrayView<byte> mx, ArrayView<byte> my, ArrayView<byte> mr)
    {
        int mxVal = mx[pixelIndex];
        int myVal = my[pixelIndex];
        int mag = (int)MathF.Sqrt(mxVal * mxVal + myVal * myVal);
        mag = IntrinsicMath.Clamp(mag, 0, 255);
        mr[pixelIndex] = (byte)mag;
    }
    private static void MagnitudeKernelUShort(Index1D pixelIndex, ArrayView<ushort> mx, ArrayView<ushort> my, ArrayView<ushort> mr)
    {
        float mxVal = mx[pixelIndex];
        float myVal = my[pixelIndex];
        float mag = MathF.Sqrt(mxVal * mxVal + myVal * myVal);
        mag = IntrinsicMath.Clamp(mag, 0f, 65535f);
        mr[pixelIndex] = (ushort)mag;
    }
    #endregion

    #region NormalizeKernel
    private static void NormalizeKernel(Index1D pixelIndex, ArrayView<float> mr, ArrayView<byte> output, float maxMR)
    {
        float normalized = mr[pixelIndex] * 255f / maxMR;
        output[pixelIndex] = (byte)IntrinsicMath.Clamp((int)normalized, 0, 255);
    }
    private static void NormalizeKernelInt(Index1D pixelIndex, ArrayView<int> mr, ArrayView<byte> output, int maxMR)
    {
        float normalized = mr[pixelIndex] * 255f / maxMR;
        output[pixelIndex] = (byte)IntrinsicMath.Clamp((int)normalized, 0, 255);
    }
    private static void NormalizeKernelInt16(Index1D pixelIndex, ArrayView<short> mr, ArrayView<byte> output, short maxMR)
    {
        float normalized = mr[pixelIndex] * 255f / maxMR;
        output[pixelIndex] = (byte)IntrinsicMath.Clamp((int)normalized, 0, 255);
    }
    private static void NormalizeKernelByte(Index1D pixelIndex, ArrayView<byte> mr, ArrayView<byte> output, byte maxMR)
    {
        float normalized = mr[pixelIndex] * 255f / maxMR;
        output[pixelIndex] = (byte)IntrinsicMath.Clamp((int)normalized, 0, 255);
    }
    private static void NormalizeKernelUShort(Index1D pixelIndex, ArrayView<ushort> mr, ArrayView<byte> output, ushort maxMR)
    {
        float normalized = mr[pixelIndex] * 255f / maxMR;
        output[pixelIndex] = (byte)IntrinsicMath.Clamp((int)normalized, 0, 255);
    }
    #endregion
}
