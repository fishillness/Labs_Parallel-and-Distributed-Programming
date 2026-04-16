using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Lab3_ProgramB;

internal class Program
{
    static void Main(string[] args)
    {
        string path = "..\\..\\..\\img\\";
        string[] images = { "img1.jpg", "img2.jpg", "img3.jpg" };
        int repeats = 3;

        foreach (string imgName in images)
        {
            string inputPath = path + imgName;
            List<double> times = new List<double>();
            Console.WriteLine($"Изображение {imgName}:");

            var (imageBytes, width, height) = LoadImage(inputPath);

            var gpu = InitializeCUDA();
            if (gpu == null)
                return;

            var (context, accelerator, intensityKernel, sobelXKernel, sobelYKernel,
                 magnitudeKernel, normalizeKernel) = gpu.Value;

            int pixelCount = width * height;
            int stride = width * 3; 
            int byteCount = stride * height;

            using var gpuImageBytes = accelerator.Allocate1D<byte>(byteCount);
            using var gpuIntensity = accelerator.Allocate1D<float>(pixelCount);
            using var gpuMX = accelerator.Allocate1D<float>(pixelCount);
            using var gpuMY = accelerator.Allocate1D<float>(pixelCount);
            using var gpuMR = accelerator.Allocate1D<float>(pixelCount);
            using var gpuOutput = accelerator.Allocate1D<byte>(pixelCount);

            gpuImageBytes.CopyFromCPU(imageBytes);

            for (int i = 0; i < repeats; i++)
            {
                Stopwatch sw = Stopwatch.StartNew();

                intensityKernel(pixelCount, gpuImageBytes.View, gpuIntensity.View, width, stride);
                accelerator.Synchronize();

                sobelXKernel(pixelCount, gpuIntensity.View, gpuMX.View, width, height);
                accelerator.Synchronize();

                sobelYKernel(pixelCount, gpuIntensity.View, gpuMY.View, width, height);
                accelerator.Synchronize();

                magnitudeKernel(pixelCount, gpuMX.View, gpuMY.View, gpuMR.View);
                accelerator.Synchronize();

                // Поиск  MRmax
                float[] mrCpu = new float[pixelCount];
                gpuMR.CopyToCPU(mrCpu);
                float maxMR = mrCpu.Max();
                Console.WriteLine($"MR max: {maxMR:F4}");

                normalizeKernel(pixelCount, gpuMR.View, gpuOutput.View, maxMR);
                accelerator.Synchronize();

                sw.Stop();
                double time = sw.Elapsed.TotalSeconds;
                times.Add(time);
                Console.WriteLine($"  [{i}]: {Math.Round(time, 2)}");
            }

            byte[] outputBytes = new byte[pixelCount];
            gpuOutput.CopyToCPU(outputBytes);

            SaveImage(outputBytes, width, height, path, imgName);

            context.Dispose();
            accelerator.Dispose();

            Console.WriteLine($"Для изображения {imgName} среднее время равно {Math.Round(times.Average(), 2)} \n");
        }
    }

    static (Context context, Accelerator accelerator,
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

    static (byte[] imageBytes, int width, int height) LoadImage(string inputPath)
    {
        using var image = Image.Load<Rgb24>(inputPath);
        int width = image.Width;
        int height = image.Height;

        byte[] imageBytes = new byte[width * 3 * height];
        image.CopyPixelDataTo(imageBytes);

        return (imageBytes, width, height);
    }

    static void SaveImage(byte[] outputBytes, int width, int height, string path, string imgName)
    {
        using var outputImage = Image.LoadPixelData<L8>(outputBytes, width, height);

        outputImage.Save(path + "new_" + imgName);
    }

    static void IntensityKernel(Index1D pixelIndex, ArrayView<byte> imageBytes, ArrayView<float> intensity, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int byteIdx = y * stride + x * 3;

        float r = imageBytes[byteIdx];     
        float g = imageBytes[byteIdx + 1]; 
        float b = imageBytes[byteIdx + 2]; 

        intensity[pixelIndex] = (r + g + b) / 3f;
    }
    // MX
    static void SobelXKernel(Index1D pixelIndex, ArrayView<float> intensity, ArrayView<float> mx, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1); 
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1); 
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1); 
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1); 

        // Применяем ядро Собеля X
        // [-1  0  1]
        // [-2  0  2]
        // [-1  0  1]
        float sum =
            -1f * intensity[yU * width + xL] + 1f * intensity[yU * width + xR] +
            -2f * intensity[y * width + xL] + 2f * intensity[y * width + xR] +
            -1f * intensity[yD * width + xL] + 1f * intensity[yD * width + xR];

        mx[pixelIndex] = sum;
    }
    // MY
    static void SobelYKernel(Index1D pixelIndex, ArrayView<float> intensity, ArrayView<float> my, int width, int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        // Применяем ядро Собеля Y
        // [-1  -2  -1]
        // [ 0   0   0]
        // [ 1   2   1]
        float sum =
            -1f * intensity[yU * width + xL] + -2f * intensity[yU * width + x] + -1f * intensity[yU * width + xR] +
             1f * intensity[yD * width + xL] + 2f * intensity[yD * width + x] + 1f * intensity[yD * width + xR];

        my[pixelIndex] = sum;
    }

    // MRv = √(MXv^2 + MYv^2)
    static void MagnitudeKernel(Index1D pixelIndex, ArrayView<float> mx, ArrayView<float> my, ArrayView<float> mr)
    {
        float mxVal = mx[pixelIndex];
        float myVal = my[pixelIndex];
        mr[pixelIndex] = (float)Math.Sqrt(mxVal * mxVal + myVal * myVal);
    }

    // MRv = MRv * 255 / MRMax
    static void NormalizeKernel(Index1D pixelIndex, ArrayView<float> mr, ArrayView<byte> output, float maxMR)
    {
        float normalized = mr[pixelIndex] * 255f / maxMR;
        output[pixelIndex] = (byte)IntrinsicMath.Clamp((int)normalized, 0, 255);
    }
}
