using ILGPU;
using ILGPU.Runtime;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;


internal class Program
{
    static void Main(string[] args)
    {
        string path = "..\\..\\..\\img\\";
        string[] images = { "img1.jpg", "img2.jpg", "img3.jpg" };
        int repeats = 3;


        foreach (string imgName in images)
        {
            List<double> times = new List<double>();
            Console.WriteLine($"Изображение {imgName}:");

            string inputPath = path + imgName;
            var (bitmap, imageBytes, stride, width, height) = LoadImage(inputPath);

            var gpu = InitializeOpenCL();
            if (gpu == null)
                return;

            var (context, accelerator, invertKernel, convolutionKernel) = gpu.Value;

            int pixelCount = width * height;
            int byteCount = stride * height;

            using MemoryBuffer1D<byte, Stride1D.Dense> gpuPixels = accelerator.Allocate1D<byte>(byteCount);
            using MemoryBuffer1D<byte, Stride1D.Dense> gpuResult = accelerator.Allocate1D<byte>(byteCount);

            gpuPixels.CopyFromCPU(imageBytes);

            for (int i = 0; i < repeats; i++)
            {
                Stopwatch sw = Stopwatch.StartNew();

                invertKernel(pixelCount, gpuPixels.View, width, stride);
                accelerator.Synchronize();

                convolutionKernel(pixelCount, gpuPixels.View, gpuResult.View, width, height, stride);
                accelerator.Synchronize();

                sw.Stop();
                double time = sw.Elapsed.TotalSeconds;
                times.Add(time);
                Console.WriteLine($"  [{i}]: {Math.Round(time, 2)}");
            }
            byte[] resultImageBytes = new byte[byteCount];
            gpuResult.CopyToCPU(resultImageBytes);

            SaveImage(bitmap, resultImageBytes, path, imgName);

            context.Dispose();
            accelerator.Dispose();

            Console.WriteLine($"Для изображения {imgName} среднее время равно {Math.Round(times.Average(), 2)} \n");
        }
    }

    static (Context context, Accelerator accelerator,
            Action<Index1D, ArrayView<byte>, int, int> invertKernel,
            Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int> convolutionKernel)?
            InitializeOpenCL()
    {
        Context context = Context.CreateDefault();

        Device? clDevice = context.Devices
            .Where(d => d.AcceleratorType == AcceleratorType.OpenCL)
            .OrderByDescending(d => d.MaxNumThreads)
            .FirstOrDefault();

        if (clDevice == null)
        {
            Console.WriteLine("Устройство OpenCL не найдено");
            context.Dispose();
            return null;
        }

        Console.WriteLine($"Устройство OpenCL: {clDevice.Name}");

        Accelerator accelerator = clDevice.CreateAccelerator(context);

        Action<Index1D, ArrayView<byte>, int, int> invertKernel = accelerator.LoadAutoGroupedStreamKernel
                                                       <Index1D, ArrayView<byte>, int, int>(InvertKernel);

        Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int> convolutionKernel = accelerator.LoadAutoGroupedStreamKernel
                                                       <Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int>(ConvolutionKernel);

        return (context, accelerator, invertKernel, convolutionKernel);
    }

    static (Bitmap bitmap, byte[] imageBytes, int stride, int width, int height) LoadImage(string inputPath)
    {
        Bitmap bitmap = new Bitmap(inputPath);
        int width = bitmap.Width;
        int height = bitmap.Height;

        BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0, width, height),
                                            ImageLockMode.ReadWrite,
                                            PixelFormat.Format24bppRgb);

        int stride = bmpData.Stride;
        int byteCount = stride * height;

        byte[] imageBytes = new byte[byteCount];
        Marshal.Copy(bmpData.Scan0, imageBytes, 0, byteCount);

        bitmap.UnlockBits(bmpData);

        return (bitmap, imageBytes, stride, width, height);
    }

    static void InvertKernel(Index1D pixelIndex, ArrayView<byte> imageBytes, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int idx = y * stride + x * 3;

        imageBytes[idx] = (byte)(255 - (int)imageBytes[idx]);
        imageBytes[idx + 1] = (byte)(255 - (int)imageBytes[idx + 1]);
        imageBytes[idx + 2] = (byte)(255 - (int)imageBytes[idx + 2]);
    }

    static void ConvolutionKernel(Index1D pixelIndex, ArrayView<byte> input, ArrayView<byte> output,
                                    int width, int height, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        for (int channel = 0; channel < 3; channel++)
        {
            int sum = 0;

            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int ny = IntrinsicMath.Clamp(y + ky, 0, height - 1);
                    int nx = IntrinsicMath.Clamp(x + kx, 0, width - 1);
                    int kernelVal = (ky == 0 && kx == 0) ? 9 : -1;

                    sum += (int)input[ny * stride + nx * 3 + channel] * kernelVal;
                }
            }

            output[y * stride + x * 3 + channel] = (byte)IntrinsicMath.Clamp(sum, 0, 255);
        }
    }

    static void SaveImage(Bitmap bitmap, byte[] resultImageBytes, string path, string imgName)
    {
        int width = bitmap.Width;
        int height = bitmap.Height;

        BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0, width, height),
                                             ImageLockMode.WriteOnly,
                                             PixelFormat.Format24bppRgb);

        Marshal.Copy(resultImageBytes, 0, bmpData.Scan0, resultImageBytes.Length);
        bitmap.UnlockBits(bmpData);

        bitmap.Save(path + "new_" + imgName, ImageFormat.Jpeg);
        bitmap.Dispose();
    }
}



