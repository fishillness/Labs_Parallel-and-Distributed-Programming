using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

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
        string[] images = { "img1.jpg", "img2.jpg" };
        int repeats = 3;

        var gpu = InitializeOpenCL();
        if (gpu == null)
            return;

        var (context, accelerator) = gpu.Value;

        foreach (string imgName in images)
        {
            Console.WriteLine($"\nИзображение {imgName}:");

            string inputPath = path + imgName;
            var (bitmap, imageBytes, stride, width, height) = LoadImage(inputPath);

            int pixelCount = width * height;

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
                List<double> times = new List<double>();

                switch (type)
                {
                    case PixelType.UInt8:
                        {
                            byte[] input = (byte[])imageBytes.Clone();
                            RunForType(
                                accelerator,
                                pixelCount, width, height, stride,
                                input, repeats,
                                "uint8",
                                InvertKernelByte, ConvolutionKernelByte,
                                out byte[] resultBytes, times);
                            SaveImage(bitmap, resultBytes, path, $"{type}_{imgName}");
                            break;
                        }
                    case PixelType.Int16:
                        {
                            short[] input = BytesToInt16(imageBytes);
                            RunForType(
                                accelerator,
                                pixelCount, width, height, stride,
                                input, repeats,
                                "int16",
                                InvertKernelInt16, ConvolutionKernelInt16,
                                out short[] result, times);
                            byte[] resultBytes = Int16ToBytes(result);
                            SaveImage(bitmap, resultBytes, path, $"{type}_{imgName}");
                            break;
                        }
                    case PixelType.Int32:
                        {
                            int[] input = BytesToInt32(imageBytes);
                            RunForType(
                                accelerator,
                                pixelCount, width, height, stride,
                                input, repeats,
                                "int32",
                                InvertKernelInt32, ConvolutionKernelInt32,
                                out int[] result, times);
                            byte[] resultBytes = Int32ToBytes(result);
                            SaveImage(bitmap, resultBytes, path, $"{type}_{imgName}");
                            break;
                        }
                    case PixelType.Float32:
                        {
                            float[] input = BytesToFloat(imageBytes);
                            RunForType(
                                accelerator,
                                pixelCount, width, height, stride,
                                input, repeats,
                                "float32",
                                InvertKernelFloat, ConvolutionKernelFloat,
                                out float[] result, times);
                            byte[] resultBytes = FloatToBytes(result);
                            SaveImage(bitmap, resultBytes, path, $"{type}_{imgName}");
                            break;
                        }
                    case PixelType.Float16:
                        {
                            ushort[] input = BytesToUInt16(imageBytes); 
                            RunForType(
                                accelerator,
                                pixelCount, width, height, stride,
                                input, repeats,
                                "float16(ushort)",
                                InvertKernelUShort, ConvolutionKernelUShort,
                                out ushort[] result, times);
                            byte[] resultBytes = UInt16ToBytes(result);
                            SaveImage(bitmap, resultBytes, path, $"{type}_{imgName}");
                            break;
                        }
                }

                Console.WriteLine($"  Среднее время для {type}: {Math.Round(times.Average(), 4)} c");

                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            bitmap.Dispose();
        }

        accelerator.Dispose();
        context.Dispose();
    }

    private static (Context context, Accelerator accelerator)? InitializeOpenCL()
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
        return (context, accelerator);
    }

    private static (Bitmap bitmap, byte[] imageBytes, int stride, int width, int height) LoadImage(string inputPath)
    {
        Bitmap bitmap = new Bitmap(inputPath);
        int width = bitmap.Width;
        int height = bitmap.Height;

        BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0, width, height),
                             ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

        int stride = bmpData.Stride;
        int byteCount = stride * height;

        byte[] imageBytes = new byte[byteCount];
        Marshal.Copy(bmpData.Scan0, imageBytes, 0, byteCount);

        bitmap.UnlockBits(bmpData);

        return (bitmap, imageBytes, stride, width, height);
    }

    private static void SaveImage(Bitmap bitmap, byte[] resultImageBytes, string path, string imgName)
    {
        int width = bitmap.Width;
        int height = bitmap.Height;

        BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0, width, height),
                             ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

        Marshal.Copy(resultImageBytes, 0, bmpData.Scan0, resultImageBytes.Length);
        bitmap.UnlockBits(bmpData);

        string fullName = System.IO.Path.Combine(path, "new_" + imgName);
        bitmap.Save(fullName, ImageFormat.Jpeg);
    }
    #region BytesToT
    private static short[] BytesToInt16(byte[] src)
    {
        short[] dst = new short[src.Length];
        for (int i = 0; i < src.Length; i++)
            dst[i] = (short)src[i];
        return dst;
    }

    private static int[] BytesToInt32(byte[] src)
    {
        int[] dst = new int[src.Length];
        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i];
        return dst;
    }

    private static float[] BytesToFloat(byte[] src)
    {
        float[] dst = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i];
        return dst;
    }

    private static ushort[] BytesToUInt16(byte[] src)
    {
        ushort[] dst = new ushort[src.Length];
        for (int i = 0; i < src.Length; i++)
            dst[i] = (ushort)src[i];
        return dst;
    }
    #endregion

    #region TToBytes
    private static byte[] Int16ToBytes(short[] src)
    {
        byte[] dst = new byte[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            int v = src[i];
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            dst[i] = (byte)v;
        }
        return dst;
    }

    private static byte[] Int32ToBytes(int[] src)
    {
        byte[] dst = new byte[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            int v = src[i];
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            dst[i] = (byte)v;
        }
        return dst;
    }

    private static byte[] FloatToBytes(float[] src)
    {
        byte[] dst = new byte[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            float v = src[i];
            if (v < 0f) v = 0f;
            if (v > 255f) v = 255f;
            dst[i] = (byte)v;
        }
        return dst;
    }

    private static byte[] UInt16ToBytes(ushort[] src)
    {
        byte[] dst = new byte[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            int v = src[i];
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            dst[i] = (byte)v;
        }
        return dst;
    }
    #endregion

    private static void RunForType<T>(Accelerator accelerator,
        int pixelCount, int width, int height, int stride,
        T[] inputData, int repeats, string typeName,
        Action<Index1D, ArrayView<T>, int, int> invertKernelMethod,
        Action<Index1D, ArrayView<T>, ArrayView<T>, int, int, int> convolutionKernelMethod,
        out T[] resultData, List<double> times) where T : unmanaged
    {
        int elementCount = inputData.Length;

        using var gpuInput = accelerator.Allocate1D<T>(elementCount);
        using var gpuResult = accelerator.Allocate1D<T>(elementCount);

        gpuInput.CopyFromCPU(inputData);

        var invertKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<T>, int, int>(invertKernelMethod);

        var convolutionKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<T>, ArrayView<T>, int, int, int>(convolutionKernelMethod);

        for (int i = 0; i < repeats; i++)
        {
            var sw = Stopwatch.StartNew();

            invertKernel(pixelCount, gpuInput.View, width, stride);
            accelerator.Synchronize();

            convolutionKernel(pixelCount, gpuInput.View, gpuResult.View, width, height, stride);
            accelerator.Synchronize();

            sw.Stop();
            double time = sw.Elapsed.TotalSeconds;
            times.Add(time);
            Console.WriteLine($"    [{i}] {typeName}: {Math.Round(time, 4)} c");
        }

        resultData = new T[elementCount];
        gpuResult.CopyToCPU(resultData);
    }

    #region  InvertKernel
    private static void InvertKernelByte(Index1D pixelIndex, ArrayView<byte> imageBytes, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int idx = y * stride + x * 3;

        imageBytes[idx] = (byte)(255 - imageBytes[idx]);
        imageBytes[idx + 1] = (byte)(255 - imageBytes[idx + 1]);
        imageBytes[idx + 2] = (byte)(255 - imageBytes[idx + 2]);
    }

    private static void InvertKernelInt16(Index1D pixelIndex, ArrayView<short> imageBytes, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int idx = y * stride + x * 3;

        imageBytes[idx] = (short)(255 - imageBytes[idx]);
        imageBytes[idx + 1] = (short)(255 - imageBytes[idx + 1]);
        imageBytes[idx + 2] = (short)(255 - imageBytes[idx + 2]);
    }

    private static void InvertKernelInt32(Index1D pixelIndex, ArrayView<int> imageBytes, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int idx = y * stride + x * 3;

        imageBytes[idx] = 255 - imageBytes[idx];
        imageBytes[idx + 1] = 255 - imageBytes[idx + 1];
        imageBytes[idx + 2] = 255 - imageBytes[idx + 2];
    }

    private static void InvertKernelFloat(Index1D pixelIndex, ArrayView<float> imageBytes, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int idx = y * stride + x * 3;

        imageBytes[idx] = 255f - imageBytes[idx];
        imageBytes[idx + 1] = 255f - imageBytes[idx + 1];
        imageBytes[idx + 2] = 255f - imageBytes[idx + 2];
    }

    private static void InvertKernelUShort(Index1D pixelIndex, ArrayView<ushort> imageBytes, int width, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int idx = y * stride + x * 3;

        imageBytes[idx] = (ushort)(255 - imageBytes[idx]);
        imageBytes[idx + 1] = (ushort)(255 - imageBytes[idx + 1]);
        imageBytes[idx + 2] = (ushort)(255 - imageBytes[idx + 2]);
    }
    #endregion

    #region ConvolutionKernel
    private static void ConvolutionKernelByte(Index1D pixelIndex, ArrayView<byte> input,
        ArrayView<byte> output, int width, int height, int stride)
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

                    int idx = ny * stride + nx * 3 + channel;
                    sum += input[idx] * kernelVal;
                }
            }

            int outIdx = y * stride + x * 3 + channel;
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[outIdx] = (byte)sum;
        }
    }

    private static void ConvolutionKernelInt16(Index1D pixelIndex, ArrayView<short> input,
        ArrayView<short> output, int width, int height, int stride)
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

                    int idx = ny * stride + nx * 3 + channel;
                    sum += input[idx] * kernelVal;
                }
            }

            int outIdx = y * stride + x * 3 + channel;
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[outIdx] = (short)sum;
        }
    }

    private static void ConvolutionKernelInt32(Index1D pixelIndex, ArrayView<int> input,
        ArrayView<int> output, int width, int height, int stride)
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

                    int idx = ny * stride + nx * 3 + channel;
                    sum += input[idx] * kernelVal;
                }
            }

            int outIdx = y * stride + x * 3 + channel;
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[outIdx] = sum;
        }
    }

    private static void ConvolutionKernelFloat(Index1D pixelIndex, ArrayView<float> input,
        ArrayView<float> output, int width, int height, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        for (int channel = 0; channel < 3; channel++)
        {
            float sum = 0f;

            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int ny = IntrinsicMath.Clamp(y + ky, 0, height - 1);
                    int nx = IntrinsicMath.Clamp(x + kx, 0, width - 1);
                    int kernelVal = (ky == 0 && kx == 0) ? 9 : -1;

                    int idx = ny * stride + nx * 3 + channel;
                    sum += input[idx] * kernelVal;
                }
            }

            int outIdx = y * stride + x * 3 + channel;
            if (sum < 0f) sum = 0f;
            if (sum > 255f) sum = 255f;
            output[outIdx] = sum;
        }
    }

    private static void ConvolutionKernelUShort(Index1D pixelIndex, ArrayView<ushort> input,
        ArrayView<ushort> output, int width, int height, int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        for (int channel = 0; channel < 3; channel++)
        {
            float sum = 0f;

            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int ny = IntrinsicMath.Clamp(y + ky, 0, height - 1);
                    int nx = IntrinsicMath.Clamp(x + kx, 0, width - 1);
                    int kernelVal = (ky == 0 && kx == 0) ? 9 : -1;

                    int idx = ny * stride + nx * 3 + channel;
                    sum += (float)input[idx] * kernelVal;
                }
            }

            int outIdx = y * stride + x * 3 + channel;
            if (sum < 0f) sum = 0f;
            if (sum > 255f) sum = 255f;
            output[outIdx] = (ushort)sum;
        }
    }
    #endregion
}
