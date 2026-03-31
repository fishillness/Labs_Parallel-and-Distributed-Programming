using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
internal class Program
{
    private static void Main(string[] args)
    {
        string path = "..\\..\\..\\img\\";
        string[] images = { "img1.jpg", "img2.jpg", "img3.jpg" };
        int repeats = 3;
        int[] threadCounts = { 2, 4, 6, 8, 10, 12, 14, 16 };

        foreach (string imgName in images)
        {
            Console.WriteLine($"Изображение {imgName}:");

            foreach (int threads in threadCounts)
            {
                List<double> times = new List<double>();

                for (int i = 0; i < repeats; i++)
                {
                    Stopwatch sw = Stopwatch.StartNew();

                    ProcessImage(path, imgName, threads);

                    sw.Stop();
                    double time = sw.Elapsed.TotalSeconds;
                    times.Add(time);

                    Console.WriteLine($"        Потоков {threads}, запуск [{i}]: {Math.Round(time, 2)} сек");
                }

                Console.WriteLine($"Для изображения {imgName} среднее время при ({threads} потоков): {Math.Round(times.Average(), 2)} сек");
                Console.WriteLine();
            }
        }
    }

    private static void ProcessImage(string path, string imgName, int threads)
    {
        Bitmap img = new Bitmap(path + imgName);
        Bitmap inverted = Invert(img, threads);
        Bitmap contrasted = IncreaseContrast(inverted, threads);

        contrasted.Save(path + "new_" + imgName, ImageFormat.Jpeg);

        img.Dispose();
        inverted.Dispose();
        contrasted.Dispose();
    }

    private static Bitmap Invert(Bitmap img, int threadCount)
    {
        int width = img.Width;
        int height = img.Height;
        int colorCount = 3;

        Bitmap result = new Bitmap(width, height, PixelFormat.Format24bppRgb);
        BitmapData imgBitmapData = img.LockBits(new Rectangle(0, 0, width, height),
                                            ImageLockMode.ReadOnly,
                                            PixelFormat.Format24bppRgb);
        BitmapData resultBitmapData = result.LockBits(new Rectangle(0, 0, width, height),
                                            ImageLockMode.WriteOnly,
                                            PixelFormat.Format24bppRgb);

        int stride = Math.Abs(imgBitmapData.Stride);
        int bytes = stride * height;

        byte[] imgByte = new byte[bytes];
        byte[] resultByte = new byte[bytes];

        Marshal.Copy(imgBitmapData.Scan0, imgByte, 0, bytes);

        Thread[] threads = new Thread[threadCount];
        int chunk = height / threadCount;

        for (int t = 0; t < threadCount; t++)
        {
            int startY = t * chunk;
            int endY = (t == threadCount - 1) ? height : startY + chunk;

            threads[t] = new Thread(() =>
            {
                for (int y = startY; y < endY; y++)
                {
                    int rowStart = y * stride;

                    for (int x = 0; x < width; x++)
                    {
                        int index = rowStart + x * colorCount;

                        resultByte[index] = (byte)(255 - imgByte[index]);
                        resultByte[index + 1] = (byte)(255 - imgByte[index + 1]);
                        resultByte[index + 2] = (byte)(255 - imgByte[index + 2]);
                    }
                }
            });

            threads[t].Start();
        }

        foreach (Thread thread in threads) thread.Join();

        Marshal.Copy(resultByte, 0, resultBitmapData.Scan0, bytes);

        img.UnlockBits(imgBitmapData);
        result.UnlockBits(resultBitmapData);

        return result;
    }

    private static Bitmap IncreaseContrast(Bitmap img, int threadCount)
    {
        int[,] convolutionMatrix =
        {
            { 0, -1, 0 },
            { -1, 5, -1 },
            { 0, -1, 0 }
        };

        int width = img.Width;
        int height = img.Height;
        int colorCount = 3;
        int kSize = 3;
        int offset = kSize / 2;

        Bitmap result = new Bitmap(width, height, PixelFormat.Format24bppRgb);
        BitmapData imgBitmapData = img.LockBits(new Rectangle(0, 0, width, height),
                                            ImageLockMode.ReadOnly,
                                            PixelFormat.Format24bppRgb);
        BitmapData resultBitmapData = result.LockBits(new Rectangle(0, 0, width, height),
                                            ImageLockMode.WriteOnly,
                                            PixelFormat.Format24bppRgb);

        int stride = Math.Abs(imgBitmapData.Stride);
        int bytes = stride * height;

        byte[] imgByte = new byte[bytes];
        byte[] resultByte = new byte[bytes];

        Marshal.Copy(imgBitmapData.Scan0, imgByte, 0, bytes);

        Thread[] threads = new Thread[threadCount];
        int chunk = height / threadCount;

        for (int t = 0; t < threadCount; t++)
        {
            int startY = Math.Max(offset, t * chunk);
            int endY = (t == threadCount - 1) ? height - offset : Math.Min(height - offset, startY + chunk);

            threads[t] = new Thread(() =>
            {
                for (int y = startY; y < endY; y++)
                {
                    for (int x = offset; x < width - offset; x++)
                    {
                        int r = 0, g = 0, b = 0;

                        for (int ky = -offset; ky <= offset; ky++)
                        {
                            for (int kx = -offset; kx <= offset; kx++)
                            {
                                int px = x + kx;
                                int py = y + ky;

                                int index = py * stride + px * colorCount;
                                int kVal = convolutionMatrix[ky + offset, kx + offset];

                                b += imgByte[index] * kVal;
                                g += imgByte[index + 1] * kVal;
                                r += imgByte[index + 2] * kVal;
                            }
                        }

                        r = Math.Clamp(r, 0, 255);
                        g = Math.Clamp(g, 0, 255);
                        b = Math.Clamp(b, 0, 255);

                        int outIdx = y * stride + x * colorCount;

                        resultByte[outIdx] = (byte)b;
                        resultByte[outIdx + 1] = (byte)g;
                        resultByte[outIdx + 2] = (byte)r;
                    }
                }
            });

            threads[t].Start();
        }

        foreach (Thread thread in threads) thread.Join();

        Marshal.Copy(resultByte, 0, resultBitmapData.Scan0, bytes);

        img.UnlockBits(imgBitmapData);
        result.UnlockBits(resultBitmapData);

        return result;
    }
}