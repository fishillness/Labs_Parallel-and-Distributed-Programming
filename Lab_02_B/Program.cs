using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Drawing.Imaging;

internal class Program
{

    private static void Main(string[] args)
    {
        string path = "..\\..\\..\\img\\";
        string[] images = { "img1.jpg", "img2.jpg", "img3.jpg" };
        int repeats = 3;
        int[] threadCounts = { 2 , 4, 6, 8, 10, 12, 14, 16 };
        int threshold = 100;

        Console.WriteLine($"threshold = {threshold}");
        Console.WriteLine();

        foreach (string imgName in images)
        {
            Console.WriteLine($"Изображение {imgName}:");

            foreach (int threads in threadCounts)
            {
                List<double> times = new List<double>();

                for (int i = 0; i < repeats; i++)
                {
                    Stopwatch sw = Stopwatch.StartNew();

                    ProcessImage(path, imgName, threads, threshold);

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

    private static void ProcessImage(string path, string imgName, int threads, int threshold)
    {
        Bitmap img = new Bitmap(path + imgName);
        Bitmap intensity = ChangeIntensity(img, threads, threshold);
        intensity.Save(path + "new_intensity_" + imgName, ImageFormat.Jpeg);

        Bitmap dilated = Dilatation(intensity, threads);
        dilated.Save(path + "new_dilated_" + imgName, ImageFormat.Jpeg);

        img.Dispose();
        intensity.Dispose();
        dilated.Dispose();
    }

    private static Bitmap ChangeIntensity(Bitmap img, int threadCount, int threshold)
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
                        int idx = rowStart + x * colorCount;
                        float sum = (imgByte[idx] + imgByte[idx + 1] + imgByte[idx + 2]) / 3;

                        if (sum > threshold)
                        {
                            resultByte[idx] = 255;
                            resultByte[idx + 1] = 255;
                            resultByte[idx + 2] = 255;
                        }
                        else
                        {
                            resultByte[idx] = 0;
                            resultByte[idx + 1] = 0;
                            resultByte[idx + 2] = 0;
                        }
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

    private static Bitmap Dilatation(Bitmap img, int threadCount)
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
                        int idx = rowStart + x * colorCount;
                        
                        if (imgByte[idx] == 255 && imgByte[idx + 1] == 255 && imgByte[idx + 2] == 255)
                        {
                            for (int dy = -1; dy <= 1; dy++)
                            {
                                int ny = y + dy;
                                if (ny < 0 || ny >= height) continue;

                                for (int dx = -1; dx <= 1; dx++)
                                {
                                    int nx = x + dx;
                                    if (nx < 0 || nx >= width) continue;

                                    int nIdx = ny * stride + nx * 3;

                                    resultByte[nIdx] = 255;
                                    resultByte[nIdx + 1] = 255;
                                    resultByte[nIdx + 2] = 255;
                                }
                            }

                        }
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
