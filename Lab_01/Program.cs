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

        foreach (string imgName in images)
        {
            List<double> times = new List<double>();
            Console.WriteLine($"Изображение {imgName}:");

            for (int i = 0; i < repeats; i++)
            {
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();

                ProcessImage(path, imgName);

                double time = stopwatch.Elapsed.TotalSeconds;
                stopwatch.Stop();

                times.Add(time);
                Console.WriteLine($"  [{i}]: {Math.Round(time, 2)}");
            }

            Console.WriteLine($"Для изображения {imgName} среднее время равно {Math.Round(times.Average(), 2)}");
        }
    }

    private static void ProcessImage(string path, string imgName)
    {
        Bitmap img = new Bitmap(path + imgName);
        Bitmap inverted = InvertTestLockbit(img); //Bitmap result = InvertTestLockbit(img); //Invert(img);
        Bitmap contrasted = IncreaseContrastTestLockbit(inverted); //result = IncreaseContrastTestLockbit(img); //IncreaseContrast(result);

        contrasted.Save(path + "new_" + imgName, ImageFormat.Jpeg);

        img.Dispose();
        inverted.Dispose();
        contrasted.Dispose();
    }

    private static Bitmap Invert(Bitmap img)
    {
        Bitmap result = new Bitmap(img.Width, img.Height);

        for (int x = 0; x < img.Width; x++)
        {
            for (int y = 0; y < img.Height; y++)
            {
                Color pixelColor = img.GetPixel(x, y);
                Color invertedColor = Color.FromArgb(255 - pixelColor.R, 255 - pixelColor.G, 255 - pixelColor.B);
                result.SetPixel(x, y, invertedColor);
            }
        }

        return result;
    }

    private static Bitmap IncreaseContrast(Bitmap img)
    {
        int[,] convolutionMatrix =
        {
            {  0, -1, 0 },
            { -1, 5, -1 },
            {  0, -1, 0 }
        };

        Bitmap result = new Bitmap(img.Width, img.Height);
        int matrixSize = convolutionMatrix.GetLength(0);
        int offset = matrixSize / 2;

        for (int y = offset; y < img.Height - offset; y++)
        {
            for (int x = offset; x < img.Width - offset; x++)
            {
                int r = 0, g = 0, b = 0;

                for (int ky = -offset; ky <= offset; ky++)
                {
                    for (int kx = -offset; kx <= offset; kx++)
                    {
                        Color pixel = img.GetPixel(x + kx, y + ky);
                        int value = convolutionMatrix[ky + offset, kx + offset];
                        r += pixel.R * value;
                        g += pixel.G * value;
                        b += pixel.B * value;
                    }
                }

                r = Math.Clamp(r, 0, 255);
                g = Math.Clamp(g, 0, 255);
                b = Math.Clamp(b, 0, 255);
                result.SetPixel(x, y, Color.FromArgb(r, g, b));
            }
        }

        return result;
    }

    private static Bitmap InvertTestLockbit(Bitmap img)
    {
        int colorCount = 3;
        Bitmap result = new Bitmap(img.Width, img.Height, PixelFormat.Format24bppRgb);
        BitmapData imgBitmapData = img.LockBits(new Rectangle(0, 0, img.Width, img.Height), 
                                            ImageLockMode.ReadOnly, 
                                            PixelFormat.Format24bppRgb);
        BitmapData resultBitmapData = result.LockBits(new Rectangle(0, 0, img.Width, img.Height), 
                                            ImageLockMode.WriteOnly, 
                                            PixelFormat.Format24bppRgb);

        int stride = Math.Abs(imgBitmapData.Stride);
        int bytes = stride * img.Height;

        byte[] imgByte = new byte[bytes];
        byte[] resultByte = new byte[bytes];

        Marshal.Copy(imgBitmapData.Scan0, imgByte, 0, bytes);

        for (int y = 0; y < img.Height; y++)
        {
            int rowStart = y * stride;

            for (int x = 0; x < img.Width; x++)
            {
                int idx = rowStart + x * colorCount;

                byte b = imgByte[idx];
                byte g = imgByte[idx + 1];
                byte r = imgByte[idx + 2];

                resultByte[idx] = (byte)(255 - b);
                resultByte[idx + 1] = (byte)(255 - g);
                resultByte[idx + 2] = (byte)(255 - r);
            }
        }

        Marshal.Copy(resultByte, 0, resultBitmapData.Scan0, bytes);

        img.UnlockBits(imgBitmapData);
        result.UnlockBits(resultBitmapData);

        return result;
    }
    private static Bitmap IncreaseContrastTestLockbit(Bitmap img)
    {
        int[,] convolutionMatrix =
        {
            {  0, -1, 0 },
            { -1, 5, -1 },
            {  0, -1, 0 }
        };

        int colorCount = 3;
        int kSize = 3;
        int offset = kSize / 2;

        Bitmap result = new Bitmap(img.Width, img.Height, PixelFormat.Format24bppRgb);
        BitmapData imgBitmapData = img.LockBits(new Rectangle(0, 0, img.Width, img.Height), 
                                            ImageLockMode.ReadOnly, 
                                            PixelFormat.Format24bppRgb);
        BitmapData resultBitmapData = result.LockBits(new Rectangle(0, 0, img.Width, img.Height), 
                                            ImageLockMode.WriteOnly, 
                                            PixelFormat.Format24bppRgb);

        int stride = imgBitmapData.Stride;
        int bytes = stride * img.Height;

        byte[] imgByte = new byte[bytes];
        byte[] resultByte = new byte[bytes];

        Marshal.Copy(imgBitmapData.Scan0, imgByte, 0, bytes);

        for (int y = offset; y < img.Height - offset; y++)
        {
            for (int x = offset; x < img.Width - offset; x++)
            {
                int r = 0, g = 0, b = 0;

                for (int ky = -offset; ky <= offset; ky++)
                {
                    for (int kx = -offset; kx <= offset; kx++)
                    {
                        int px = x + kx;
                        int py = y + ky;

                        int idx = py * stride + px * colorCount;
                        int value = convolutionMatrix[ky + offset, kx + offset];

                        b += imgByte[idx] * value;
                        g += imgByte[idx + 1] * value;
                        r += imgByte[idx + 2] * value;
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

        Marshal.Copy(resultByte, 0, resultBitmapData.Scan0, bytes);

        img.UnlockBits(imgBitmapData);
        result.UnlockBits(resultBitmapData);

        return result;
    }

}