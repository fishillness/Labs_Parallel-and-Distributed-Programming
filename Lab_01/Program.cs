using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

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
        Bitmap result = Invert(img);
        result = IncreaseContrast(result);

        result.Save(path + "new_" + imgName, ImageFormat.Jpeg);

        img.Dispose();
        result.Dispose();
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
}