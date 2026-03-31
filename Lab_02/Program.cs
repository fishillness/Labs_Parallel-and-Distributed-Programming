using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

internal class Program
{
    private static void Main(string[] args)
    {
        int n = 1000000000;
        int[] data = new int[n];

        int[] threadCounts = { 2, 4, 6, 8, 10, 12, 14, 16 };

        Console.WriteLine("Тест многопоточности (Thread) для заполнения массива");
        Console.WriteLine($"Размер массива: {n}");
        Console.WriteLine("Формула: data[i] = (i % 100)^3\n");

        foreach (int threads in threadCounts)
        {
            Stopwatch sw = Stopwatch.StartNew();

            Thread[] workers = new Thread[threads];

            int chunk = n / threads;

            for (int t = 0; t < threads; t++)
            {
                int start = t * chunk;
                int end = (t == threads - 1) ? n : start + chunk;

                workers[t] = new Thread(() =>
                {
                    for (int i = start; i < end; i++)
                    {
                        int x = i % 100;
                        data[i] = x * x * x;
                    }
                });

                workers[t].Start();
            }

            for (int t = 0; t < threads; t++)
                workers[t].Join();

            sw.Stop();

            Console.WriteLine($"Потоков: {threads,2} | Время: {sw.Elapsed.TotalSeconds:F3} с");
        }

        Console.WriteLine("\nГотово.");
    }
}