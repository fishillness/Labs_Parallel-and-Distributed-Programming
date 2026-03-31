using Microsoft.VisualBasic.FileIO;
using ScottPlot;
using System.Diagnostics;
using System.Globalization;

internal class Program
{
    private static void Main(string[] args)
    {
        string tablePath = "..\\..\\..\\table\\";
        string schedulePath = "..\\..\\..\\schedule\\";
        string tableName = "BD-Patients.csv";
        int[] clusterCounts = { 3, 4, 5 };
        int repeats = 3;
        int[] threadCounts = { 2, 4, 6, 8, 10, 12, 14, 16 };
        List<int> sizes = new List<int> { 1000, 3000, 5000 };

        List<(double x, double y)> data = LoadData(tablePath, tableName);
        data = NormalizeData(data);
        sizes.Add(data.Count);

        foreach (int k in clusterCounts)
        {
            Console.WriteLine($" ----  K = {k}  ----");

            var (besrClasterForElement, centroids) = KMeans(data, k);

            for (int i = 0; i < centroids.Length; i++)
            {
                Console.WriteLine($"    Центроид {i}: ({centroids[i].x:F3}, {centroids[i].y:F3})");
            }
            
            DrawClusters(data, besrClasterForElement, k, $"{schedulePath}_clusters_k{k}.png");

            foreach (int size in sizes)
            {
                Console.WriteLine($" ----  Кол-во точек = {size}  ----");

                foreach (int threadCount in threadCounts)
                {
                    List<double> times = new List<double>();
                    List<double> csIndexes = new List<double>();

                    for (int i = 0; i < repeats; i++)
                    {
                        Stopwatch sw = Stopwatch.StartNew();

                        double csIndex = CalculateCSIndex(data.Take(size).ToList(), besrClasterForElement.Take(size).ToArray(), k, threadCount);

                        sw.Stop();
                        double time = sw.Elapsed.TotalSeconds;
                        times.Add(time);
                        csIndexes.Add(csIndex);
                    }


                    Console.WriteLine($"Потоков {threadCount}");
                    Console.WriteLine($"    Среднее время  выполнения: {Math.Round(times.Average(), 2)}");
                    Console.WriteLine($"    Средний CsIndex: {Math.Round(csIndexes.Average(), 2)}");
                    Console.WriteLine();
                }
            }
            Console.WriteLine();
        }
    }


    private static List<(double x, double y)> LoadData(string path, string tableName)
    {
        List<(double x, double y)> rows = new List<(double x, double y)>();

        TextFieldParser parser = new TextFieldParser(path + tableName);
        parser.TextFieldType = FieldType.Delimited;
        parser.SetDelimiters(",");

        string[] header = parser.ReadFields();
        int indexTemp_pvariance = Array.IndexOf(header, "Temp_pvariance");
        int indexHCO3_pvariance = Array.IndexOf(header, "HCO3_pvariance");

        while (!parser.EndOfData)
        {
            string[] fields = parser.ReadFields();

            if (double.TryParse(fields[indexTemp_pvariance], CultureInfo.InvariantCulture, out double t)
                && double.TryParse(fields[indexHCO3_pvariance], CultureInfo.InvariantCulture, out double h))
            {
                rows.Add((t, h)); //rows.Add(new double[] { t, h }); 
            }
        }

        parser.Dispose();
        return rows;
    }

    private static List<(double x, double y)> NormalizeData(List<(double x, double y)> data)
    {
        List<(double x, double y)> normalized = new List<(double x, double y)>();

        double maxTemp_pvariance = data.Max(row => row.x);
        double minTemp_pvariance = data.Min(row => row.x);

        double maxHCO3_pvariance = data.Max(row => row.y);
        double minHCO3_pvariance = data.Min(row => row.y);

        foreach ((double x, double y) row in data)
        {
            double normTemp_pvariance = (row.x - minTemp_pvariance) / (maxTemp_pvariance - minTemp_pvariance);
            double normHCO3_pvariance = (row.y - minHCO3_pvariance) / (maxHCO3_pvariance - minHCO3_pvariance);

            normalized.Add((normTemp_pvariance, normHCO3_pvariance)); 
        }
        return normalized;
    }

    private static (int[], (double x, double y)[]) KMeans(List<(double x, double y)> data, int k, int maxIter = 100)
    {
        Random rnd = new Random();
        int n = data.Count;
        int[] besrClasterForElement = new int[n];
        (double x, double y)[] centroids = new (double, double)[k];

        for (int i = 0; i < k; i++)
        {
            (double x, double y) centr = data[rnd.Next(n)];
            centroids[i] = (centr.x, centr.y);
        }

        for (int iter = 0; iter < maxIter; iter++)
        {
            bool changed = false;

            for (int i = 0; i < n; i++)
            {
                int bestCluster = 0;
                double minDist = double.MaxValue;

                for (int c = 0; c < k; c++)
                {
                    double dx = data[i].x - centroids[c].x;
                    double dy = data[i].y - centroids[c].y;
                    double dist = Math.Sqrt(dx * dx + dy * dy);

                    if (dist < minDist)
                    {
                        minDist = dist;
                        bestCluster = c;
                    }
                }

                if (besrClasterForElement[i] != bestCluster)
                {
                    besrClasterForElement[i] = bestCluster;
                    changed = true;
                }
            }

            if (!changed) break;

            double[] sumX = new double[k];
            double[] sumY = new double[k];
            int[] count = new int[k];

            for (int i = 0; i < n; i++)
            {
                sumX[besrClasterForElement[i]] += data[i].x;
                sumY[besrClasterForElement[i]] += data[i].y;
                count[besrClasterForElement[i]]++;
            }

            for (int c = 0; c < k; c++)
            {
                if (count[c] > 0)
                    centroids[c] = (sumX[c] / count[c], sumY[c] / count[c]);
            }
        }

        return (besrClasterForElement, centroids);
    }

    private static void DrawClusters(List<(double x, double y)> data, int[] besrClasterForElement, int k, string fileName)
    {
        Plot plt = new Plot();

        ScottPlot.Color[] colors =
        {
            Colors.Red,
            Colors.Blue,
            Colors.Green,
            Colors.Orange,
            Colors.Purple
        };

        for (int i = 0; i < data.Count; i++)
        {
            int cluster = besrClasterForElement[i];
            ScottPlot.Color color = colors[cluster % colors.Length];

            plt.Add.Marker(
                x: data[i].x,
                y: data[i].y,
                size: 8,
                color: color
            );
        }

        plt.Title($"K-Means Класстеризация (K={k})");
        plt.XLabel("Temp_pvariance (Нормализованные)");
        plt.YLabel("HCO3_pvariance (Нормализованные)");

        plt.SavePng(fileName, 800, 600);
    }

    private static double CalculateCSIndex(List<(double x, double y)> data, int[] besrClasterForElement, int k, int threadCount)
    {
        int n = data.Count;
        double[] csIndex = new double[n];

        Dictionary<int, List<int>> clusters = new Dictionary<int, List<int>>();

        for (int i = 0; i < k; i++)
            clusters[i] = new List<int>();

        for (int i = 0; i < n; i++)
            clusters[besrClasterForElement[i]].Add(i);

        Thread[] threads = new Thread[threadCount]; 
        int chunk = n / threadCount;

        for (int t = 0; t < threadCount; t++)
        {
            int start = t * chunk;
            int end = (t == threadCount - 1) ? n : start + chunk;

            threads[t] = new Thread(() =>
            {
                for (int i = start; i < end; i++)
                {
                    int clusterA = besrClasterForElement[i];

                    double a = 0;
                    int countA = clusters[clusterA].Count;

                    if (countA > 1)
                    {
                        foreach (int j in clusters[clusterA])
                        {
                            if (j == i) continue;
                            double dx = data[i].x - data[j].x;
                            double dy = data[i].y - data[j].y;
                            a += Math.Sqrt(dx * dx + dy * dy);
                        }
                        a /= (countA - 1);
                    }

                    double b = double.MaxValue;

                    foreach (var cluster in clusters)
                    {
                        int clusterB = cluster.Key;
                        if (clusterB == clusterA || cluster.Value.Count == 0) continue;

                        double dist = 0;
                        foreach (int j in cluster.Value)
                        {
                            double dx = data[i].x - data[j].x;
                            double dy = data[i].y - data[j].y;
                            dist += Math.Sqrt(dx * dx + dy * dy);
                        }
                        dist /= cluster.Value.Count;

                        if (dist < b)
                            b = dist;
                    }

                    double cs = 0;
                    if (a != 0 || b != 0)
                        cs = (b - a) / Math.Max(a, b);

                    csIndex[i] = cs;
                }
            });

            threads[t].Start();
        }

        foreach (Thread thread in threads)
            thread.Join();

        return csIndex.Average();
    }
}

