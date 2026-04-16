using System.Diagnostics;
using System.Text.RegularExpressions;

class Program
{
    private static readonly Dictionary<string, string> Urls = new Dictionary<string, string>
    {
        { "Пушкин", "https://rustih.ru/aleksandr-pushkin/" },
        { "Лермонтов", "https://rustih.ru/mixail-lermontov/" },
        { "Бальмонт", "https://rustih.ru/konstantin-balmont/" }
    };

    private static readonly string path = "..\\..\\..\\text\\";
    private static readonly Dictionary<string, string> files = new Dictionary<string, string>
    {
        { "Пушкин", "Пушкин.txt" },
        { "Лермонтов", "Лермонтов.txt" },
        { "Бальмонт", "Бальмонт.txt" }
    };

    static async Task Main(string[] args)
    {
        int[] threadCounts = { 2, 4, 6, 8 };

        foreach (int threadCount in threadCounts)
        {
            Console.WriteLine($"\n-------------------------------------");
            Console.WriteLine($"Потоков: {threadCount}");

            Stopwatch sw = Stopwatch.StartNew();
            await RunAnalysis(threadCount);
            sw.Stop();

            Console.WriteLine($"\nВремя выполнения с {threadCount} потоками: {sw.ElapsedMilliseconds} мс");
        }
    }

    static async Task RunAnalysis(int threadCount)
    {
        List<Task> tasks = new List<Task>();
        foreach (var url in Urls)
        {
            tasks.Add(ProcessAuthor(url.Key, url.Value, threadCount));
        }
        await Task.WhenAll(tasks);
    }

    static async Task ProcessAuthor(string author, string url, int threadCount)
    {
        Console.WriteLine($"[{author}] Начинаем анализ...");

        using HttpClient client = new HttpClient();

        string catalogHtml = await client.GetStringAsync(url);
        var poemLinks = Regex.Matches(catalogHtml, @"<a\s+class=""poem-card__link""\s+href=""([^""]+)""")
                             .Select(m => m.Groups[1].Value)
                             .Select(link => link.StartsWith("http") ? link : "https://rustih.ru" + link)
                             .Distinct()
                             .ToList();

        Console.WriteLine($"[{author}] Найдено стихов: {poemLinks.Count}");

        List<string> allWords = new List<string>();

        foreach (string poemUrl in poemLinks)
        {
            string poemHtml = await client.GetStringAsync(poemUrl);

            var contentMatch = Regex.Match(poemHtml, @"<section\s+class=""entry-content poem-text""[^>]*>(.*?)</section>", RegexOptions.Singleline);
            if (!contentMatch.Success) continue;

            var paragraphs = Regex.Matches(contentMatch.Groups[1].Value, @"<p>(.*?)</p>", RegexOptions.Singleline)
                                  .Take(5)
                                  .Select(m => Regex.Replace(m.Groups[1].Value, @"<[^>]+>", ""))
                                  .Select(m => System.Net.WebUtility.HtmlDecode(m))
                                  .Where(p => p.Length > 30 && !p.Contains("window."))
                                  .ToList();

            foreach (var p in paragraphs)
            {
                string clean = Regex.Replace(p.ToLower(), @"[^\p{L}\s]", "");
                allWords.AddRange(clean.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries));
            }
        }

        Console.WriteLine($"[{author}] Собрано слов: {allWords.Count}");

        var chunks = SplitList(allWords, threadCount);
        var chunkResults = await Task.WhenAll(chunks.Select(chunk => Task.Run(() => CountWords(chunk))));

        var finalFreq = new Dictionary<string, int>();
        foreach (var dict in chunkResults)
            foreach (var kv in dict)
                finalFreq[kv.Key] = finalFreq.GetValueOrDefault(kv.Key) + kv.Value;

        var topWords = finalFreq.OrderByDescending(x => x.Value).Take(10).ToList();

        Directory.CreateDirectory(path);
        string filePath = Path.Combine(path, files[author]);

        using (StreamWriter writer = new StreamWriter(filePath, true))
        {
            writer.WriteLine($"\n-------------------------------------");
            writer.WriteLine($"Потоков: {threadCount}");
            writer.WriteLine("Наиболее встречающиеся слова:");
            foreach (var w in topWords)
                writer.WriteLine($"  \"{w.Key}\" : {w.Value}");
        }

        Console.WriteLine($"[{author}] Готово!");
    }

    static Dictionary<string, int> CountWords(List<string> words)
    {
        var freq = new Dictionary<string, int>();
        foreach (var w in words)
            freq[w] = freq.GetValueOrDefault(w) + 1;
        return freq;
    }

    static List<List<T>> SplitList<T>(List<T> source, int chunkCount)
    {
        int size = (int)Math.Ceiling((double)source.Count / chunkCount);
        return Enumerable.Range(0, (source.Count + size - 1) / size)
                         .Select(i => source.Skip(i * size).Take(size).ToList())
                         .ToList();
    }
}

/*
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

class Program
{
    private static readonly Dictionary<string, string> Urls = new Dictionary<string, string>
        {
            { "Пушкин", "https://rustih.ru/aleksandr-pushkin/" },
            { "Лермонтов", "https://rustih.ru/mixail-lermontov/" },
            { "Бальмонт", "https://rustih.ru/konstantin-balmont/" }
        };

    private static readonly string path = "..\\..\\..\\text\\";
    private static readonly Dictionary<string, string> files = new Dictionary<string, string>
        {
            { "Пушкин", "Пушкин.txt" },
            { "Лермонтов", "Лермонтов.txt" },
            { "Бальмонт", "Бальмонт.txt" }
        };

    static async Task Main(string[] args)
    {
        int[] threadCounts = { 2, 4, 6, 8 };

        foreach (int threadCount in threadCounts)
        {
            Console.WriteLine($"\n-------------------------------------");
            Console.WriteLine($"Потоков: {threadCount}");

            Stopwatch sw = Stopwatch.StartNew();
            await RunAnalysis(threadCount);
            sw.Stop();

            long time = sw.ElapsedMilliseconds;
            Console.WriteLine($"\nВремя выполнения с {threadCount} потоками: {time} мс");
        }
    }

    static async Task RunAnalysis(int threadCount)
    {
        List<Task> tasks = new List<Task>();

        foreach (var url in Urls)
        {
            tasks.Add(ProcessAuthor(url.Key, url.Value, threadCount));
        }

        await Task.WhenAll(tasks);
    }

    static async Task ProcessAuthor(string author, string url, int threadCount)
    {
        Console.WriteLine($"[{author}] Начинаем анализ...");

        HttpClient client = new HttpClient();

        // Загружаем страницу
        string html = await client.GetStringAsync(url);
        // Извлекаем абзацы (теги <p>)
        var paragraphs = ExtractParagraphs(html);
        // Берем первые 5 четверостиший (примерно 5-10 абзацев)
        var limitedParagraphs = paragraphs.Take(8).ToList();

        // Собираем все слова
        var allWords = new List<string>();
        foreach (var paragraph in limitedParagraphs)
        {
            string cleanText = CleanText(paragraph);
            var words = cleanText.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            allWords.AddRange(words);
        }

        // Разбиваем слова на части для параллельной обработки
        var wordChunks = SplitList(allWords, threadCount);
        List<Task<Dictionary<string, int>>> tasks = new List<Task<Dictionary<string, int>>>();

        for (int i = 0; i < threadCount && i < wordChunks.Count; i++)
        {
            int chunkIndex = i;
            var chunk = wordChunks[chunkIndex];
            tasks.Add(Task.Run(() => CountWordsInChunk(chunk)));
        }

        // Ждем завершения всех потоков
        var chunkResults = await Task.WhenAll(tasks);

        // Объединяем результаты
        Dictionary<string, int> finalFrequency = new Dictionary<string, int>();
        foreach (var chunkResult in chunkResults)
        {
            foreach (var kvp in chunkResult)
            {
                if (finalFrequency.ContainsKey(kvp.Key))
                    finalFrequency[kvp.Key] += kvp.Value;
                else
                    finalFrequency[kvp.Key] = kvp.Value;
            }
        }

        // Берем топ 5-10 слов (возьмем 10)
        var topWords = finalFrequency
            .OrderByDescending(x => x.Value)
            .Take(10)
            .ToList();

        string filePath = Path.Combine(path, files[author]);

        StreamWriter writer = new StreamWriter(filePath, true);
        writer.WriteLine($"\n-------------------------------------");
        writer.WriteLine($"Потоков: {threadCount}");
        writer.WriteLine("Наиболее встречающиеся слова:");
        for (int i = 0; i < topWords.Count; i++)
        {
            writer.WriteLine($"  \"{topWords[i].Key}\" : {topWords[i].Value}");
        }

        writer.Close();
        client.Dispose();
    }

    static Dictionary<string, int> CountWordsInChunk(List<string> words)
    {
        var frequency = new Dictionary<string, int>();

        foreach (var word in words)
        {
            if (string.IsNullOrEmpty(word))
                continue;

            if (frequency.ContainsKey(word))
                frequency[word]++;
            else
                frequency[word] = 1;
        }

        return frequency;
    }

    static List<string> ExtractParagraphs(string html)
    {
        var paragraphs = new List<string>();
        var pattern = @"<p>(.*?)</p>";
        var matches = Regex.Matches(html, pattern, RegexOptions.Singleline | RegexOptions.IgnoreCase);

        foreach (Match match in matches)
        {
            string content = match.Groups[1].Value;
            // Удаляем HTML-теги внутри
            content = Regex.Replace(content, @"<[^>]+>", "");
            content = System.Net.WebUtility.HtmlDecode(content);
            if (!string.IsNullOrWhiteSpace(content))
            {
                paragraphs.Add(content);
            }
        }

        return paragraphs;
    }

    static string CleanText(string text)
    {
        // Приводим к нижнему регистру
        text = text.ToLower();

        // Удаляем знаки препинания, оставляем только буквы и пробелы
        text = Regex.Replace(text, @"[^\p{L}\s]", "");

        // Удаляем цифры
        text = Regex.Replace(text, @"\d+", "");

        return text;
    }

    static List<List<T>> SplitList<T>(List<T> source, int chunkCount)
    {
        var result = new List<List<T>>();
        if (source.Count == 0)
        {
            return result;
        }

        int chunkSize = (int)Math.Ceiling((double)source.Count / chunkCount);

        for (int i = 0; i < source.Count; i += chunkSize)
        {
            result.Add(source.Skip(i).Take(chunkSize).ToList());
        }

        return result;
    }
}
*/

/*
using System.Diagnostics;
using System.Text.RegularExpressions;

internal class Program
{
    static readonly HttpClient http = new HttpClient();

    private static void Main(string[] args)
    {
        string[] urls = {
            "https://rustih.ru/aleksandr-pushkin/",
            "https://rustih.ru/mixail-lermontov/",
            "https://rustih.ru/konstantin-balmont/"
        };

        int[] threadCounts = { 2, 4, 6, 8 };

        foreach (int threads in threadCounts)
        {
            Console.WriteLine($"\n=== Запуск с {threads} потоками ===");

            List<Process> processes = new List<Process>();

            foreach (string url in urls)
            {
                Process p = new Process();
                p.StartInfo.FileName = Environment.GetCommandLineArgs()[0];
                p.StartInfo.Arguments = $"child \"{url}\" {threads}";
                p.StartInfo.UseShellExecute = false;
                p.StartInfo.RedirectStandardOutput = true;
                p.Start();

                processes.Add(p);
            }

            foreach (var p in processes)
            {
                Console.WriteLine(p.StandardOutput.ReadToEnd());
                p.WaitForExit();
            }
        }
    }


    // -------------------- CHILD PROCESS --------------------
    static void ChildProcess(string url, int threadCount)
    {
        Stopwatch sw = Stopwatch.StartNew();

        Console.WriteLine($"\n--- Анализ: {url} ---");

        string html = http.GetStringAsync(url).Result;

        var links = Regex.Matches(html, "<a href=\"(https://rustih.ru/[^\"]+)\"")
                         .Cast<Match>()
                         .Select(m => m.Groups[1].Value)
                         .Distinct()
                         .Take(threadCount)
                         .ToList();

        Dictionary<string, int> wordCount = new Dictionary<string, int>();
        object locker = new object();

        List<Thread> threads = new List<Thread>();

        foreach (string link in links)
        {
            Thread t = new Thread(() =>
            {
                try
                {
                    string page = http.GetStringAsync(link).Result;

                    var ps = Regex.Matches(page, "<p>(.*?)</p>", RegexOptions.Singleline)
                                  .Cast<Match>()
                                  .Take(5)
                                  .Select(m => StripTags(m.Groups[1].Value));

                    foreach (string text in ps)
                    {
                        foreach (string w in SplitWords(text))
                        {
                            lock (locker)
                            {
                                if (!wordCount.ContainsKey(w))
                                    wordCount[w] = 0;
                                wordCount[w]++;
                            }
                        }
                    }
                }
                catch { }
            });

            threads.Add(t);
            t.Start();
        }

        foreach (var t in threads) t.Join();

        sw.Stop();

        Console.WriteLine("ТОП слов:");
        foreach (var pair in wordCount.OrderByDescending(p => p.Value).Take(10))
            Console.WriteLine($"{pair.Key} : {pair.Value}");

        Console.WriteLine($"Время выполнения: {sw.ElapsedMilliseconds} мс");
    }

    static string StripTags(string s) =>
        Regex.Replace(s, "<.*?>", "").ToLower();

    static IEnumerable<string> SplitWords(string text)
    {
        foreach (var w in Regex.Split(text, @"[^а-яА-Я]+"))
        {
            string word = w.ToLower();
            if (word.Length < 2) continue;
            if (Regex.IsMatch(word, @"^\d+$")) continue;
            yield return word;
        }
    }
}
*/