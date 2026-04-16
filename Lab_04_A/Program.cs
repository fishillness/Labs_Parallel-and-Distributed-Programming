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

        // 1. Получаем HTML каталога
        string catalogHtml = await client.GetStringAsync(url);

        // 2. Находим все ссылки на стихи
        var poemLinks = Regex.Matches(catalogHtml, @"<a\s+class=""poem-card__link""\s+href=""([^""]+)""")
                             .Select(m => m.Groups[1].Value)
                             .Select(link => link.StartsWith("http") ? link : "https://rustih.ru" + link)
                             .Distinct()
                             .ToList();

        Console.WriteLine($"[{author}] Найдено стихов: {poemLinks.Count}");

        // 3. Собираем все слова из стихов
        List<string> allWords = new List<string>();

        foreach (string poemUrl in poemLinks)
        {
            string poemHtml = await client.GetStringAsync(poemUrl);

            // Ищем контейнер со стихами
            var contentMatch = Regex.Match(poemHtml, @"<section\s+class=""entry-content poem-text""[^>]*>(.*?)</section>", RegexOptions.Singleline);
            if (!contentMatch.Success) continue;

            // Берем первые 5 абзацев
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

        // 4. Параллельный подсчёт частоты слов
        var chunks = SplitList(allWords, threadCount);
        var chunkResults = await Task.WhenAll(chunks.Select(chunk => Task.Run(() => CountWords(chunk))));

        // 5. Объединяем результаты
        var finalFreq = new Dictionary<string, int>();
        foreach (var dict in chunkResults)
            foreach (var kv in dict)
                finalFreq[kv.Key] = finalFreq.GetValueOrDefault(kv.Key) + kv.Value;

        // 6. Берём топ-10 и сохраняем
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
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

class Program
{
    static async Task Main(string[] args)
    {
        string authorUrl = "https://rustih.ru/mixail-lermontov/";
        var wordsFrequency = new Dictionary<string, int>(StringComparer.InvariantCultureIgnoreCase);
        var allAnalyzedText = new List<string>();

        using HttpClient client = new HttpClient();
        client.DefaultRequestHeaders.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)");

        Console.WriteLine("=== АНАЛИЗ СТИХОВ МИХАИЛА ЛЕРМОНТОВА ===\n");
        Console.WriteLine("Загрузка каталога: " + authorUrl);

        // 1. Загружаем каталог
        string catalogHtml = await client.GetStringAsync(authorUrl);

        // 2. Ищем ссылки на стихи в формате <a class="poem-card__link" href="...">
        var poemUrls = new List<string>();

        // Ищем все ссылки с классом poem-card__link
        var linkMatches = Regex.Matches(catalogHtml, @"<a\s+class=""poem-card__link""\s+href=""([^""]+)""");

        foreach (Match match in linkMatches)
        {
            string url = match.Groups[1].Value;
            if (!url.StartsWith("http"))
                url = "https://rustih.ru" + url;
            if (!poemUrls.Contains(url))
                poemUrls.Add(url);
        }

        Console.WriteLine($"\nНайдено стихотворений на первой странице: {poemUrls.Count}");

        if (poemUrls.Count == 0)
        {
            Console.WriteLine("Не удалось найти ссылки на стихи. Проверьте структуру страницы.");
            return;
        }

        foreach (var u in poemUrls)
        {
            Console.WriteLine($"  - {u}");
        }

        Console.WriteLine($"\n{new string('=', 80)}");

        int poemNumber = 0;
        foreach (string poemUrl in poemUrls)
        {
            poemNumber++;
            Console.WriteLine($"\n[{poemNumber}] Обработка: {poemUrl}");
            Console.WriteLine(new string('-', 80));

            try
            {
                string poemHtml = await client.GetStringAsync(poemUrl);

                // Извлекаем название стихотворения из тега <h1 class="entry-title">
                var titleMatch = Regex.Match(poemHtml, @"<h1\s+class=""entry-title""[^>]*>([^<]+)</h1>");
                string poemTitle = titleMatch.Success ? titleMatch.Groups[1].Value.Trim() : "Без названия";
                Console.WriteLine($"Название: {poemTitle}");

                // Ищем контент со стихами - в классе entry-content poem-text
                var contentMatch = Regex.Match(poemHtml, @"<section\s+class=""entry-content poem-text""[^>]*>(.*?)</section>", RegexOptions.Singleline);

                if (!contentMatch.Success)
                    contentMatch = Regex.Match(poemHtml, @"<div\s+class=""entry-content poem-text""[^>]*>(.*?)</div>", RegexOptions.Singleline);

                if (!contentMatch.Success)
                    contentMatch = Regex.Match(poemHtml, @"<div\s+class=""poem-text""[^>]*>(.*?)</div>", RegexOptions.Singleline);

                if (!contentMatch.Success)
                    contentMatch = Regex.Match(poemHtml, @"<article[^>]*>(.*?)</article>", RegexOptions.Singleline);

                string contentHtml = contentMatch.Success ? contentMatch.Groups[1].Value : poemHtml;

                // Извлекаем все абзацы <p> из контента
                var pMatches = Regex.Matches(contentHtml, @"<p[^>]*>(.*?)</p>", RegexOptions.Singleline);
                var paragraphs = new List<string>();

                foreach (Match pMatch in pMatches)
                {
                    if (paragraphs.Count >= 5) break;

                    string pText = Regex.Replace(pMatch.Groups[1].Value, @"<[^>]+>", "");
                    pText = System.Net.WebUtility.HtmlDecode(pText).Trim();

                    // Фильтруем мусор: не берём абзацы с HTML-тегами, скриптами, слишком короткие
                    if (pText.Length > 30 &&
                        !pText.Contains("window.") &&
                        !pText.Contains("yaContext") &&
                        !pText.Contains("function") &&
                        !pText.StartsWith("<"))
                    {
                        paragraphs.Add(pText);
                    }
                }

                if (paragraphs.Count == 0)
                {
                    Console.WriteLine("  ⚠ Не найдено четверостиший в этом стихотворении");
                    continue;
                }

                string poemText = string.Join(" ", paragraphs);
                allAnalyzedText.Add($"--- {poemTitle} ---\n{poemText}\n");

                Console.WriteLine("\nАнализируемый текст (первые 5 четверостиший):");
                Console.WriteLine(new string('.', 60));
                foreach (var p in paragraphs)
                {
                    // Показываем первые 150 символов каждого четверостишия
                    string preview = p.Length > 150 ? p.Substring(0, 150) + "..." : p;
                    Console.WriteLine(preview);
                    Console.WriteLine();
                }
                Console.WriteLine(new string('.', 60));

                var words = Tokenize(poemText);
                Console.WriteLine($"  Найдено слов: {words.Count}");

                foreach (string word in words)
                {
                    wordsFrequency.TryGetValue(word, out int count);
                    wordsFrequency[word] = count + 1;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Ошибка при обработке: {ex.Message}");
            }
        }

        // Вывод результатов
        if (allAnalyzedText.Any())
        {
            Console.WriteLine("\n\n");
            Console.WriteLine(new string('=', 80));
            Console.WriteLine("ВЕСЬ ПРОАНАЛИЗИРОВАННЫЙ ТЕКСТ:");
            Console.WriteLine(new string('=', 80));

            foreach (string text in allAnalyzedText)
            {
                Console.WriteLine(text);
                Console.WriteLine(new string('-', 80));
            }

            Console.WriteLine("\n\n");
            Console.WriteLine(new string('=', 80));
            Console.WriteLine("ТОП-10 НАИБОЛЕЕ ЧАСТЫХ СЛОВ:");
            Console.WriteLine(new string('=', 80));

            var topWords = wordsFrequency.OrderByDescending(kvp => kvp.Value).Take(10).ToList();
            Console.WriteLine($"\n{"СЛОВО",-20} {"ЧАСТОТА",10}");
            Console.WriteLine(new string('-', 32));

            foreach (var (word, freq) in topWords)
            {
                Console.WriteLine($"{word,-20} {freq,10}");
            }

            Console.WriteLine($"\nВсего уникальных слов: {wordsFrequency.Count}");
            Console.WriteLine($"Всего слов (с повторениями): {wordsFrequency.Values.Sum()}");
            Console.WriteLine($"\nОбработано стихотворений: {poemNumber}");
        }
        else
        {
            Console.WriteLine("\nНЕ УДАЛОСЬ НАЙТИ ТЕКСТЫ СТИХОВ");
        }
    }

    static List<string> Tokenize(string text)
    {
        text = text.ToLowerInvariant();
        // Оставляем только русские и английские буквы, дефис и апостроф
        string clean = Regex.Replace(text, @"[^a-zа-яё\-']+", " ");
        var words = clean.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        // Убираем слишком короткие слова (1-2 буквы) и числа
        return words.Where(w => w.Length > 2 && !char.IsDigit(w[0])).ToList();
    }
}

*/
/*
internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
    }
}
*/