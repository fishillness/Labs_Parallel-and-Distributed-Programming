using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Lab3_ProgramB;

internal class Program
{
    static void Main(string[] args)
    {
        string path = "..\\..\\..\\img\\";
        string[] images = { "img1.jpg", "img2.jpg", "img3.jpg" };
        int repeats = 3;
        string inputPath = path + images[0];


        foreach (string imgName in images)
        {
            List<double> times = new List<double>();
            Console.WriteLine($"Изображение {imgName}:");

            var (imageBytes, width, height) = LoadImage(inputPath);

            // --- Инициализация ILGPU (CUDA) ---
            var gpu = InitializeCUDA();
            if (gpu == null)
                return;

            var (context, accelerator, intensityKernel, sobelXKernel, sobelYKernel,
                 magnitudeKernel, normalizeKernel) = gpu.Value;

            int pixelCount = width * height;
            int stride = width * 3; // ImageSharp: без выравнивания, ровно width*3 байт на строку
            int byteCount = stride * height;

            // --- Выделение буферов на GPU ---
            using var gpuImageBytes = accelerator.Allocate1D<byte>(byteCount);
            using var gpuIntensity = accelerator.Allocate1D<float>(pixelCount);
            using var gpuMX = accelerator.Allocate1D<float>(pixelCount);
            using var gpuMY = accelerator.Allocate1D<float>(pixelCount);
            using var gpuMR = accelerator.Allocate1D<float>(pixelCount);
            using var gpuOutput = accelerator.Allocate1D<byte>(pixelCount);

            // --- Копирование изображения с CPU на GPU ---
            gpuImageBytes.CopyFromCPU(imageBytes);

            for (int i = 0; i < repeats; i++)
            {
                // ============================================================
                // Начало замера времени (этапы 2–7: вся вычислительная обработка)
                // ============================================================
                Stopwatch sw = Stopwatch.StartNew();

                // --- Этап 2: Вычисление матрицы интенсивности I = (R + G + B) / 3 ---
                // Каждый поток: читает три канала своего пикселя и записывает среднее.
                intensityKernel(pixelCount, gpuImageBytes.View, gpuIntensity.View, width, stride);
                accelerator.Synchronize();

                // --- Этап 3: Фильтр Собеля по X — получение матрицы MX ---
                // Ядро [-1 0 1 / -2 0 2 / -1 0 1] выделяет вертикальные границы.
                sobelXKernel(pixelCount, gpuIntensity.View, gpuMX.View, width, height);
                accelerator.Synchronize();

                // --- Этап 4: Фильтр Собеля по Y — получение матрицы MY ---
                // Ядро [-1 -2 -1 / 0 0 0 / 1 2 1] выделяет горизонтальные границы.
                sobelYKernel(pixelCount, gpuIntensity.View, gpuMY.View, width, height);
                accelerator.Synchronize();

                // --- Этап 5: Матрица градиента MR = sqrt(MX² + MY²) ---
                // Объединяет X- и Y-компоненты в полную силу перехода.
                magnitudeKernel(pixelCount, gpuMX.View, gpuMY.View, gpuMR.View);
                accelerator.Synchronize();

                // --- Этап 6: Нахождение максимального значения MR (на CPU) ---
                // ILGPU не имеет встроенной редукции max, поэтому копируем MR на CPU
                // и ищем максимум там. Это необходимо для нормализации.
                float[] mrCpu = new float[pixelCount];
                gpuMR.CopyToCPU(mrCpu);
                float maxMR = mrCpu.Max();
                Console.WriteLine($"MR max: {maxMR:F4}");

                // --- Этап 7: Нормализация MR: MR[v] = MR[v] * 255 / maxMR ---
                // Приводим все значения к диапазону [0, 255] для сохранения как изображение.
                normalizeKernel(pixelCount, gpuMR.View, gpuOutput.View, maxMR);
                accelerator.Synchronize();

                // ============================================================
                // Конец замера времени
                // ============================================================
                sw.Stop();
                double time = sw.Elapsed.TotalSeconds;
                times.Add(time);
                Console.WriteLine($"  [{i}]: {Math.Round(time, 2)}");
            }
                       

            // --- Копирование результата с GPU на CPU ---
            byte[] outputBytes = new byte[pixelCount];
            gpuOutput.CopyToCPU(outputBytes);

            SaveImage(outputBytes, width, height, path, images[0]);

            context.Dispose();
            accelerator.Dispose();

            Console.WriteLine($"Для изображения {imgName} среднее время равно {Math.Round(times.Average(), 2)} \n");
        }
    }

    // =========================================================
    // Инициализирует ILGPU с бэкендом CUDA (NVIDIA GPU).
    // Возвращает null, если CUDA-устройство не найдено (нужна видеокарта NVIDIA).
    // Context и Accelerator в возвращаемом кортеже должны быть освобождены
    // вызывающим кодом через using.
    // =========================================================
    static (
        Context context,
        Accelerator accelerator,
        Action<Index1D, ArrayView<byte>, ArrayView<float>, int, int> intensityKernel,
        Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> sobelXKernel,
        Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> sobelYKernel,
        Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> magnitudeKernel,
        Action<Index1D, ArrayView<float>, ArrayView<byte>, float> normalizeKernel
    )? InitializeCUDA()
    {
        // Context.CreateDefault() включает CUDA, OpenCL и CPU — в зависимости от железа
        var context = Context.CreateDefault();

        // Ищем CUDA-устройство с наибольшим числом потоков (NVIDIA GPU)
        var cudaDevice = context.Devices
            .Where(d => d.AcceleratorType == AcceleratorType.Cuda)
            .OrderByDescending(d => d.MaxNumThreads)
            .FirstOrDefault();

        if (cudaDevice == null)
        {
            Console.WriteLine("Error: CUDA device not found. Requires an NVIDIA GPU and installed CUDA drivers.");
            context.Dispose();
            return null;
        }

        Console.WriteLine($"CUDA device: {cudaDevice.Name}");

        // Создаём акселератор — объект, через который ILGPU управляет устройством
        var accelerator = cudaDevice.CreateAccelerator(context);

        // Компилируем GPU-ядра (транслируются в PTX для CUDA).
        // LoadAutoGroupedStreamKernel автоматически подбирает оптимальный размер блока.
        var intensityKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<float>, int, int>(IntensityKernel);

        var sobelXKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(SobelXKernel);

        var sobelYKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(SobelYKernel);

        var magnitudeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(MagnitudeKernel);

        var normalizeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<byte>, float>(NormalizeKernel);

        return (context, accelerator,
                intensityKernel, sobelXKernel, sobelYKernel,
                magnitudeKernel, normalizeKernel);
    }

    // =========================================================
    // Загружает цветное изображение через ImageSharp и возвращает
    // сырые байты в формате RGB (3 байта на пиксель, без выравнивания).
    // Возвращает кортеж: массив байтов изображения, ширина, высота.
    // =========================================================
    static (byte[] imageBytes, int width, int height) LoadImage(string inputPath)
    {
        using var image = Image.Load<Rgb24>(inputPath);
        int width = image.Width;
        int height = image.Height;

        // CopyPixelDataTo даёт плотный массив без выравнивания (stride = width * 3)
        byte[] imageBytes = new byte[width * 3 * height];
        image.CopyPixelDataTo(imageBytes);

        return (imageBytes, width, height);
    }

    // =========================================================
    // Сохраняет одноканальный (grayscale) результат в файл.
    // Входной массив outputBytes: один байт = одна точка изображения.
    // Имя выходного файла: <исходное_имя>_processed.png.
    // Возвращает путь к сохранённому файлу.
    // =========================================================
    static void SaveImage(byte[] outputBytes, int width, int height, string path, string imgName)
    {
        // Создаём grayscale-изображение формата L8 (8 бит на пиксель)
        using var outputImage = Image.LoadPixelData<L8>(outputBytes, width, height);

        outputImage.Save(path + "new_" + imgName);
    }

    // =========================================================
    // GPU-ядро: вычисление матрицы интенсивности
    // I[v] = (R + G + B) / 3 — перевод цветного пикселя в яркость.
    // Каждый поток обрабатывает один пиксель.
    // imageBytes: RGB-байты (порядок R, G, B — как в ImageSharp Rgb24).
    // intensity: выходной массив float, индексируется как y * width + x.
    // =========================================================
    static void IntensityKernel(
        Index1D pixelIndex,
        ArrayView<byte> imageBytes,
        ArrayView<float> intensity,
        int width,
        int stride)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;
        int byteIdx = y * stride + x * 3;

        float r = imageBytes[byteIdx];     // R
        float g = imageBytes[byteIdx + 1]; // G
        float b = imageBytes[byteIdx + 2]; // B

        intensity[pixelIndex] = (r + g + b) / 3.0f;
    }

    // =========================================================
    // GPU-ядро: фильтр Собеля по оси X — получение матрицы MX.
    // Ядро: [-1 0 1 / -2 0 2 / -1 0 1].
    // Выделяет вертикальные границы (перепады яркости слева направо).
    // Граничная стратегия — расширение рамки (дублирование крайних пикселей).
    // =========================================================
    static void SobelXKernel(
        Index1D pixelIndex,
        ArrayView<float> intensity,
        ArrayView<float> mx,
        int width,
        int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        // Координаты соседних строк и столбцов с учётом границ изображения
        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1); // верхняя строка
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1); // нижняя строка
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);  // левый столбец
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);  // правый столбец

        // Применяем ядро Собеля X: средний столбец равен нулю (не учитывается)
        // [-1  0  1]
        // [-2  0  2]
        // [-1  0  1]
        float sum =
            -1.0f * intensity[yU * width + xL] + 1.0f * intensity[yU * width + xR] +
            -2.0f * intensity[y * width + xL] + 2.0f * intensity[y * width + xR] +
            -1.0f * intensity[yD * width + xL] + 1.0f * intensity[yD * width + xR];

        mx[pixelIndex] = sum;
    }

    // =========================================================
    // GPU-ядро: фильтр Собеля по оси Y — получение матрицы MY.
    // Ядро: [-1 -2 -1 / 0 0 0 / 1 2 1].
    // Выделяет горизонтальные границы (перепады яркости сверху вниз).
    // Граничная стратегия — расширение рамки (дублирование крайних пикселей).
    // =========================================================
    static void SobelYKernel(
        Index1D pixelIndex,
        ArrayView<float> intensity,
        ArrayView<float> my,
        int width,
        int height)
    {
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        int yU = IntrinsicMath.Clamp(y - 1, 0, height - 1);
        int yD = IntrinsicMath.Clamp(y + 1, 0, height - 1);
        int xL = IntrinsicMath.Clamp(x - 1, 0, width - 1);
        int xR = IntrinsicMath.Clamp(x + 1, 0, width - 1);

        // Применяем ядро Собеля Y: средняя строка равна нулю (не учитывается)
        // [-1  -2  -1]
        // [ 0   0   0]
        // [ 1   2   1]
        float sum =
            -1.0f * intensity[yU * width + xL] + -2.0f * intensity[yU * width + x] + -1.0f * intensity[yU * width + xR] +
             1.0f * intensity[yD * width + xL] + 2.0f * intensity[yD * width + x] + 1.0f * intensity[yD * width + xR];

        my[pixelIndex] = sum;
    }

    // =========================================================
    // GPU-ядро: вычисление матрицы градиента MR = sqrt(MX² + MY²).
    // Объединяет горизонтальную и вертикальную компоненты Собеля
    // в суммарную «силу» перехода (границы) в данной точке.
    // =========================================================
    static void MagnitudeKernel(
        Index1D pixelIndex,
        ArrayView<float> mx,
        ArrayView<float> my,
        ArrayView<float> mr)
    {
        float mxVal = mx[pixelIndex];
        float myVal = my[pixelIndex];
        mr[pixelIndex] = (float)Math.Sqrt(mxVal * mxVal + myVal * myVal);
    }

    // =========================================================
    // GPU-ядро: нормализация MR[v] = MR[v] * 255 / maxMR.
    // Приводит все значения к диапазону [0, 255].
    // Результат записывается как байт в выходной массив.
    // =========================================================
    static void NormalizeKernel(
        Index1D pixelIndex,
        ArrayView<float> mr,
        ArrayView<byte> output,
        float maxMR)
    {
        float normalized = mr[pixelIndex] * 255.0f / maxMR;
        output[pixelIndex] = (byte)IntrinsicMath.Clamp((int)normalized, 0, 255);
    }
}
