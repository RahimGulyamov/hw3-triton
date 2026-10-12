# Домашнее задание 3: инференс модели в Triton

Предобученная MobileNetV3 Small классифицирует изображения на 1000 классов ImageNet. Triton обслуживает HTTP и gRPC через Python backend; внутри него выполняется PyTorch TorchScript. Есть две кастомные метрики, клиентские проверки, Performance Analyzer и конфигурация поиска Model Analyzer.

**Статус:** инференс проверен через HTTP/gRPC, обе кастомные метрики прошли проверку под нагрузкой, получены результаты Performance Analyzer. Результаты Model Analyzer и повторное измерение выбранной конфигурации ещё нужны для завершения отчёта. Все приведённые ниже цифры взяты из сохранённых артефактов запуска.

## Требования и выбор решения

- Linux x86_64, Docker с Compose v2, рекомендуются 8 ГБ RAM и 30 ГБ свободного диска.
- Для сборки нужен интернет: NGC, PyPI, download.pytorch.org. После сборки веса находятся в образе; при старте скачивание не требуется.
- GPU не нужен: используется `KIND_CPU`, один поток PyTorch на экземпляр модели. На Apple Silicon в Compose указана эмуляция `linux/amd64`, но её работоспособность не проверена; замеры для отчёта предполагаются на Linux x86_64.
- Базовый образ `nvcr.io/nvidia/tritonserver:23.12-py3`, NumPy `1.26.4`, PyTorch `2.1.2+cpu`, torchvision `0.16.2+cpu`, клиент `2.41.0`, Model Analyzer `1.35.0`. Прямые зависимости фиксированы в requirements-файлах; полный список установленных зависимостей сохраняется при профилировании.

Python backend позволяет разместить метрики непосредственно в модели. Дополнительная BLS-прокси не требуется. MobileNet выбрана как небольшой настоящий классификатор, пригодный для CPU; обучение с нуля не относится к цели задания.

Проверить Linux-машинку с SSH-алиасом `one-vm` из локального терминала:

```bash
bash scripts/check_vm.sh
```

Проверка показывает архитектуру, CPU, RAM, свободное место, Docker/Compose и доступность HTTPS-источников сборки. Она не устанавливает пакеты и не запускает контейнеры; результат записывается в `results/vm_preflight.txt`. HTTP 401 от `nvcr.io/v2/` ожидаем без получения registry token. Успешная проверка сети не гарантирует загрузку всех слоёв и весов: это проверяется сборкой.

Если общий адрес реестра возвращает ошибку, отдельно проверьте метаданные именно нужных образов, без скачивания слоёв:

```bash
bash scripts/check_registry.sh
```

Результат сохранится в `results/registry_check.txt`. Ошибка от общего `/v2/` сама по себе не доказывает недоступность конкретных образов; до сборки нужно проверить их manifests.

Для сравнения доступа из терминала и Docker выполните на Mac `python3 scripts/diagnose_registry.py`. Скрипт проверяет HTTPS, настройки прокси и manifests обоих образов; образы не скачиваются. Успешный `docker manifest inspect` проверяет доступ к manifest из Docker CLI; скачивание слоёв через Docker daemon требует отдельной проверки.

Если образы доступны с Mac, их можно скачать там и передать на VM через `docker save`/`docker load`. Из обычного терминала Mac в папке проекта:

```bash
bash scripts/run_on_vm.sh
```

Скрипт сначала использует уже имеющиеся на VM образы либо скачивает их напрямую. При недоступности реестра с VM он скачивает **большие образы** для `linux/amd64` через локальный Docker Desktop и передаёт их по SSH. Если реестр недоступен и с Mac, выполнение остановится до передачи кода и запуска модели.

Затем код передаётся в новый каталог `/tmp/hw3-run-<идентификатор>`, сборка и замеры выполняются на Linux, а логи/CSV/отчёты возвращаются в локальный `results/vm_run_<идентификатор>/`. Существующие удалённые каталоги не перезаписываются. Для запуска нужны свободные порты 8000–8002. Проверен shell-синтаксис скрипта; полный запуск ещё не выполнен.

В случае ошибки сборки или проверки скрипт забирает доступные результаты и сохраняет ненулевой exit status. Копии весов и checkpoints остаются на VM для последующего выбора конфигурации; в локальный каталог возвращаются компактные результаты. Повторное профилирование финального варианта выполняется после анализа этих результатов.

Перенос образов через архив документирован в [Docker save](https://docs.docker.com/reference/cli/docker/image/save/) и [инструкции NVIDIA по загрузке образов на машину без доступа к реестру](https://docs.nvidia.com/dgx/dgx-os-8-user-guide/latest/appendix_g_installing_docker_containers.html).

## Запуск одной командой

Из корня проекта:

```bash
docker compose up --build -d --wait triton
```

Для последовательного запуска сервера, проверки HTTP/gRPC, метрик, Performance Analyzer и Model Analyzer:

```bash
bash scripts/run_all.sh
```

Скрипт сохраняет логи и результаты в `results/` и останавливается при первой ошибке. Перед независимым повторным экспериментом перенесите предыдущие результаты в другой каталог. По завершении Model Analyzer сервер остаётся остановленным. Выбор конфигурации, повторное измерение выбранного варианта и заполнение отчёта выполняются после анализа полученных результатов.

Оба образа собираются до замеров. При сборке профилировщика `scripts/validate_analyzer_config.py` проверяет YAML штатным парсером установленного Model Analyzer 1.35.0, без запуска профилирования. Такая же проверка выполняется перед отдельным запуском Model Analyzer; её вывод сохраняется в `results/model_analyzer/validation.log`. Для CPU-профилирования используется `cpu_only: true`; параметр `dcgm_disable` эта версия не поддерживает.

На Mac ARM64 можно попробовать этот же скрипт в обычном терминале с работающим Docker Desktop: Compose уже задаёт `linux/amd64` для всех контейнеров. Docker Desktop [поддерживает эмуляцию](https://docs.docker.com/build/building/multi-platform/), но весь стек Triton/PyTorch/Model Analyzer на этом Mac пока не проверен. В отчёте такие замеры нужно обозначить как CPU-инференс в виртуальной машине с эмуляцией x86_64 на Apple Silicon.

Dockerfile скачивает предобученные веса, экспортирует их в TorchScript и сравнивает экспорт с исходной моделью на батчах 1, 4 и 8. Затем контейнер запускает `tritonserver` из `ENTRYPOINT`. Первая сборка может занять несколько минут.

```bash
docker compose ps
docker compose logs triton
curl -f http://localhost:8000/v2/health/ready
curl -f http://localhost:8000/v2/models/mobilenet/ready
curl -f http://localhost:8000/v2/models/mobilenet
```

Порты: `8000` — HTTP, `8001` — gRPC, `8002` — Prometheus. Модель должна иметь статус `READY`. Для остановки:

```bash
docker compose down
```

Можно использовать Docker напрямую:

```bash
docker build --platform linux/amd64 -t hw3-triton:23.12 .
docker run --rm --shm-size=1g -p 8000:8000 -p 8001:8001 -p 8002:8002 hw3-triton:23.12
```

## Запуск на GitHub Actions

Для профилирования на временной Linux x86_64 машине подготовлен `.github/workflows/profile.yaml`. Он запускает тот же `scripts/run_all.sh`: сборку, HTTP/gRPC проверки, проверку метрик, Performance Analyzer и Model Analyzer. Доступ к базовым образам проверяется в начале workflow; успешный запуск ещё не подтверждён.

1. Создайте публичный репозиторий GitHub и загрузите исходники, включая `.github/workflows/profile.yaml`, в основную ветку. Архив исходников без презентаций, локальных диагностик и весов создаётся командой `python3 scripts/package_source.py`; результат — `results/source.zip`. Для загрузки в Git архив сначала нужно распаковать.
2. Откройте **Actions → Triton CPU profiling → Run workflow**. Автоматического запуска при каждом push нет.
3. После завершения скачайте artifact `triton-cpu-results-<run_id>`. Распакуйте ZIP, а затем `profiling-results.tar.gz` в отдельный каталог: `mkdir -p results/github_run && tar -xzf /путь/к/profiling-results.tar.gz -C results/github_run`.
4. Для анализа доступны `results/github_run/results/run.log`, `performance.md`, результаты метрик, CSV/отчёты Model Analyzer и `configurations/` с конфигурациями вариантов. Даже при ошибке сохраняются доступные результаты; наличие artifact само по себе не означает успешного профилирования.

Временная машина освобождает место от ненужных для задания предустановленных инструментов, проверяет минимум 30 ГиБ свободного места и сохраняет фактические характеристики в `runner.txt`. Архив результатов не включает большие копии весов. Artifact хранится 7 дней: скачайте его и добавьте измерения и отчёт в итоговый репозиторий.

Замеры нужно описывать как **CPU-инференс на GitHub-hosted runner**, с характеристиками из `runner.txt`. Сравнивать конфигурации следует внутри одного запуска; различия аппаратуры между запусками могут влиять на результаты. После анализа потребуется повторная проверка выбранного варианта.

Использование стандартных hosted runners бесплатно для публичных репозиториев согласно [документации GitHub](https://docs.github.com/en/actions/reference/runners/github-hosted-runners). Порядок скачивания результатов описан в [документации artifacts](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/download-workflow-artifacts).

## Контракт и клиентские запросы

Вход `IMAGE`: FP32, форма `[B, 3, 224, 224]`, `1 <= B <= 8`. Выход `LOGITS`: FP32, `[B, 1000]`. Размерность батча не указывается в `dims` конфигурации, потому что её добавляет Triton при `max_batch_size > 0`.

Для настоящей картинки клиент преобразует её в RGB, изменяет меньшую сторону до 256, выполняет центральный кроп 224×224 и нормализует каналы с ImageNet mean/std. `softmax` и top-5 считаются на клиенте. Сервер принимает уже подготовленный тензор, а не JPEG.

Примеры запросов с воспроизводимым синтетическим входом и сравнением с локальной моделью:

```bash
docker compose --profile tools run --rm client clients/infer.py --protocol http --url triton:8000 --verify
docker compose --profile tools run --rm client clients/infer.py --protocol grpc --url triton:8001 --batch-size 4 --verify
docker compose --profile tools run --rm client clients/smoke.py
```

`smoke.py` проверяет оба протокола, батчи 1/4/8, совпадение логитов с локальным инференсом (`rtol=atol=1e-4`), отказ на неправильную форму входа и корректный ответ после такого отказа. Ожидаемые сообщения — `Remote/local logits match` и `PASS ...`. Это ожидаемые условия проверки, а не зафиксированный результат запуска.

Для классификации своей картинки положите файл в `examples/photo.jpg`:

```bash
docker compose --profile tools run --rm client clients/infer.py --url triton:8000 --image /examples/photo.jpg --verify
```

Синтетический вход проверяет инференс и протокол; вывод top-5 на шуме не оценивает качество классификации. Для примеров реальной классификации сохраните вывод на своих изображениях в итоговом отчёте.

## Кастомные метрики

| Метрика | Тип | Смысл |
| --- | --- | --- |
| `hw3_processing_seconds_total` | COUNTER | Сумма времени процессинга запросов в секундах |
| `hw3_inflight_requests` | GAUGE | Число запросов, выполняемых внутри `execute` |

Время измеряется `time.perf_counter()`: от входа в `execute` до формирования ответов. Оно включает объединение тензоров, инференс и разбиение ответов. Ожидание в очереди Triton и клиентская предобработка не включаются. При динамическом батчинге все запросы разделяют интервал процессинга: для N запросов в COUNTER добавляется `N × elapsed`. Это сумма request-seconds, которая может превышать прошедшее календарное время; для загрузки устройства используются штатные метрики.

GAUGE считает **запросы**, а не изображения в клиентском батче. Для каждого синхронного экземпляра он равен `len(requests)` во время выполнения и возвращается к нулю в `finally`, в том числе при исключении. Очередь не включена: gauge может быть меньше клиентской concurrency. Метрики имеют labels `model`, `version`, `instance`; instance — PID процесса Python backend, поскольку API версии 23.12 не предоставляет имя экземпляра. Для общего значения нужно суммировать экземпляры.

```bash
curl -f http://localhost:8002/metrics
docker compose --profile tools run --rm client clients/check_metrics.py --url triton:8000 --metrics-url http://triton:8002/metrics
```

Проверка создаёт 16 параллельных клиентов по 20 запросов, опрашивает метрики во время нагрузки и сохраняет `results/metrics/samples.csv` и `summary.json`. Успех: прирост counter > 0, наблюдаемый максимум gauge > 0, gauge после завершения = 0. Если короткий инференс не попал в интервалы опроса, увеличьте `--requests-per-worker`; искусственной задержки в модели нет.

Примеры PromQL:

```promql
sum(hw3_inflight_requests)
sum(rate(hw3_processing_seconds_total[1m]))
```

## Performance Analyzer

Сервер и SDK-контейнер находятся в одной сети Compose. SDK обращается к `triton:8000`; это предусмотренная заданием альтернатива `--net host`.

```bash
docker compose --profile tools run --rm perf
python3 scripts/perf_report.py results/perf_analyzer/baseline
```

Скрипт запускает `perf_analyzer -m mobilenet` с concurrency 1/2/4/8/16/32, HTTP, клиентским батчем 1, окнами измерения 15 секунд и максимум 20 попытками. Стабильность проверяется по throughput и **средней задержке**, с допустимой вариацией 10%; p99 также измеряется и попадает в отчёт, но не используется как критерий остановки. Каждая точка сохраняется отдельными `.log` и `.csv` в `results/perf_analyzer/baseline/`; настройки сохраняются в `settings.txt`. При нестабильном результате или ошибке команда завершается с ненулевым кодом. Входы генерирует Performance Analyzer; оценивается производительность обслуживания тензоров, без загрузки JPEG и клиентской предобработки.

Каждый экземпляр модели выполняет 10 прогревочных инференсов с нулевым тензором при инициализации, до перехода в READY. Прогрев не учитывается в кастомных метриках запросов. Model Analyzer использует те же параметры стабильности и длительность окон. p99 отражает задержки в хвосте распределения; стабильность среднего значения сама по себе не гарантирует стабильность p99. Перед выбором конфигурации нужно сравнить повторные замеры. Назначение параметров описано в [документации Performance Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/docs/cli.html).

Скрипт `perf_report.py` читает фактические значения из логов и переводит задержки из микросекунд в миллисекунды. Без замеров он выдаёт ошибку. Скопируйте выведенную таблицу в раздел результатов. Перед тестированием завершите клиентские проверки; они могут влиять на измерения.

## Model Analyzer

Отдельный target Dockerfile расширяет **тот же серверный образ**, устанавливает Model Analyzer через pip и содержит Performance Analyzer из SDK того же релиза. `triton_launch_mode: local` запускает `tritonserver` внутри этого контейнера с доступом к PyTorch и весам. Docker socket не нужен.

Для зависимостей `tritonclient[all]`, которые устанавливает Model Analyzer, зафиксирован `cuda-python==12.3.0`: версия 13 удаляет старые пространства имён CUDA API согласно [release notes NVIDIA](https://nvidia.github.io/cuda-python/cuda-bindings/latest/release/13.0.0-notes.html). Это зависимость инструментов; вычисления модели выполняются на CPU. При сборке образ проверяет запуск SDK через `perf_analyzer --version`, импорт CUDA shared-memory модуля клиента и загрузку CLI Model Analyzer. `scripts/run_all.sh` собирает оба образа до начала измерений.

Сначала завершите сервер, чтобы не создавать конкуренцию за CPU:

```bash
docker compose stop triton
docker compose --profile tools build analyzer
docker compose --profile tools run --rm analyzer
```

Конфигурация `profiling/model_analyzer.yaml` задаёт ручной перебор:

| Параметр | Значения |
| --- | --- |
| `max_batch_size` | 1, 4, 8 |
| `instance_group.count` | 1, 2 CPU-экземпляра |
| `max_queue_delay_microseconds` | 0, 1000 |
| Клиентская concurrency | 1, 2, 4, 8, 16 |
| Клиентский batch size | 1 |

Получается 12 конфигураций × 5 уровней нагрузки; отдельно сравнивается исходная конфигурация, если она включена в результаты инструмента. У всех перебираемых вариантов включён dynamic batching; исходный `config.pbtxt` его не включает. `max_batch_size=1` позволяет оценить поведение планировщика без объединения разных запросов.

Цель поиска — throughput. Явного latency budget нет, поэтому победитель по throughput может иметь неприемлемую p99: решение о финальной конфигурации принимается после изучения обеих величин. Ожидаемые гипотезы, которые нужно подтвердить замерами: несколько экземпляров могут использовать разные CPU-ядра; batching может уменьшить расходы на вызов модели; задержка накопления может увеличить p99 при малой нагрузке. Улучшение заранее не гарантируется.

Результаты сохраняются в `results/model_analyzer/`: логи `profile.log`/`report.log`/`triton.log`, окружение, CSV, summary-отчёты, подробные отчёты, checkpoints и репозиторий вариантов. Скрипт генерирует подробные отчёты для всех найденных конфигураций. Имена вариантов находятся в `config_names.txt`. Для отдельного повторного отчёта выберите реальные имена из этого файла:

```bash
docker compose --profile tools run --rm --entrypoint model-analyzer analyzer report \
  --report-model-configs mobilenet_config_default,mobilenet_config_0 \
  --checkpoint-directory /results/model_analyzer/checkpoints \
  --export-path /results/model_analyzer
```

Имена выше — пример синтаксиса; замените их именами, сгенерированными вашим запуском. Не удаляйте checkpoints до генерации отчётов. Для повторного независимого эксперимента переименуйте каталог `results/model_analyzer`, чтобы не смешивать измерения; существующий каталог вариантов автоматически не перезаписывается.

## Повторная проверка выбранной конфигурации

Укажите имя выбранного варианта из `config_names.txt` вместо `mobilenet_config_0`:

```bash
cp results/model_analyzer/model_repository/mobilenet_config_0/config.pbtxt results/selected-config.pbtxt
python3 scripts/select_config.py results/selected-config.pbtxt
docker compose up -d --wait triton
docker compose cp results/selected-config.pbtxt triton:/models/mobilenet/config.pbtxt
docker compose restart triton
docker compose up -d --wait triton
docker compose --profile tools run --rm client clients/smoke.py
docker compose --profile tools run --rm perf /results/perf_analyzer/selected
python3 scripts/perf_report.py results/perf_analyzer/selected
```

`select_config.py` меняет только имя модели в **копии** конфигурации на `mobilenet`, чтобы оно совпадало с именем каталога. После удаления/пересоздания контейнера вернётся исходная конфигурация из образа. Для постоянного использования замените `model_repository/mobilenet/config.pbtxt` выбранным файлом и пересоберите сервер. Клиентские батчи 4/8 работают только при соответствующем `max_batch_size`; для выбранного варианта с меньшим лимитом используйте `smoke.py --max-batch-size ...`.

Сравнивайте исходный и выбранный варианты при одинаковой concurrency, входах, протоколе и ресурсах. Для финального вывода повторите измерение лучших вариантов, чтобы проверить, что прирост превышает шум.

## Результаты и выводы

Измерения проведены 12 октября 2026 года на GitHub-hosted runner, Linux x86_64. Исходные артефакты сохранены в [results/github_baseline_20261012](results/github_baseline_20261012). Характеристики машины доступны в [runner.txt](results/github_baseline_20261012/runner.txt), версии Docker — в [host.txt](results/github_baseline_20261012/host.txt). Файл `artifact_sha256.json` содержит контрольные суммы импортированных исходных файлов.

| Параметр | Значение |
| --- | --- |
| CPU | AMD EPYC 7763; runner предоставил 4 vCPU |
| RAM | `free -h`: 15 GiB total, округлённое значение |
| ОС | Ubuntu 22.04, Linux 6.8.0-1064-azure |
| Docker / Compose | 28.0.4 / 2.38.2 |
| Triton | образ 23.12-py3, сервер 2.41.0 |
| Конфигурация модели | 1 экземпляр KIND_CPU, 1 поток PyTorch, max_batch_size=8, dynamic batching выключен |
| Нагрузка PA | HTTP, batch size=1, concurrency 1/2/4/8/16/32 |
| Стабилизация PA | окна 15 s, до 20 попыток, вариация throughput и mean latency ≤10% |

Контейнеру сервера не заданы дополнительные CPU quota или memory limit. Максимум batch size=8 разрешает клиентские батчи до восьми изображений; при выключенном dynamic batching отдельные запросы Performance Analyzer не объединяются. Каждое выполнение в этом тесте обрабатывает одно изображение. Окружение является виртуальной машиной общего назначения; повторные запуски могут дать другие результаты.

### Корректность HTTP/gRPC

В [smoke.log](results/github_baseline_20261012/checks/smoke.log) зафиксированы успешные проверки обоих протоколов для batch size 1, 4 и 8: серверные логиты совпадают с локальным TorchScript-инференсом при `rtol=atol=1e-4`. Неверная форма тензора отклоняется; последующий корректный запрос успешно обрабатывается. В серверном логе модель имеет статус READY.

Примеры отдельных HTTP/gRPC запросов сохранены в [http.log](results/github_baseline_20261012/checks/http.log) и [grpc.log](results/github_baseline_20261012/checks/grpc.log). Использован синтетический тензор с фиксированным seed; эти запросы проверяют численную корректность и протоколы, но не качество классификации на реальных изображениях.

### Кастомные метрики

Нагрузка: 16 клиентов по 20 запросов, всего **320 успешных запросов**. Результаты взяты из [summary.json](results/github_baseline_20261012/metrics/summary.json); временной ряд содержит 89 измерений за 1.904 s и сохранён в [samples.csv](results/github_baseline_20261012/metrics/samples.csv).

| Показатель | Измеренное значение |
| --- | ---: |
| Прирост processing COUNTER | 1.756555 s |
| Максимум наблюдаемого in-flight GAUGE | 1 |
| GAUGE после завершения нагрузки | 0 |
| Среднее время процессинга на запрос, прирост COUNTER / 320 | 5.489 ms |

COUNTER монотонно увеличивается во всех сохранённых samples. GAUGE положителен в 76 из 89 измерений и возвращается к нулю после нагрузки. Максимум 1 соответствует конфигурации: один синхронный экземпляр Python backend выполняет один запрос за раз, а остальные клиентские запросы ждут в очереди Triton. Очередь не входит в значение GAUGE.

![Кастомные метрики во время нагрузки](results/github_baseline_20261012/figures/metrics.png)

### Performance Analyzer

Все шесть точек достигли заданной стабильности. Логи и CSV доступны в [perf_analyzer/baseline](results/github_baseline_20261012/perf_analyzer/baseline); latency переведена из микросекунд в миллисекунды. p99 измерен, однако стабилизация проверялась по средней задержке. Проверка среднего значения не гарантирует устойчивость p99 при повторных запусках.

| Concurrency | Throughput, infer/s | Mean latency, ms | p99 latency, ms |
| ---: | ---: | ---: | ---: |
| 1 | 153.875 | 6.497 | 7.771 |
| 2 | 181.109 | 11.040 | 13.462 |
| 4 | 177.954 | 22.470 | 26.921 |
| 8 | 180.011 | 44.421 | 53.674 |
| 16 | 178.476 | 89.575 | 102.591 |
| 32 | 179.732 | 177.736 | 198.510 |

![Throughput и latency под нагрузкой](results/github_baseline_20261012/figures/performance.png)

Переход от concurrency=1 к 2 увеличивает throughput на **17.70%**, до 181.109 infer/s. При concurrency от 2 до 32 throughput остаётся в диапазоне 177.954–181.109 infer/s. В этом эксперименте исходная конфигурация достигает насыщения уже примерно при двух одновременных клиентских запросах.

Увеличение concurrency с 2 до 32 повышает среднюю задержку с 11.040 до 177.736 ms, то есть в **16.10 раза**, без прироста throughput. p99 повышается с 13.462 до 198.510 ms. Для исходной конфигурации concurrency=2 — наиболее выгодная из проверенных точек по сочетанию throughput и задержки; concurrency=1 даёт минимальную задержку.

Серверная детализация объясняет рост задержки:

| Concurrency | Средняя очередь Triton, ms | Compute infer, ms |
| ---: | ---: | ---: |
| 1 | 0.025 | 5.211 |
| 2 | 4.323 | 5.342 |
| 32 | 170.930 | 5.444 |

При concurrency=32 очередь занимает около **96.86%** средней серверной задержки 176.475 ms. Время compute infer изменяется гораздо меньше: с 5.211 до 5.444 ms. Это указывает на накопление запросов перед единственным экземпляром модели. Из этих данных нельзя заключить, что заняты все четыре vCPU: общий CPU utilization в этом запуске не измерялся. Увеличение числа экземпляров и dynamic batching остаются гипотезами для проверки Model Analyzer.

Таблицу и графики можно воспроизвести из исходных данных:

```bash
python3 scripts/perf_report.py results/github_baseline_20261012/perf_analyzer/baseline
# Нужен matplotlib==3.7.5; пакет включён в requirements-analyzer.txt.
python3 scripts/plot_results.py results/github_baseline_20261012
```

PNG и SVG графиков приложены в [figures](results/github_baseline_20261012/figures).

### Model Analyzer и выбор конфигурации

**Результатов Model Analyzer в этих артефактах нет.** Запуск остановился после Performance Analyzer на сборке образа профилировщика. Исправление Dockerfile подготовлено отдельно; запуск Model Analyzer и получение его отчётов ещё нужно подтвердить. Нельзя утверждать, что оптимальная конфигурация найдена, или заменять исходную конфигурацию на основании только этих данных.

После успешного профилирования требуется сравнить исходную и три лучшие конфигурации: max batch size, число экземпляров, queue delay, concurrency, throughput, p99 и RAM. Сравнение выполняется при одинаковой concurrency на одной машине. Затем нужно повторно измерить выбранный вариант; прирост throughput рассчитывается как `(T_selected / T_baseline - 1) × 100%`. Если улучшения не подтвердятся, вывод должен опираться на измеренные задержки очереди/compute и использование ресурсов. При смене runner исходный вариант также нужно повторно измерить на новой машине.

### Локальные проверки кода

Выполнены проверка Compose-конфигурации, синтаксиса Python и shell, а также три unit-теста обработки ошибок в метриках. Unit-тесты используют заглушки Triton API; реальный инференс подтверждён отдельными smoke-логами выше.

```bash
python3 -m unittest discover -s tests -v
```

## Структура

```text
Dockerfile                           сервер и образ профилировщика
.github/workflows/profile.yaml       ручное профилирование на GitHub Actions
compose.yaml                         сервер, клиенты, SDK, Model Analyzer
requirements*.txt                    фиксированные прямые зависимости
model_repository/mobilenet/           config.pbtxt и Python backend
clients/                             HTTP/gRPC, smoke, метрики под нагрузкой
scripts/                             экспорт, профилирование, таблица PA
profiling/model_analyzer.yaml         пространство поиска конфигураций
examples/                            пользовательские изображения
results/                             реальные логи, CSV и отчёты
docs/lecture-notes.md                 связь решения с лекциями
```

Веса создаются при сборке и находятся в образе, поэтому не требуют коммита. Checkpoints и большие копии весов в репозитории вариантов исключены из Git; CSV, логи и итоговые отчёты нужно приложить к сдаче. Презентации для запуска решения не нужны и исключены из Docker build context. `scripts/package_source.py` создаёт архив только исходников; для сдачи ZIP-архивом отдельно добавьте измерения и завершённый отчёт.

## Источники

- Лекция №13 «MLOps, Model Serving»: слайды 25–31 — протокол и Python backend; 38–46 — batching, Performance Analyzer и Model Analyzer.
- [Документация Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/contents.html).
- [Python backend и Custom Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#custom-metrics).
- [Конфигурация Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_analyzer/docs/config.html), [режимы запуска](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2530/user-guide/docs/model_analyzer/docs/launch_modes.html).
- [MobileNetV3 Small, веса и preprocessing](https://docs.pytorch.org/vision/0.16/models/generated/torchvision.models.mobilenet_v3_small.html).
- [Состав Triton 23.12](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2480/release-notes/rel-23-12.html).

Официальная документация по ссылкам может описывать более новые релизы; исполняемые образы и Python-пакеты этого решения зафиксированы отдельно.
