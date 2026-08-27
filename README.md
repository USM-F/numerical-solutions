# numerical-solutions

Репозиторий численных решений задач общей теории относительности. Старые реализации на Fortran хранятся в `fortran/`, а новые расчёты добавляются как подкоманды общего Python CLI.

## Установка

Требуется Python 3.10 или новее.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
```

## Расчёт голой сингулярности

Расчёт запускается установленной командой:

```bash
numerical-solutions naked-singularity
```

или напрямую через Python-модуль:

```bash
python -m numerical_solutions naked-singularity
```

Пример с изменёнными параметрами:

```bash
numerical-solutions naked-singularity \
  --mass-parameter 1.0 \
  --amplitude 1.0 \
  --x-min 1.8 \
  --x-max 7.0 \
  --rtol 1e-9 \
  --atol 1e-12 \
  --accuracy-threshold 1e-6 \
  --samples 1000 \
  --output-dir plots
```

Команда печатает минимум метрической функции, результат проверки горизонта и невязки обратно-прямого хода. Графики `metric_function.png` и `scalar_field.png` сохраняются в `plots/` или в каталоге из `--output-dir`. Каталог `plots/` игнорируется Git.

Полная математическая постановка, алгоритм и известные ограничения описаны в [`doc/naked-singularity-problem.md`](doc/naked-singularity-problem.md). Исходная презентация не требуется для запуска и намеренно не отслеживается.

## Тесты

```bash
python -m pytest
```

Набор содержит модульные тесты модели, решателя и графиков, регрессионную проверку параметров презентации и E2E-запуск CLI.
