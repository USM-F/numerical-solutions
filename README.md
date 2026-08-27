# numerical-solutions

Репозиторий численных решений задач общей теории относительности. Старые реализации на Fortran хранятся в `fortran/`, а новые расчёты добавляются как подкоманды общего Python CLI.

## Установка

Требуется Python 3.10 или новее.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
```

Все команды проекта, включая тесты, запускаются только интерпретатором
`.venv/bin/python`; активация окружения не требуется.

## Расчёт голой сингулярности

Расчёт запускается через Python-модуль:

```bash
.venv/bin/python -m numerical_solutions naked-singularity
```

Пример с изменёнными параметрами:

```bash
.venv/bin/python -m numerical_solutions naked-singularity \
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

Полная математическая постановка, алгоритм и известные ограничения описаны в
[`doc/naked-singularity-problem.md`](doc/naked-singularity-problem.md).

## Заряженная скалярная чёрная дыра

Одиночный поиск решения раздела 2.6:

```bash
.venv/bin/python -m numerical_solutions charged-black-hole solve \
  --charge 1 --scalar-mass 1 --beta 0 --psi-h 0.3
```

Поиск атласа ветвей:

```bash
.venv/bin/python -m numerical_solutions charged-black-hole atlas
```

Быстрый первичный screening и углублённый повтор неразрешённых ячеек:

```bash
.venv/bin/python -m numerical_solutions charged-black-hole atlas \
  --disable-homotopy --jobs 3
.venv/bin/python -m numerical_solutions charged-black-hole atlas \
  --retry-invalid
```

Исправленная постановка и алгоритм приведены в
[`doc/charged-scalar-black-hole-problem.md`](doc/charged-scalar-black-hole-problem.md),
а построчная проверка формул PDF — в
[`doc/CSF2023-06-12-review.md`](doc/CSF2023-06-12-review.md).
Результаты записываются в игнорируемый Git каталог `results/`.

## Тесты

```bash
.venv/bin/python -m pytest
```

Набор содержит модульные и E2E-тесты. Отдельная обязательная CI-задача
`symbolic-verification` проверяет геометрию, горизонтный ряд до восьмого
порядка, асимптотические сектора и негативные подстановки с SymPy.
