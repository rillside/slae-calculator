import copy
from typing import Any


def format_num(x: float, precision: int = 4) -> str:
    """
    Форматирует вещественное число в аккуратную строку без избыточных нулей.

    Если число по модулю близко к нулю (< 1e-9), возвращает "0".
    Если число целое (например, 3.0), возвращает целое строковое представление "3".
    В остальных случаях округляет до указанной точности и убирает хвостовые нули.

    Args:
        x (float): Исходное вещественное число.
        precision (int, optional): Максимальное число знаков после запятой. По умолчанию 4.

    Returns:
        str: Отформатированная строка с числом.
    """
    if abs(x) < 1e-9:
        return "0"
    rounded = round(x, precision)
    if rounded == int(rounded):
        return str(int(rounded))
    s = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
    return s


def format_matrix(matrix: list[list[float]], precision: int = 4) -> str:
    """
    Форматирует двумерную матрицу в виде выровненной по столбцам текстовой таблицы.

    Выравнивает все элементы каждого столбца по максимальной ширине текстового
    представления элементов этого столбца, помещая строки в квадратные скобки.

    Args:
        matrix (list[list[float]]): Двумерный список чисел (матрица).
        precision (int, optional): Точность форматирования чисел. По умолчанию 4.

    Returns:
        str: Текстовое представление матрицы с выровненными столбцами.
    """
    if not matrix or not matrix[0]:
        return "[]"
    formatted = [[format_num(val, precision) for val in row] for row in matrix]
    col_widths = [
        max(len(formatted[i][j]) for i in range(len(formatted)))
        for j in range(len(formatted[0]))
    ]
    lines = []
    for row in formatted:
        line = "  [ " + "  ".join(val.rjust(col_widths[j]) for j, val in enumerate(row)) + " ]"
        lines.append(line)
    return "\n".join(lines)


def format_augmented_matrix(
    A: list[list[float]], B: list[float], precision: int = 4
) -> str:
    """
    Форматирует расширенную матрицу системы [A | B] с вертикальной чертой-разделителем.

    Выравнивает элементы матрицы коэффициентов A и вектор свободных членов B,
    разделяя их вертикальной чертой '|'.

    Args:
        A (list[list[float]]): Матрица коэффициентов системы уравнений.
        B (list[float]): Вектор свободных членов (правая часть уравнений).
        precision (int, optional): Точность форматирования чисел. По умолчанию 4.

    Returns:
        str: Текстовое представление расширенной матрицы [A | B].
    """
    if not A or not A[0]:
        return "[]"
    formatted_A = [[format_num(val, precision) for val in row] for row in A]
    formatted_B = [format_num(val, precision) for val in B]
    col_widths = [
        max(len(formatted_A[i][j]) for i in range(len(formatted_A)))
        for j in range(len(formatted_A[0]))
    ]
    b_width = max(len(val) for val in formatted_B)
    lines = []
    for i in range(len(A)):
        left = "  ".join(formatted_A[i][j].rjust(col_widths[j]) for j in range(len(formatted_A[0])))
        right = formatted_B[i].rjust(b_width)
        lines.append(f"  [ {left}  |  {right} ]")
    return "\n".join(lines)


def find_minor_matrix(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    """
    Возвращает подматрицу (минорную матрицу), полученную вычеркиванием строки i и столбца j.

    Args:
        matrix (list[list[float]]): Исходная квадратная матрица.
        i (int): Номер вычеркиваемой строки (начиная с 1).
        j (int): Номер вычеркиваемого столбца (начиная с 1).

    Returns:
        list[list[float]]: Новая матрица размерности (n-1) x (n-1).
    """
    sub_matrix = copy.deepcopy(matrix)
    sub_matrix.pop(i - 1)
    for row in sub_matrix:
        row.pop(j - 1)
    return sub_matrix


def determinant_matrix(matrix: list[list[float]]) -> float:
    """
    Вычисляет определитель квадратной матрицы методом разложения Лапласа.

    Для матрицы 1x1 возвращает единственный элемент.
    Для матрицы 2x2 вычисляет по формуле: a11*a22 - a12*a21.
    Для матриц порядка n > 2 выполняет рекурсивное разложение по элементам первой строки
    через алгебраические дополнения и миноры.

    Args:
        matrix (list[list[float]]): Квадратная матрица n x n.

    Returns:
        float: Значение определителя матрицы det(A).
    """
    n = len(matrix)
    if n == 1:
        return float(matrix[0][0])
    elif n == 2:
        return float(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])

    res = 0.0
    for j, element in enumerate(matrix[0]):
        minor = find_minor_matrix(matrix, 1, j + 1)
        cofactor = ((-1.0) ** (1 + (j + 1))) * determinant_matrix(minor)
        res += float(element) * cofactor

    return res


def solve_cramer(
    A: list[list[float]], B: list[float]
) -> tuple[list[float] | None, list[str]]:
    """
    Решает квадратную СЛАУ методом Крамера с формированием пошагового протокола.

    Алгоритм:
    1. Вычисляет главный определитель системы Delta = det(A).
    2. Проверяет условие применимости (Delta != 0).
    3. Для каждой неизвестной формирует матрицу A_j заменой j-го столбца на B.
    4. Вычисляет определители Delta_j и корни x_j = Delta_j / Delta.

    Args:
        A (list[list[float]]): Квадратная матрица коэффициентов системы (n x n).
        B (list[float]): Вектор свободных членов размерности n.

    Returns:
        tuple[list[float] | None, list[str]]:
            - Первый элемент: вектор найденных решений [x_1, ..., x_n] или None при Delta = 0.
            - Второй элемент: список строк с подробными промежуточными шагами решения.
    """
    steps: list[str] = []
    n = len(A)
    steps.append("МЕТОД КРАМЕРА")
    steps.append("1. Вычисление главного определителя системы Δ = det(A):")
    steps.append(format_matrix(A))

    det_A = determinant_matrix(A)
    steps.append(f"Δ = {format_num(det_A)}")

    if abs(det_A) < 1e-9:
        steps.append("\nГлавный определитель Δ = 0.")
        steps.append(
            "Метод Крамера неприменим, так как деление на ноль невозможно. "
            "Система либо несовместна, либо имеет бесконечно много решений."
        )
        return None, steps

    steps.append("\n2. Вычисление вспомогательных определителей Δ_i:")
    solutions: list[float] = []
    for j in range(n):
        Aj = copy.deepcopy(A)
        for i in range(n):
            Aj[i][j] = B[i]

        det_j = determinant_matrix(Aj)
        xj = det_j / det_A
        solutions.append(xj)

        steps.append(f"\nМатрица A_{j+1} (столбец {j+1} заменён на вектор свободных членов B):")
        steps.append(format_matrix(Aj))
        steps.append(f"Δ_{j+1} = {format_num(det_j)}")
        steps.append(
            f"x_{j+1} = Δ_{j+1} / Δ = {format_num(det_j)} / {format_num(det_A)} = {format_num(xj)}"
        )

    steps.append("\n3. Итоговый ответ:")
    for j, val in enumerate(solutions):
        steps.append(f"x_{j+1} = {format_num(val)}")

    return solutions, steps


def solve_matrix(
    A: list[list[float]], B: list[float]
) -> tuple[list[float] | None, list[str]]:
    """
    Решает квадратную СЛАУ матричным методом (X = A^(-1) * B) с подробным протоколом.

    Алгоритм:
    1. Вычисляет определитель det(A). Если det(A) = 0, матрица вырождена, решения нет.
    2. Вычисляет матрицу алгебраических дополнений C_ij = (-1)^(i+j) * det(M_ij).
    3. Транспонирует матрицу C для получения союзной (присоединенной) матрицы A* = C^T.
    4. Строит обратную матрицу A^(-1) = (1 / det(A)) * A*.
    5. Перемножает матрицу A^(-1) на вектор свободных членов B: X = A^(-1) * B.

    Args:
        A (list[list[float]]): Квадратная матрица коэффициентов системы (n x n).
        B (list[float]): Вектор свободных членов размерности n.

    Returns:
        tuple[list[float] | None, list[str]]:
            - Первый элемент: вектор решений [x_1, ..., x_n] или None при вырожденной матрице.
            - Второй элемент: список строк с подробными промежуточными шагами вычислений.
    """
    steps: list[str] = []
    n = len(A)
    steps.append("МАТРИЧНЫЙ МЕТОД")
    steps.append("1. Вычисление определителя матрицы системы det(A):")
    steps.append(format_matrix(A))

    det_A = determinant_matrix(A)
    steps.append(f"det(A) = {format_num(det_A)}")

    if abs(det_A) < 1e-9:
        steps.append("\nОпределитель матрицы равен нулю: det(A) = 0.")
        steps.append(
            "Матрица является вырожденной, обратная матрица A^(-1) не существует. "
            "Матричный метод неприменим."
        )
        return None, steps

    if n == 1:
        inv_A = [[1.0 / det_A]]
        steps.append(f"\nДля матрицы 1x1 обратная матрица: A^(-1) = [{format_num(inv_A[0][0])}]")
    else:
        steps.append("\n2. Нахождение матрицы алгебраических дополнений C:")
        C: list[list[float]] = []
        for i in range(n):
            c_row: list[float] = []
            for j in range(n):
                minor = find_minor_matrix(A, i + 1, j + 1)
                det_m = determinant_matrix(minor)
                sign = (-1) ** ((i + 1) + (j + 1))
                cofactor = sign * det_m
                c_row.append(cofactor)
                steps.append(
                    f"A_{i+1}{j+1} = (-1)^({i+1}+{j+1}) * M_{i+1}{j+1} = "
                    f"{'+' if sign > 0 else '-'}1 * {format_num(det_m)} = {format_num(cofactor)}"
                )
            C.append(c_row)

        steps.append("\nМатрица алгебраических дополнений C:")
        steps.append(format_matrix(C))

        adj_A = [[C[j][i] for j in range(n)] for i in range(n)]
        steps.append(
            "\n3. Транспонирование матрицы дополнений (присоединенная матрица A* = C^T):"
        )
        steps.append(format_matrix(adj_A))

        # Обратная матрица A^(-1) = (1 / det(A)) * A*
        inv_A = [[adj_A[i][j] / det_A for j in range(n)] for i in range(n)]
        steps.append(
            f"\n4. Вычисление обратной матрицы A^(-1) = (1 / det(A)) * A* = (1 / {format_num(det_A)}) * A*:"
        )
        steps.append(format_matrix(inv_A))

    steps.append("\n5. Нахождение вектора неизвестных X = A^(-1) * B:")
    X: list[float] = []
    for i in range(n):
        calc_terms: list[str] = []
        total = 0.0
        for j in range(n):
            term = inv_A[i][j] * B[j]
            total += term
            calc_terms.append(f"({format_num(inv_A[i][j])} * {format_num(B[j])})")
        X.append(total)
        steps.append(f"x_{i+1} = " + " + ".join(calc_terms) + f" = {format_num(total)}")

    steps.append("\n6. Итоговый ответ:")
    for i, val in enumerate(X):
        steps.append(f"x_{i+1} = {format_num(val)}")

    return X, steps


def solve_gauss(
    A: list[list[float]], B: list[float]
) -> tuple[list[float] | None, list[str]]:
    """
    Решает СЛАУ методом последовательного исключения неизвестных
    «Метод Гаусса с частичным выбором ведущего элемента» (Partial Pivoting Gaussian Elimination).

    Его суть состоит из 4 строгих этапов:
    ЭТАП 1. Составление расширенной матрицы [A | B].
            Объединяем коэффициенты при переменных и свободные члены в одну таблицу.

    ЭТАП 2. Прямой ход (приведение к ступенчатому виду, сверху вниз, слева направо):
            - Идем строго по столбцам слева направо (col = 0, 1, 2, ...).
            - В текущем столбце ищем строку с НАИБОЛЬШИМ ПО МОДУЛЮ числом (ведущий элемент).
              Зачем? Если там ноль, делить на него нельзя. А если взять максимальное число,
              деление на него минимизирует погрешности округления типа float.
            - Переставляем эту лучшую строку на место текущей рабочей строки (lead_row).
            - С помощью этой строки обнуляем все элементы, стоящие СТРОГО ПОД НЕЙ.
            - Сдвигаемся на строку ниже (lead_row += 1) и повторяем для следующего столбца.
            - На выходе получаем ступенчатую (верхнетреугольную) матрицу.

    ЭТАП 3. Анализ совместности (Теорема Кронекера–Капелли):
            - Проверяем матрицу на противоречия: если появилась строка вида [0 0 ... 0 | b]
              при b != 0 (уравнение 0 = b), система НЕ ИМЕЕТ РЕШЕНИЙ (несовместна).
            - Считаем ранг (количество ненулевых строк). Если ранг < числа неизвестных (n),
              уравнений меньше, чем переменных => БЕСКОНЕЧНО МНОГО РЕШЕНИЙ.
            - Если ранг == n, то решение ЕДИНСТВЕННОЕ, и мы переходим к шагу 4.

    ЭТАП 4. Обратный ход (нахождение неизвестных, снизу вверх):
            - Из самого нижнего оставшегося уравнения выражаем последнюю переменную.
            - Поднимаемся на строчку выше, подставляем уже найденные переменные
              и находим следующую, и так до самого верха (до x_1).

    Args:
        A (list[list[float]]): Матрица коэффициентов системы размера m x n.
        B (list[float]): Вектор свободных членов размерности m.

    Returns:
        tuple[list[float] | None, list[str]]:
            - Первый элемент: вектор решений [x_1, ..., x_n] при единственном решении,
              либо None (если решений нет или их бесконечно много).
            - Второй элемент: подробный список текстовых строк с описанием каждого шага.
    """
    steps: list[str] = []

    m = len(A)
    n = len(A[0])

    steps.append("МЕТОД ГАУССА")
    steps.append("1. Составление расширенной матрицы системы [A | B]:")

    M = [[float(A[i][j]) for j in range(n)] + [float(B[i])] for i in range(m)]
    steps.append(
        format_augmented_matrix(
            [[M[i][j] for j in range(n)] for i in range(m)], [M[i][n] for i in range(m)]
        )
    )
    steps.append("\n2. Прямой ход метода Гаусса (приведение к ступенчатому виду):")

    # Индекс текущей рабочей строки
    lead_row = 0

    # Список для хранения координат опорных элементов в формате (row, col)
    pivots: list[tuple[int, int]] = []

    # Идем слева направо по каждому столбцу матрицы коэффициентов
    for col in range(n):
        if lead_row >= m:
            break

        # Начинаем поиск строки с максимальным по модулю числом в текущем столбце col
        max_row = lead_row
        # Запоминаем абсолютное значение элемента в этой позиции
        max_val = abs(M[lead_row][col])

        # Проверяем все строки, расположенные строго ниже lead_row
        for r in range(lead_row + 1, m):
            # Если находим элемент больше по модулю, чем текущий максимум
            if abs(M[r][col]) > max_val:
                # Обновляем максимальное значение
                max_val = abs(M[r][col])
                # Запоминаем индекс этой строки
                max_row = r

        # Если даже максимальный элемент в этом столбце практически равен 0
        if max_val < 1e-9:
            # Обнулять нечего
            continue

        # Если строка с максимальным числом оказалась ниже, чем текущая lead_row
        if max_row != lead_row:
            # Меняем местами текущую строку lead_row и найденную строку max_row
            M[lead_row], M[max_row] = M[max_row], M[lead_row]
            # Записываем факт перестановки строк в отчет
            steps.append(f"\nПерестановка строк: меняем строку {lead_row + 1} и строку {max_row + 1}:")
            steps.append(
                format_augmented_matrix(
                    [[M[i][j] for j in range(n)] for i in range(m)],
                    [M[i][n] for i in range(m)],
                )
            )

        # Запоминаем значение ведущего элемента
        pivot_val = M[lead_row][col]

        pivots.append((lead_row, col))
        eliminated = False

        # Обнуляем элементы во всех строках, расположенных СТРОГО НИЖЕ опорной строки
        for r in range(lead_row + 1, m):
            # Вычисляем коэффициент домножения: во сколько раз текущий элемент больше опорного
            factor = M[r][col] / pivot_val

            # Если элемент под опорным не 0
            if abs(factor) > 1e-9:
                eliminated = True
                steps.append(
                    f"\nПреобразование строки {r + 1}: R_{r + 1} = R_{r + 1} - ({format_num(factor)}) * R_{lead_row + 1}"
                )
                # Вычитаем из каждого элемента строки r соответствующий элемент строки lead_row, умноженный на factor
                for c in range(col, n + 1):
                    M[r][c] -= factor * M[lead_row][c]
                # Принудительно ставим чистый ноль 0.0 в текущий столбец, чтобы убрать крошечную погрешность float
                M[r][col] = 0.0

        if eliminated:
            steps.append("Матрица после шага:")
            steps.append(
                format_augmented_matrix(
                    [[M[i][j] for j in range(n)] for i in range(m)],
                    [M[i][n] for i in range(m)],
                )
            )
        lead_row += 1

    steps.append("\nИтоговая ступенчатая матрица:")
    steps.append(
        format_augmented_matrix(
            [[M[i][j] for j in range(n)] for i in range(m)], [M[i][n] for i in range(m)]
        )
    )

    steps.append("\n3. Анализ совместности системы:")

    for r in range(m - 1, -1, -1):
        # Проверяем, являются ли все коэффициенты при неизвестных в строке r нулевыми
        all_zeros = all(abs(M[r][c]) < 1e-9 for c in range(n))
        # Если все коэффициенты при неизвестных нули, а свободный член справа не равен нулю
        if all_zeros and abs(M[r][n]) > 1e-9:
            steps.append(f"Строка {r + 1} имеет вид: 0 = {format_num(M[r][n])}, что невозможно.")
            steps.append("\nВывод: Система не имеет решений (несовместна).")
            return None, steps

    # Ранг матрицы равен количеству найденных опорных элементов (числу ненулевых строк в ступенчатом виде)
    rank = len(pivots)
    steps.append(f"Ранг матрицы коэффициентов rank(A) = {rank}, число неизвестных n = {n}.")

    # Если уровнений меньше, чем переменных
    if rank < n:
        steps.append("\nВывод: Система совместна и имеет бесконечно много решений")
        return None, steps

    steps.append("\n4. Обратный ход метода Гаусса (нахождение неизвестных):")

    X = [0.0] * n
    for r, c in reversed(pivots):
        sub_sum = 0.0
        terms: list[str] = []

        # Перебираем все столбцы правее текущего опорного элемента (там стоят уже найденные переменные)
        for j in range(c + 1, n):
            if abs(M[r][j]) > 1e-9:
                sub_sum += M[r][j] * X[j]
                terms.append(f"({format_num(M[r][j])} * {format_num(X[j])})")

        # Вычисляем правую часть: из свободного члена вычитаем сумму перенесенных слагаемых
        rhs = M[r][n] - sub_sum

        # Делим полученную правую часть на коэффициент при текущей неизвестной M[r][c]
        val = rhs / M[r][c]

        # Записываем найденное значение неизвестной в итоговый вектор корней на позицию c
        X[c] = val

        if terms:
            expr_sub = " - " + " - ".join(terms)
        else:
            expr_sub = ""

        steps.append(
            f"x_{c + 1} = ({format_num(M[r][n])}{expr_sub}) / {format_num(M[r][c])} = {format_num(val)}"
        )

    steps.append("\n5. Итоговый ответ:")

    for j in range(n):
        steps.append(f"x_{j + 1} = {format_num(X[j])}")

    return X, steps


def solve_slae_all(A: list[list[float]], B: list[float]) -> dict[str, Any]:
    """
    Комплексно решает СЛАУ всеми тремя методами с полной проверкой применимости.

    Выполняет:
    1. Решение методом Гаусса (применим к любой системе).
    2. Проверку квадратности (m == n). Если система прямоугольная, фиксирует
       неприменимость методов Крамера и матричного метода с теоретическим обоснованием.
    3. Для квадратных систем проверяет условие невырожденности det(A) != 0.
       Если det(A) = 0, блокирует вычисление по Крамеру и матричным методом.
    4. Если условия соблюдены, вычисляет корни методами Крамера и матричным методом.
    5. Формирует структурированный словарь результатов и сводный текстовый отчет.

    Args:
        A (list[list[float]]): Матрица коэффициентов системы уравнений (m x n).
        B (list[float]): Вектор свободных членов (m элементов).

    Returns:
        dict[str, Any]: Словарь, содержащий:
            - "is_square" (bool): является ли матрица квадратной;
            - "m" (int): число уравнений;
            - "n" (int): число неизвестных;
            - "gauss" (dict): результаты метода Гаусса;
            - "cramer" (dict): результаты метода Крамера;
            - "matrix" (dict): результаты матричного метода;
            - "summary" (list[str]): строки сводного отчета.
    """
    m = len(A)
    n = len(A[0])
    is_square = (m == n)

    results: dict[str, Any] = {
        "is_square": is_square,
        "m": m,
        "n": n,
        "gauss": {"applicable": True, "solution": None, "steps": [], "message": ""},
        "cramer": {"applicable": False, "solution": None, "steps": [], "message": ""},
        "matrix": {"applicable": False, "solution": None, "steps": [], "message": ""},
        "summary": [],
    }

    # Решаем методом Гаусса
    gauss_sol, gauss_steps = solve_gauss(A, B)
    results["gauss"]["solution"] = gauss_sol
    results["gauss"]["steps"] = gauss_steps

    if not is_square:
        msg = (
            f"Неприменим: матрица системы не является квадратной "
            f"(число уравнений m = {m} не равно числу неизвестных n = {n}). "
            f"Прямоугольные системы можно решать только методом Гаусса."
        )
        results["cramer"]["applicable"] = False
        results["cramer"]["message"] = msg
        results["cramer"]["steps"] = [
            "МЕТОД КРАМЕРА",
            msg,
        ]

        results["matrix"]["applicable"] = False
        results["matrix"]["message"] = msg
        results["matrix"]["steps"] = [
            "МАТРИЧНЫЙ МЕТОД",
            msg,
        ]
    else:
        det_A = determinant_matrix(A)
        if abs(det_A) < 1e-9:
            msg_cramer = (
                f"Неприменим: главный определитель системы равен нулю (Δ = {format_num(det_A)}). "
                f"Формулы Крамера требуют деления на Δ."
            )
            results["cramer"]["applicable"] = False
            results["cramer"]["message"] = msg_cramer
            results["cramer"]["steps"] = [
                "МЕТОД КРАМЕРА",
                f"Главный определитель Δ = det(A) = {format_num(det_A)}.",
                msg_cramer,
            ]

            msg_matrix = (
                f"Неприменим: определитель матрицы равен нулю (det(A) = {format_num(det_A)}). "
                f"Матрица является вырожденной, обратная матрица A^(-1) не существует."
            )
            results["matrix"]["applicable"] = False
            results["matrix"]["message"] = msg_matrix
            results["matrix"]["steps"] = [
                "МАТРИЧНЫЙ МЕТОД",
                f"Определитель матрицы det(A) = {format_num(det_A)}.",
                msg_matrix,
            ]
        else:
            cramer_sol, cramer_steps = solve_cramer(A, B)
            results["cramer"]["applicable"] = True
            results["cramer"]["solution"] = cramer_sol
            results["cramer"]["steps"] = cramer_steps

            matrix_sol, matrix_steps = solve_matrix(A, B)
            results["matrix"]["applicable"] = True
            results["matrix"]["solution"] = matrix_sol
            results["matrix"]["steps"] = matrix_steps

    summary: list[str] = []
    summary.append("СВОДНЫЙ ОТЧЕТ")
    summary.append(f"Размерность системы: {m} уравнений, {n} неизвестных.")
    if is_square:
        summary.append(f"Матрица квадратная ({m}x{n}).")
    else:
        summary.append(f"Матрица прямоугольная ({m}x{n}).")

    summary.append("\nПрименимость методов")
    summary.append("1. Метод Гаусса: ПРИМЕНИМ")
    if results["cramer"]["applicable"]:
        summary.append("2. Метод Крамера: ПРИМЕНИМ")
    else:
        summary.append(f"2. Метод Крамера: НЕ ПРИМЕНИМ ({results['cramer']['message']})")

    if results["matrix"]["applicable"]:
        summary.append("3. Матричный метод: ПРИМЕНИМ")
    else:
        summary.append(f"3. Матричный метод: НЕ ПРИМЕНИМ ({results['matrix']['message']})")

    summary.append("\nИтог решения")
    if gauss_sol is not None:
        summary.append("Система имеет единственное решение:")
        for idx, val in enumerate(gauss_sol):
            summary.append(f"  x_{idx+1} = {format_num(val)}")
    else:
        # Извлекаем последний содержательный вывод из шагов Гаусса
        last_concl = [line for line in gauss_steps if "Система" in line]
        if last_concl:
            summary.append(f"Результат: {last_concl[-1].strip()}")
        else:
            summary.append("Решение не найдено.")

    results["summary"] = summary
    return results
