import tkinter as tk
from tkinter import messagebox, ttk

from core import solve_slae_all


class App:
    """
    Главный класс графического интерфейса приложения для решения СЛАУ.

    Отвечает за:
    - Построение адаптивного оконного интерфейса на Tkinter / ttk;
    - Динамическое создание сетки ввода коэффициентов матрицы произвольного размера;
    - Загрузку демонстрационных пресетов (квадратных, прямоугольных, вырожденных);
    - Валидацию пользовательского ввода и обработку ошибок;
    - Вызов математического ядра и вывод пошаговых решений во вкладках.
    """

    def __init__(self, root: tk.Tk) -> None:
        """
        Инициализирует графический интерфейс приложения.

        Args:
            root (tk.Tk): Главное окно Tkinter.
        """
        self.root: tk.Tk = root
        self.root.title("Решение СЛАУ")
        self.root.geometry("920x760")
        self.root.minsize(750, 550)

        # Хранение ссылок на поля ввода (Entry)
        self.entries_A: list[list[ttk.Entry]] = []  # 2D список Entry для матрицы A
        self.entries_B: list[ttk.Entry] = []        # 1D список Entry для вектора B
        self.current_m: int = 0
        self.current_n: int = 0

        self._apply_styles()
        self._create_widgets()

        # Создаем начальную систему 3x3 по умолчанию и заполняем типовым примером
        self.set_system_dimensions(3, 3)
        self.load_preset_3x3()

    def _apply_styles(self) -> None:
        """
        Настраивает визуальные стили ttk для кнопок, вкладок и меток.
        """
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Header.TLabel", font=("Arial", 11, "bold"))
        style.configure("SubHeader.TLabel", font=("Arial", 10))
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))
        style.configure("TNotebook.Tab", font=("Arial", 10), padding=[10, 5])

    def _create_widgets(self) -> None:
        """
        Создает и размещает основные визуальные контейнеры интерфейса.
        """
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Панель параметров системы (размеры и кнопки примеров)
        config_frame = ttk.LabelFrame(main_frame, text=" Параметры системы (СЛАУ) ", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 8))

        dim_frame = ttk.Frame(config_frame)
        dim_frame.pack(side=tk.LEFT, fill=tk.X)

        ttk.Label(dim_frame, text="Число уравнений (m):").grid(row=0, column=0, padx=4, pady=2, sticky=tk.W)
        self.m_entry = ttk.Entry(dim_frame, width=5, justify="center")
        self.m_entry.grid(row=0, column=1, padx=4, pady=2)
        self.m_entry.insert(0, "3")

        ttk.Label(dim_frame, text="Число переменных (n):").grid(row=0, column=2, padx=4, pady=2, sticky=tk.W)
        self.n_entry = ttk.Entry(dim_frame, width=5, justify="center")
        self.n_entry.grid(row=0, column=3, padx=4, pady=2)
        self.n_entry.insert(0, "3")

        create_btn = ttk.Button(dim_frame, text="Создать сетку", command=self._on_create_matrix_click)
        create_btn.grid(row=0, column=4, padx=8, pady=2)

        # Блок кнопок с готовыми примерами
        presets_frame = ttk.Frame(config_frame)
        presets_frame.pack(side=tk.RIGHT, padx=5)

        ttk.Label(presets_frame, text="Примеры:", font=("Arial", 9, "italic")).pack(side=tk.LEFT, padx=4)
        ttk.Button(presets_frame, text="3×3 (квадратная)", command=self.load_preset_3x3).pack(side=tk.LEFT, padx=3)
        ttk.Button(presets_frame, text="2×3 (прямоугольная)", command=self.load_preset_2x3).pack(side=tk.LEFT, padx=3)
        ttk.Button(presets_frame, text="3×3 (det = 0)", command=self.load_preset_det0).pack(side=tk.LEFT, padx=3)

        # 2. Область ввода матрицы со скроллбарами
        self.matrix_box = ttk.LabelFrame(main_frame, text=" Ввод коэффициентов [A | B] ", padding=8)
        self.matrix_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8))

        self.canvas = tk.Canvas(self.matrix_box, height=150, highlightthickness=0)
        self.scroll_y = ttk.Scrollbar(self.matrix_box, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_x = ttk.Scrollbar(self.matrix_box, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.matrix_grid_frame = ttk.Frame(self.canvas)

        self.matrix_grid_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.matrix_grid_frame, anchor="nw")
        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        # 3. Панель действий
        actions_frame = ttk.Frame(main_frame)
        actions_frame.pack(fill=tk.X, pady=(0, 8))

        solve_btn = ttk.Button(
            actions_frame,
            text="★ Решить СЛАУ всеми методами",
            style="Accent.TButton",
            command=self.solve_system
        )
        solve_btn.pack(side=tk.LEFT, padx=5, ipady=3)

        clear_btn = ttk.Button(
            actions_frame,
            text="Очистить поля",
            command=self.clear_fields
        )
        clear_btn.pack(side=tk.LEFT, padx=5, ipady=3)

        self.status_label = ttk.Label(
            actions_frame,
            text="",
            font=("Arial", 10, "bold"),
            foreground="#0066cc"
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # 4. Область результатов (Notebook с четырьмя вкладками)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.summary_text: tk.Text = self._create_tab(" Сводка ")
        self.gauss_text: tk.Text = self._create_tab(" Метод Гаусса ")
        self.cramer_text: tk.Text = self._create_tab(" Метод Крамера ")
        self.matrix_text: tk.Text = self._create_tab(" Матричный метод ")

    def _create_tab(self, tab_title: str) -> tk.Text:
        """
        Создает вкладку в блокноте с многострочным текстовым полем и скроллбаром.

        Args:
            tab_title (str): Название вкладки.

        Returns:
            tk.Text: Созданный виджет текстового поля для вывода информации.
        """
        tab_frame = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(tab_frame, text=tab_title)

        text_widget = tk.Text(
            tab_frame,
            font=("Consolas", 10),
            wrap=tk.WORD,
            bg="#fdfdfd",
            fg="#222222",
            padx=8,
            pady=8
        )
        scroll = ttk.Scrollbar(tab_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scroll.set)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget.insert(
            tk.END,
            "Здесь будет отображено подробное решение после нажатия «Решить СЛАУ всеми методами»."
        )
        text_widget.configure(state=tk.DISABLED)

        return text_widget

    def _on_create_matrix_click(self) -> None:
        """
        Обрабатывает нажатие на кнопку «Создать сетку», проверяя корректность размеров m и n.
        """
        try:
            m = int(self.m_entry.get().strip())
            n = int(self.n_entry.get().strip())

            if m < 1 or n < 1:
                raise ValueError("Размеры должны быть не менее 1")
            if m > 15 or n > 15:
                raise ValueError("Максимальный поддерживаемый размер для ручного ввода: 15x15")

        except ValueError as e:
            messagebox.showerror(
                "Ошибка ввода",
                f"Некорректный размер системы: {e}.\nВведите целые числа от 1 до 15."
            )
            return

        self.set_system_dimensions(m, n)

    def set_system_dimensions(self, m: int, n: int) -> None:
        """
        Динамически перестраивает сетку полей ввода Entry для m уравнений и n переменных.

        Args:
            m (int): Количество уравнений (строк матрицы).
            n (int): Количество неизвестных (столбцов матрицы коэффициентов).
        """
        self.current_m = m
        self.current_n = n

        self.m_entry.delete(0, tk.END)
        self.m_entry.insert(0, str(m))
        self.n_entry.delete(0, tk.END)
        self.n_entry.insert(0, str(n))

        # Очищаем старые поля ввода
        for widget in self.matrix_grid_frame.winfo_children():
            widget.destroy()

        self.entries_A = []
        self.entries_B = []

        # Заголовки столбцов переменных x1 ... xn
        ttk.Label(self.matrix_grid_frame, text="", width=6).grid(row=0, column=0, padx=2, pady=3)
        for j in range(n):
            lbl = ttk.Label(
                self.matrix_grid_frame,
                text=f"x{j+1}",
                font=("Arial", 9, "bold"),
                anchor="center"
            )
            lbl.grid(row=0, column=j + 1, padx=3, pady=3)

        # Разделитель равенства
        eq_header = ttk.Label(
            self.matrix_grid_frame,
            text="=",
            font=("Arial", 10, "bold"),
            anchor="center"
        )
        eq_header.grid(row=0, column=n + 1, padx=4, pady=3)

        # Заголовок для свободного члена b
        b_header = ttk.Label(
            self.matrix_grid_frame,
            text="b (св. член)",
            font=("Arial", 9, "bold"),
            anchor="center"
        )
        b_header.grid(row=0, column=n + 2, padx=4, pady=3)

        # Создание строк сетки
        for i in range(m):
            row_label = ttk.Label(
                self.matrix_grid_frame,
                text=f"Ур. {i+1}:",
                font=("Arial", 9, "bold")
            )
            row_label.grid(row=i + 1, column=0, padx=3, pady=3, sticky=tk.E)

            row_entries: list[ttk.Entry] = []
            for j in range(n):
                ent = ttk.Entry(self.matrix_grid_frame, width=7, justify="center")
                ent.grid(row=i + 1, column=j + 1, padx=3, pady=3)
                row_entries.append(ent)
            self.entries_A.append(row_entries)

            eq_lbl = ttk.Label(self.matrix_grid_frame, text="=", font=("Arial", 9, "bold"))
            eq_lbl.grid(row=i + 1, column=n + 1, padx=4, pady=3)

            b_ent = ttk.Entry(self.matrix_grid_frame, width=8, justify="center")
            b_ent.grid(row=i + 1, column=n + 2, padx=4, pady=3)
            self.entries_B.append(b_ent)

        # Адаптация высоты холста под количество строк
        target_height = min(220, max(120, (m + 1) * 36 + 20))
        self.canvas.configure(height=target_height)

        # Обновление индикатора типа матрицы
        if m == n:
            self.status_label.config(text=f"Матрица {m}×{n} (квадратная)")
        else:
            self.status_label.config(text=f"Матрица {m}×{n} (прямоугольная, только Гаусс)")

    def fill_matrix_values(self, A: list[list[float]], B: list[float]) -> None:
        """
        Заполняет поля сетки значениями матрицы A и вектора B.

        Args:
            A (list[list[float]]): Матрица коэффициентов.
            B (list[float]): Вектор свободных членов.
        """
        m = len(A)
        n = len(A[0])
        self.set_system_dimensions(m, n)

        for i in range(m):
            for j in range(n):
                self.entries_A[i][j].delete(0, tk.END)
                val_a = A[i][j]
                self.entries_A[i][j].insert(0, str(int(val_a) if val_a == int(val_a) else val_a))

            self.entries_B[i].delete(0, tk.END)
            val_b = B[i]
            self.entries_B[i].insert(0, str(int(val_b) if val_b == int(val_b) else val_b))

    def clear_fields(self) -> None:
        """
        Очищает содержимое всех ячеек ввода матрицы A и вектора B.
        """
        for row in self.entries_A:
            for ent in row:
                ent.delete(0, tk.END)
        for ent in self.entries_B:
            ent.delete(0, tk.END)

    def load_preset_3x3(self) -> None:
        """
        Загружает типовую совместную определенную систему 3x3 с единственным решением.
        Уравнения:
          2*x1 + 1*x2 + 1*x3 = 7
          1*x1 - 1*x2 + 1*x3 = 2
          1*x1 + 2*x2 - 1*x3 = 2
        Решение: x1=1, x2=2, x3=3.
        """
        A = [
            [2.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 2.0, -1.0],
        ]
        B = [7.0, 2.0, 2.0]
        self.fill_matrix_values(A, B)

    def load_preset_2x3(self) -> None:
        """
        Загружает прямоугольную систему 2x3 (2 уравнения, 3 неизвестных).
        Имеет бесконечно много решений.
        """
        A = [
            [1.0, 2.0, 3.0],
            [2.0, -1.0, 1.0],
        ]
        B = [9.0, 8.0]
        self.fill_matrix_values(A, B)

    def load_preset_det0(self) -> None:
        """
        Загружает квадратную систему 3x3 с вырожденной матрицей (det = 0).
        Крамер и матричный метод для нее неприменимы.
        """
        A = [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]
        B = [4.0, 8.0, 12.0]
        self.fill_matrix_values(A, B)

    def get_input_matrices(self) -> tuple[list[list[float]], list[float]]:
        """
        Считывает и валидирует числа из всех полей ввода сетки.

        Поддерживает десятичные дроби через точку и через запятую.

        Returns:
            tuple[list[list[float]], list[float]]: Матрица A и вектор B.

        Raises:
            ValueError: Если какое-либо поле пустое или содержит нечисловые символы.
        """
        A: list[list[float]] = []
        B: list[float] = []

        for i, row in enumerate(self.entries_A):
            row_values: list[float] = []
            for j, ent in enumerate(row):
                raw_val = ent.get().strip().replace(",", ".")
                if not raw_val:
                    raise ValueError(f"Поле коэффициента в строке {i+1}, столбце {j+1} пусто.")
                try:
                    num = float(raw_val)
                    row_values.append(num)
                except ValueError:
                    raise ValueError(
                        f"Некорректное число в строке {i+1}, столбце {j+1}: '{raw_val}'"
                    )
            A.append(row_values)

        for i, ent in enumerate(self.entries_B):
            raw_val = ent.get().strip().replace(",", ".")
            if not raw_val:
                raise ValueError(f"Поле свободного члена (b) в строке {i+1} пусто.")
            try:
                num = float(raw_val)
                B.append(num)
            except ValueError:
                raise ValueError(
                    f"Некорректное число свободного члена в строке {i+1}: '{raw_val}'"
                )

        return A, B

    def _set_tab_content(self, text_widget: tk.Text, lines: list[str]) -> None:
        """
        Обновляет текст в текстовом поле вкладки, временно снимая защиту от редактирования.

        Args:
            text_widget (tk.Text): Целевой виджет текстового поля.
            lines (list[str]): Список строк для отображения.
        """
        text_widget.configure(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, "\n".join(lines))
        text_widget.configure(state=tk.DISABLED)

    def solve_system(self) -> None:
        """
        Основной обработчик: считывает матрицу, запускает комплексный расчет
        и распределяет пошаговые протоколы решений по соответствующим вкладкам.
        """
        try:
            A, B = self.get_input_matrices()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода данных", str(e))
            return

        try:
            results = solve_slae_all(A, B)
        except Exception as e:
            messagebox.showerror("Ошибка вычислений", f"Произошла ошибка при вычислении:\n{e}")
            return

        # Заполняем вкладки подробными отчетами
        self._set_tab_content(self.summary_text, results["summary"])
        self._set_tab_content(self.gauss_text, results["gauss"]["steps"])
        self._set_tab_content(self.cramer_text, results["cramer"]["steps"])
        self._set_tab_content(self.matrix_text, results["matrix"]["steps"])

        # Автоматически переключаем фокус на вкладку «Сводка»
        self.notebook.select(0)
