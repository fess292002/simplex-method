import numpy as np
from tkinter import *
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText


class SimplexSolver:
    def __init__(self):
        self.root = Tk()
        self.root.title("Решение задач линейного программирования")
        self.root.geometry("1000x750")

        self.style = ttk.Style()
        self.style.configure('TFrame', padding=5)
        self.style.configure('TButton', padding=5)
        self.style.configure('TLabel', padding=5)

        self.create_widgets()
        self.create_menu()

    def create_menu(self):
        menubar = Menu(self.root)

        # Меню "Файл"
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

        # Меню "Справка"
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="О методе", command=self.show_method_info)
        help_menu.add_command(label="Помощь", command=self.show_help)
        menubar.add_cascade(label="Справка", menu=help_menu)

        self.root.config(menu=menubar)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Фрейм параметров
        input_frame = ttk.LabelFrame(main_frame, text="Параметры задачи", padding=10)
        input_frame.pack(fill=X, padx=5, pady=5)

        # Поля ввода
        ttk.Label(input_frame, text="Кол-во переменных:").grid(row=0, column=0, sticky=E)
        self.var_count = ttk.Entry(input_frame, width=5)
        self.var_count.grid(row=0, column=1, sticky=W)
        self.var_count.insert(0, "2")

        ttk.Label(input_frame, text="Кол-во ограничений:").grid(row=1, column=0, sticky=E)
        self.constr_count = ttk.Entry(input_frame, width=5)
        self.constr_count.grid(row=1, column=1, sticky=W)
        self.constr_count.insert(0, "2")

        # Тип задачи
        ttk.Label(input_frame, text="Тип задачи:").grid(row=2, column=0, sticky=E)
        self.problem_type = StringVar(value="max")
        ttk.Radiobutton(input_frame, text="Максимизация", variable=self.problem_type, value="max").grid(row=2, column=1,
                                                                                                        sticky=W)
        ttk.Radiobutton(input_frame, text="Минимизация", variable=self.problem_type, value="min").grid(row=2, column=2,
                                                                                                       sticky=W)

        # Кнопка создания таблицы
        ttk.Button(input_frame, text="Создать таблицу", command=self.generate_input_table).grid(row=3, column=0,
                                                                                                columnspan=3, pady=10)

        # Область ввода данных с прокруткой
        table_container = ttk.Frame(main_frame)
        table_container.pack(fill=BOTH, expand=True)

        self.canvas = Canvas(table_container, borderwidth=0)
        scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.matrix_frame = ttk.LabelFrame(self.scrollable_frame, text="Ввод данных", padding=10)
        self.matrix_frame.pack(fill=BOTH, expand=True)

        # Фрейм результатов
        result_frame = ttk.LabelFrame(main_frame, text="Результаты", padding=10)
        result_frame.pack(fill=X, padx=5, pady=5)

        # Кнопки управления
        button_frame = ttk.Frame(result_frame)
        button_frame.pack(fill=X, pady=5)

        ttk.Button(button_frame, text="Решить", command=self.solve_problem).pack(side=LEFT, padx=5)
        ttk.Button(button_frame, text="Очистить", command=self.clear_all).pack(side=LEFT, padx=5)
        ttk.Button(button_frame, text="О методе", command=self.show_method_info).pack(side=RIGHT, padx=5)

        # Вывод результатов
        self.result_text = ScrolledText(result_frame, height=10, font=('Arial', 10), wrap=WORD)
        self.result_text.pack(fill=BOTH, expand=True)

    def show_method_info(self):
        info_window = Toplevel(self.root)
        info_window.title("Теория симплекс-метода")
        info_window.geometry("900x700")

        notebook = ttk.Notebook(info_window)
        notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Вкладка теории
        theory_frame = ttk.Frame(notebook)
        notebook.add(theory_frame, text="Основы метода")

        theory_text = ScrolledText(theory_frame, wrap=WORD, font=('Arial', 11))
        theory_text.pack(fill=BOTH, expand=True, padx=5, pady=5)

        theory_content = """СИМПЛЕКС-МЕТОД (Джордж Данциг, 1947)

Алгоритм для решения задач линейного программирования вида:

Максимизировать: Z = c₁x₁ + c₂x₂ + ... + cₙxₙ
При условиях:
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ ≤ b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ ≤ b₂
...
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ ≤ bₘ
xᵢ ≥ 0 для всех i

Основные этапы:
1. Приведение к стандартной форме
2. Построение начальной симплекс-таблицы
3. Проверка оптимальности
4. Выбор разрешающих элементов
5. Пересчет таблицы
6. Повторение до достижения оптимума"""
        theory_text.insert(END, theory_content)
        theory_text.config(state=DISABLED)

        # Вкладка формул
        formulas_frame = ttk.Frame(notebook)
        notebook.add(formulas_frame, text="Формулы")

        formulas_text = ScrolledText(formulas_frame, wrap=WORD, font=('Courier', 11))
        formulas_text.pack(fill=BOTH, expand=True, padx=5, pady=5)

        formulas_content = """ОСНОВНЫЕ ФОРМУЛЫ:

1. Приведение к стандартной форме:
   Максимизация: F → max ≡ -F → min
   Ограничения типа ≥: умножить на -1

2. Добавление slack-переменных:
   a₁₁x₁ + ... + a₁ₙxₙ + s₁ = b₁
   ...
   aₘ₁x₁ + ... + aₘₙxₙ + sₘ = bₘ

3. Критерий оптимальности:
   Для min: все Δⱼ ≥ 0
   Для max: все Δⱼ ≤ 0
   где Δⱼ = cⱼ - Σ(cᵢ·aᵢⱼ)

4. Пересчет таблицы:
   aᵢⱼ(new) = aᵢⱼ - (aᵢk·aᵣⱼ)/aᵣk
   bᵢ(new) = bᵢ - (aᵢk·bᵣ)/aᵣk"""
        formulas_text.insert(END, formulas_content)
        formulas_text.config(state=DISABLED)

        # Вкладка об основателях
        founders_frame = ttk.Frame(notebook)
        notebook.add(founders_frame, text="История")

        founders_text = ScrolledText(founders_frame, wrap=WORD, font=('Arial', 11))
        founders_text.pack(fill=BOTH, expand=True, padx=5, pady=5)

        founders_content = """ОСНОВАТЕЛИ:

Джордж Данциг (1914-2005)
- Разработал симплекс-метод в 1947
- Работал в RAND Corporation
- "Отец" линейного программирования

Леонид Канторович (1912-1986)
- Советский математик
- Нобелевская премия по экономике (1975)
- Разработал теорию оптимального распределения ресурсов

Джон фон Нейман (1903-1957)
- Теория двойственности
- Внес фундаментальный вклад в методы оптимизации

Применение:
- Оптимальное планирование
- Логистика и транспорт
- Управление производством
- Финансовое планирование"""
        founders_text.insert(END, founders_content)
        founders_text.config(state=DISABLED)

        ttk.Button(info_window, text="Закрыть", command=info_window.destroy).pack(pady=10)

    def show_help(self):
        help_window = Toplevel(self.root)
        help_window.title("Помощь")
        help_window.geometry("600x400")

        help_text = ScrolledText(help_window, wrap=WORD, font=('Arial', 11))
        help_text.pack(fill=BOTH, expand=True, padx=10, pady=10)

        help_content = """ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ:

1. Введите количество переменных и ограничений
2. Нажмите "Создать таблицу"
3. Заполните коэффициенты целевой функции
4. Заполните коэффициенты ограничений
5. Выберите тип неравенств (≤, =, ≥)
6. Нажмите "Решить"
7. Просмотрите результаты

Для подробной информации о методе нажмите "О методе\""""
        help_text.insert(END, help_content)
        help_text.config(state=DISABLED)

        ttk.Button(help_window, text="Закрыть", command=help_window.destroy).pack(pady=10)

    def generate_input_table(self):
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        try:
            num_vars = int(self.var_count.get())
            num_constr = int(self.constr_count.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Введите целые числа")
            return

        if num_vars <= 0 or num_constr <= 0:
            messagebox.showerror("Ошибка", "Значения должны быть > 0")
            return

        # Целевая функция
        ttk.Label(self.matrix_frame, text="Целевая функция:").grid(row=0, column=0, sticky=E)
        self.obj_coeffs = []
        for i in range(num_vars):
            entry = ttk.Entry(self.matrix_frame, width=8)
            entry.grid(row=0, column=i + 1, padx=2, pady=2)
            self.obj_coeffs.append(entry)
            ttk.Label(self.matrix_frame, text=f"x{i + 1}").grid(row=1, column=i + 1)

        # Ограничения
        ttk.Label(self.matrix_frame, text="Ограничения:").grid(row=2, column=0, sticky=NE)

        self.constr_coeffs = []
        self.constr_rhs = []
        self.constr_types = []

        for i in range(num_constr):
            # Коэффициенты
            row_entries = []
            for j in range(num_vars):
                entry = ttk.Entry(self.matrix_frame, width=8)
                entry.grid(row=i + 3, column=j + 1, padx=2, pady=2)
                row_entries.append(entry)
            self.constr_coeffs.append(row_entries)

            # Тип неравенства
            constr_type = StringVar(value="<=")
            self.constr_types.append(constr_type)

            frame = ttk.Frame(self.matrix_frame)
            frame.grid(row=i + 3, column=num_vars + 1, padx=5)

            ttk.Radiobutton(frame, text="≤", variable=constr_type, value="<=").pack(side=LEFT)
            ttk.Radiobutton(frame, text="=", variable=constr_type, value="=").pack(side=LEFT)
            ttk.Radiobutton(frame, text="≥", variable=constr_type, value=">=").pack(side=LEFT)

            # Правая часть
            rhs_entry = ttk.Entry(self.matrix_frame, width=8)
            rhs_entry.grid(row=i + 3, column=num_vars + 2, padx=5, pady=2)
            self.constr_rhs.append(rhs_entry)

            ttk.Label(self.matrix_frame, text="→").grid(row=i + 3, column=num_vars + 3, padx=5)

    def get_input_data(self):
        try:
            # Целевая функция
            c = [float(entry.get()) for entry in self.obj_coeffs]

            # Ограничения
            A = []
            b = []
            for i in range(len(self.constr_coeffs)):
                row = [float(entry.get()) for entry in self.constr_coeffs[i]]
                A.append(row)
                b.append(float(self.constr_rhs[i].get()))

                if self.constr_types[i].get() == ">=":
                    A[i] = [-x for x in A[i]]
                    b[i] = -b[i]

            return np.array(c), np.array(A), np.array(b)

        except ValueError:
            messagebox.showerror("Ошибка", "Введите числа во все поля")
            return None, None, None

    def solve_problem(self):
        c, A, b = self.get_input_data()
        if c is None:
            return

        if self.problem_type.get() == "max":
            c = -c

        num_slack = len(b)
        c_full = np.concatenate([c, np.zeros(num_slack)])
        A_full = np.hstack([A, np.eye(num_slack)])
        basis = list(range(len(c), len(c) + num_slack))

        solution, value, basis = self.simplex_method(c_full, A_full, b, basis)
        self.display_results(solution, value, basis, len(c))

    def simplex_method(self, c, A, b, basis):
        m, n = A.shape

        while True:
            non_basis = [j for j in range(n) if j not in basis]
            B = A[:, basis]
            N = A[:, non_basis]

            try:
                B_inv = np.linalg.inv(B)
                x_B = B_inv @ b
            except np.linalg.LinAlgError:
                return None, None, None

            c_B = c[basis]
            c_N = c[non_basis]
            lambda_ = np.linalg.solve(B.T, c_B)
            r_N = c_N - (N.T @ lambda_)

            if np.all(r_N >= 0):
                x = np.zeros(n)
                x[basis] = x_B
                return x, c @ x, basis

            entering = non_basis[np.argmin(r_N)]
            d_B = -B_inv @ A[:, entering]

            if np.all(d_B >= 0):
                return None, np.inf, None

            ratios = [-x_B[i] / d_B[i] if d_B[i] < 0 else np.inf for i in range(m)]
            leaving_idx = np.argmin(ratios)
            basis[leaving_idx] = entering

    def display_results(self, solution, value, basis, num_vars):
        self.result_text.config(state=NORMAL)
        self.result_text.delete(1.0, END)

        if solution is None:
            self.result_text.insert(END, "Задача не имеет решения или неограничена")
            self.result_text.config(state=DISABLED)
            return

        if self.problem_type.get() == "max":
            value = -value

        self.result_text.insert(END, f"Оптимальное значение: {value:.4f}\n\n")
        self.result_text.insert(END, "Значения переменных:\n")

        for i in range(num_vars):
            self.result_text.insert(END, f"x{i + 1} = {solution[i]:.4f}\n")

        self.result_text.insert(END, "\nДоп. переменные:\n")
        for i in range(len(solution) - num_vars):
            self.result_text.insert(END, f"s{i + 1} = {solution[num_vars + i]:.4f}\n")

        if basis:
            self.result_text.insert(END, "\nБазисные переменные:\n")
            self.result_text.insert(END, " ".join(
                f"x{var + 1}" if var < num_vars else f"s{var - num_vars + 1}"
                for var in basis))

        self.result_text.config(state=DISABLED)

    def clear_all(self):
        self.var_count.delete(0, END)
        self.var_count.insert(0, "2")
        self.constr_count.delete(0, END)
        self.constr_count.insert(0, "2")
        self.problem_type.set("max")

        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        self.result_text.config(state=NORMAL)
        self.result_text.delete(1.0, END)
        self.result_text.config(state=DISABLED)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    try:
        import numpy as np
    except ImportError:
        print("Установите numpy: pip install numpy")
        exit(1)

    app = SimplexSolver()
    app.run()