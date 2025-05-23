import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import json
import re
from fractions import Fraction
from datetime import datetime
from tkinter import *
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LinearProgrammingSolver:
    def __init__(self):
        self.root = Tk()
        self.root.title("Симплекс-метод: решение задач ЛП")
        self.root.geometry("1200x800")

        self.solutions_history = []
        self.current_solution = None
        self.plot_figure = None

        self.style = ttk.Style()
        self.style.configure('TFrame', padding=5)
        self.style.configure('TButton', padding=5)
        self.style.configure('TLabel', padding=5)

        self.create_widgets()
        self.create_menu()

    def create_menu(self):
        menubar = Menu(self.root)

        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Сохранить решение", command=self.save_solution)
        file_menu.add_command(label="Загрузить решение", command=self.load_solution)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="Инструкция", command=self.show_instructions)
        help_menu.add_command(label="Теория метода", command=self.show_method_theory)
        help_menu.add_command(label="О программе", command=self.show_about)
        menubar.add_cascade(label="Справка", menu=help_menu)

        self.root.config(menu=menubar)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        input_frame = ttk.LabelFrame(main_frame, text="Ввод задачи", width=400)
        input_frame.pack(side=LEFT, fill=Y, padx=5, pady=5)

        result_frame = ttk.LabelFrame(main_frame, text="Результаты")
        result_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=5, pady=5)

        # Поле ввода целевой функции
        ttk.Label(input_frame, text="Целевая функция:").pack(anchor=W, pady=(5, 0))
        self.objective_entry = ttk.Entry(input_frame)
        self.objective_entry.pack(fill=X, padx=5, pady=5)
        ttk.Label(input_frame, text="Пример: 3x1 + 5x2 или 2*x1 - 4*x2").pack(anchor=W)

        # Выбор типа оптимизации
        ttk.Label(input_frame, text="Тип оптимизации:").pack(anchor=W, pady=(5, 0))
        self.problem_type = StringVar(value="max")
        ttk.Radiobutton(input_frame, text="Максимизация", variable=self.problem_type, value="max").pack(anchor=W)
        ttk.Radiobutton(input_frame, text="Минимизация", variable=self.problem_type, value="min").pack(anchor=W)

        # Поле ввода ограничений
        ttk.Label(input_frame, text="Ограничения (по одному в строке):").pack(anchor=W, pady=(10, 0))
        self.constraints_text = ScrolledText(input_frame, height=8, width=40)
        self.constraints_text.pack(fill=BOTH, padx=5, pady=5)
        ttk.Label(input_frame, text="Примеры:\n2x1 + 3x2 <= 10\nx1 - x2 >= 2\nx1 + x2 = 5").pack(anchor=W)

        # Кнопки управления
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=X, pady=10)
        ttk.Button(button_frame, text="Решить", command=self.solve_problem).pack(side=LEFT, padx=5)
        ttk.Button(button_frame, text="Очистить", command=self.clear_input).pack(side=LEFT, padx=5)
        ttk.Button(button_frame, text="График", command=self.plot_solution).pack(side=RIGHT, padx=5)

        # Область для графика
        self.plot_frame = ttk.Frame(result_frame)
        self.plot_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # Область вывода результатов
        self.result_text = ScrolledText(result_frame, height=10, wrap=WORD)
        self.result_text.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # История решений
        ttk.Label(result_frame, text="История решений:").pack(anchor=W)
        self.history_listbox = Listbox(result_frame, height=5)
        self.history_listbox.pack(fill=BOTH, padx=5, pady=5)
        self.history_listbox.bind('<<ListboxSelect>>', self.show_selected_solution)

    def show_instructions(self):
        """Показывает инструкцию по использованию"""
        instructions = """Инструкция по использованию:

1. Введите целевую функцию в формате:
   Пример: 3x1 + 5x2  или  2*x1 - 4*x2

2. Выберите тип оптимизации (максимизация/минимизация)

3. Введите ограничения (по одному в строке):
   Примеры:
   2x1 + 3x2 <= 10
   x1 - x2 >= 2
   x1 + x2 = 5

4. Нажмите "Решить" для получения решения
5. Для 2D задач можно построить график"""
        messagebox.showinfo("Инструкция", instructions)

    def show_method_theory(self):
        """Показывает теорию симплекс-метода"""
        theory = """СИМПЛЕКС-МЕТОД

Основная идея:
Метод последовательно перебирает вершины многогранника решений, 
двигаясь в направлении улучшения значения целевой функции.

Основные этапы:
1. Приведение задачи к каноническому виду
2. Построение начального опорного решения
3. Проверка оптимальности текущего решения
4. Переход к соседней вершине многогранника
5. Повторение до нахождения оптимального решения

Основатели метода:
- Джордж Данциг (1947 г.) - разработал симплекс-метод
- Леонид Канторович - основатель линейного программирования
- Джон фон Нейман - теория двойственности

Преимущества:
- Эффективен на практике для большинства задач
- Гарантированно находит глобальный оптимум
- Позволяет анализировать чувствительность решения"""

        info_window = Toplevel(self.root)
        info_window.title("Теория симплекс-метода")
        info_window.geometry("700x500")

        text = ScrolledText(info_window, wrap=WORD, font=('Arial', 11))
        text.pack(fill=BOTH, expand=True, padx=10, pady=10)
        text.insert(END, theory)
        text.config(state=DISABLED)

        ttk.Button(info_window, text="Закрыть", command=info_window.destroy).pack(pady=10)

    def show_about(self):
        """Показывает информацию о программе"""
        about = """Программа для решения задач линейного программирования

Версия 1.0

Используемые технологии:
- Python 3
- Библиотеки: NumPy, SciPy, Matplotlib
- Графический интерфейс: Tkinter

Разработчик: Штоколов Егор Владимирович"""
        messagebox.showinfo("О программе", about)

    def parse_linear_expression(self, expr, var_count):
        """Парсит линейное выражение и возвращает коэффициенты"""
        expr = expr.replace(' ', '').replace('*', '')
        coeffs = [0.0] * var_count

        for i in range(1, var_count + 1):
            var = f'x{i}'
            if var in expr:
                pos = expr.find(var)
                coef_part = expr[:pos]

                # Определяем коэффициент
                if not coef_part or coef_part[-1] in '+-':
                    coef = 1.0
                else:
                    coef_str = ''
                    for c in reversed(coef_part):
                        if c.isdigit() or c in '.+-':
                            coef_str = c + coef_str
                        else:
                            break
                    try:
                        coef = float(coef_str) if coef_str else 1.0
                    except:
                        coef = 1.0

                # Учитываем знак
                if '-' in coef_part:
                    sign = -1 if coef_part.count('-') % 2 == 1 else 1
                    coef *= sign

                coeffs[i - 1] = coef
                expr = expr[pos + len(var):]

        return coeffs

    def solve_problem(self):
        """Основной метод решения задачи"""
        try:
            objective = self.objective_entry.get()
            if not objective:
                messagebox.showerror("Ошибка", "Введите целевую функцию")
                return

            constraints_text = self.constraints_text.get("1.0", END)

            # Определяем количество переменных
            var_matches = re.findall(r'x(\d+)', objective + constraints_text)
            var_count = max([int(m) for m in var_matches] + [0])

            if var_count == 0:
                messagebox.showerror("Ошибка", "Не найдены переменные в задаче")
                return
            elif var_count > 2:
                messagebox.showwarning("Предупреждение",
                                       "Графическое отображение доступно только для 2 переменных")

            # Парсим целевую функцию
            c = self.parse_linear_expression(objective, var_count)
            if self.problem_type.get() == "max":
                c = [-x for x in c]  # Для максимизации инвертируем

            # Парсим ограничения
            A_ub = []
            b_ub = []
            A_eq = []
            b_eq = []

            for line in constraints_text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if '<=' in line:
                    left, right = line.split('<=')
                    coeffs = self.parse_linear_expression(left.strip(), var_count)
                    try:
                        val = float(right.strip())
                        A_ub.append(coeffs)
                        b_ub.append(val)
                    except:
                        continue

                elif '>=' in line:
                    left, right = line.split('>=')
                    coeffs = self.parse_linear_expression(left.strip(), var_count)
                    try:
                        val = float(right.strip())
                        A_ub.append([-x for x in coeffs])  # Инвертируем для >=
                        b_ub.append(-val)
                    except:
                        continue

                elif '=' in line:
                    left, right = line.split('=')
                    coeffs = self.parse_linear_expression(left.strip(), var_count)
                    try:
                        val = float(right.strip())
                        A_eq.append(coeffs)
                        b_eq.append(val)
                    except:
                        continue

            # Решаем задачу
            bounds = [(0, None)] * var_count  # x >= 0
            result = linprog(c, A_ub=A_ub if A_ub else None,
                             b_ub=b_ub if b_ub else None,
                             A_eq=A_eq if A_eq else None,
                             b_eq=b_eq if b_eq else None,
                             bounds=bounds, method='highs')

            # Формируем решение
            solution = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'objective': objective,
                'constraints': constraints_text,
                'solution': result.x.tolist() if result.success else None,
                'optimal_value': -result.fun if self.problem_type.get() == "max" and result.success else result.fun if result.success else None,
                'success': result.success,
                'message': result.message if hasattr(result, 'message') else 'Решение найдено',
                'var_count': var_count
            }

            self.current_solution = solution
            self.solutions_history.append(solution)
            self.update_history_list()
            self.display_results(solution)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при решении задачи: {str(e)}")

    def display_results(self, solution):
        """Отображает результаты решения"""
        self.result_text.config(state=NORMAL)
        self.result_text.delete(1.0, END)

        if solution['success']:
            self.result_text.insert(END, "════════ РЕШЕНИЕ НАЙДЕНО ════════\n\n")

            # Отображаем оптимальное значение
            if solution['optimal_value'] is not None:
                opt_value = solution['optimal_value']
                opt_value_frac = Fraction(opt_value).limit_denominator()
                self.result_text.insert(END,
                                        f"Оптимальное значение: {opt_value:.6f} ({opt_value_frac})\n\n")

            # Отображаем значения переменных
            if solution['solution'] is not None:
                self.result_text.insert(END, "Значения переменных:\n")
                for i, val in enumerate(solution['solution']):
                    val_frac = Fraction(val).limit_denominator()
                    self.result_text.insert(END, f"x{i + 1} = {val:.6f} ({val_frac})\n")

                # Для 2D задач показываем дополнительные сведения
                if solution['var_count'] == 2:
                    x1, x2 = solution['solution']
                    self.result_text.insert(END, f"\nТочка решения: ({x1:.4f}, {x2:.4f})\n")
        else:
            self.result_text.insert(END, "════════ РЕШЕНИЕ НЕ НАЙДЕНО ════════\n\n")
            self.result_text.insert(END, f"Причина: {solution['message']}")

        self.result_text.config(state=DISABLED)

    def plot_solution(self):
        """Строит график для 2D задачи"""
        if not self.current_solution or not self.current_solution['success']:
            messagebox.showwarning("Предупреждение", "Нет решения для построения графика")
            return

        if self.current_solution['var_count'] != 2:
            messagebox.showwarning("Предупреждение",
                                   "График можно построить только для задач с 2 переменными")
            return

        try:
            # Очищаем предыдущий график
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            fig, ax = plt.subplots(figsize=(8, 6))

            # Получаем решение
            x_sol, y_sol = self.current_solution['solution']

            # Создаем сетку для построения
            x = np.linspace(0, max(10, x_sol * 1.5), 100)
            y = np.linspace(0, max(10, y_sol * 1.5), 100)
            X, Y = np.meshgrid(x, y)

            # Вычисляем значения целевой функции
            Z = np.zeros_like(X)
            obj_expr = self.current_solution['objective'].replace('x1', 'X').replace('x2', 'Y')
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        Z[i, j] = eval(obj_expr, {'X': X[i, j], 'Y': Y[i, j]})
                    except:
                        Z[i, j] = np.nan

            # Строим контуры целевой функции
            cs = ax.contour(X, Y, Z, levels=20)
            ax.clabel(cs, inline=True, fontsize=10)

            # Отмечаем оптимальное решение
            ax.plot(x_sol, y_sol, 'ro', markersize=10)
            ax.annotate(f'Оптимум\n({x_sol:.2f}, {y_sol:.2f})',
                        xy=(x_sol, y_sol), xytext=(x_sol + 1, y_sol + 1),
                        arrowprops=dict(facecolor='black', shrink=0.05))

            # Добавляем ограничения
            for line in self.current_solution['constraints'].split('\n'):
                line = line.strip()
                if not line or '=' in line:
                    continue

                if '<=' in line:
                    left = line.split('<=')[0].strip()
                    try:
                        if 'x1' in left and 'x2' in left:
                            # Ограничение вида a*x1 + b*x2 <= c
                            a, b = self.parse_linear_expression(left, 2)
                            if a != 0 and b != 0:
                                const_val = float(line.split('<=')[1].strip())
                                y_const = (const_val - a * x) / b
                                ax.plot(x, y_const, 'r--', alpha=0.7,
                                        label=f'{a}x1 + {b}x2 <= {const_val}')
                    except:
                        continue

            # Настраиваем график
            ax.set_xlabel('x1', fontsize=12)
            ax.set_ylabel('x2', fontsize=12)
            ax.set_title('Графическое решение задачи ЛП', fontsize=14)
            ax.grid(True)
            ax.legend()

            # Встраиваем график в интерфейс
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)

            self.plot_figure = fig

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении графика: {str(e)}")

    def update_history_list(self):
        """Обновляет список истории решений"""
        self.history_listbox.delete(0, END)
        for i, sol in enumerate(self.solutions_history):
            if sol['success'] and sol['optimal_value'] is not None:
                opt_value = sol['optimal_value']
                self.history_listbox.insert(END,
                                            f"{i + 1}. {sol['timestamp']} - F = {opt_value:.2f}")
            else:
                self.history_listbox.insert(END,
                                            f"{i + 1}. {sol['timestamp']} - Ошибка")

    def show_selected_solution(self, event):
        """Показывает выбранное решение из истории"""
        selection = self.history_listbox.curselection()
        if selection:
            self.current_solution = self.solutions_history[selection[0]]
            self.display_results(self.current_solution)

    def save_solution(self):
        """Сохраняет текущее решение в файл"""
        if not self.current_solution:
            messagebox.showwarning("Предупреждение", "Нет решения для сохранения")
            return

        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Сохранить решение"
            )

            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(self.current_solution, f, indent=4, ensure_ascii=False)
                messagebox.showinfo("Успех", "Решение успешно сохранено")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении: {str(e)}")

    def load_solution(self):
        """Загружает решение из файла"""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Загрузить решение"
            )

            if filepath:
                with open(filepath, 'r') as f:
                    solution = json.load(f)

                self.current_solution = solution
                self.solutions_history.append(solution)
                self.update_history_list()
                self.display_results(solution)
                messagebox.showinfo("Успех", "Решение успешно загружено")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке: {str(e)}")

    def clear_input(self):
        """Очищает все поля ввода"""
        self.objective_entry.delete(0, END)
        self.constraints_text.delete("1.0", END)
        self.problem_type.set("max")

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        self.result_text.config(state=NORMAL)
        self.result_text.delete(1.0, END)
        self.result_text.config(state=DISABLED)

    def run(self):
        """Запускает главный цикл программы"""
        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = LinearProgrammingSolver()
        app.run()
    except ImportError as e:
        print(f"Необходимо установить зависимости: {e}")
        print("Попробуйте: pip install numpy scipy matplotlib")