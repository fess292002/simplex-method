import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
from fractions import Fraction
from datetime import datetime
from tkinter import *
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class NonlinearSolver:
    def __init__(self):
        self.root = Tk()
        self.root.title("Решение задач нелинейного программирования")
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
        help_menu.add_command(label="О методе", command=self.show_method_info)
        help_menu.add_command(label="История и основатели", command=self.show_history_info)
        menubar.add_cascade(label="Справка", menu=help_menu)

        self.root.config(menu=menubar)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        input_frame = ttk.LabelFrame(main_frame, text="Ввод задачи", width=400)
        input_frame.pack(side=LEFT, fill=Y, padx=5, pady=5)

        result_frame = ttk.LabelFrame(main_frame, text="Результаты")
        result_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=5, pady=5)

        ttk.Label(input_frame, text="Целевая функция (используйте x[0], x[1], ...):").pack(anchor=W, pady=(5, 0))
        self.objective_entry = ttk.Entry(input_frame)
        self.objective_entry.pack(fill=X, padx=5, pady=5)

        ttk.Label(input_frame, text="Тип оптимизации:").pack(anchor=W, pady=(5, 0))
        self.problem_type = StringVar(value="min")
        ttk.Radiobutton(input_frame, text="Минимизация", variable=self.problem_type, value="min").pack(anchor=W)
        ttk.Radiobutton(input_frame, text="Максимизация", variable=self.problem_type, value="max").pack(anchor=W)

        ttk.Label(input_frame, text="Начальное приближение (через запятую):").pack(anchor=W, pady=(10, 0))
        self.initial_guess_entry = ttk.Entry(input_frame)
        self.initial_guess_entry.pack(fill=X, padx=5, pady=5)

        ttk.Label(input_frame, text="Ограничения (по одному в строке):").pack(anchor=W, pady=(10, 0))
        self.constraints_text = Text(input_frame, height=8, width=40)
        self.constraints_text.pack(fill=BOTH, padx=5, pady=5)

        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=X, pady=10)

        ttk.Button(button_frame, text="Решить", command=self.solve_problem).pack(side=LEFT, padx=5)
        ttk.Button(button_frame, text="Очистить", command=self.clear_input).pack(side=LEFT, padx=5)
        ttk.Button(button_frame, text="Построить график", command=self.plot_solution).pack(side=RIGHT, padx=5)

        self.plot_frame = ttk.Frame(result_frame)
        self.plot_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)

        self.result_text = ScrolledText(result_frame, height=10, wrap=WORD)
        self.result_text.pack(fill=BOTH, expand=True, padx=5, pady=5)

        ttk.Label(result_frame, text="История решений:").pack(anchor=W)
        self.history_listbox = Listbox(result_frame, height=5)
        self.history_listbox.pack(fill=BOTH, padx=5, pady=5)
        self.history_listbox.bind('<<ListboxSelect>>', self.show_selected_solution)

    def float_to_fraction(self, num, max_denominator=1000):
        """Преобразует float в дробь с заданным максимальным знаменателем"""
        frac = Fraction(num).limit_denominator(max_denominator)
        if frac.denominator == 1:
            return f"{frac.numerator}"
        return f"{frac.numerator}/{frac.denominator}"

    def show_instructions(self):
        info_window = Toplevel(self.root)
        info_window.title("Инструкция по использованию")
        info_window.geometry("700x500")

        text = ScrolledText(info_window, wrap=WORD, font=('Arial', 11))
        text.pack(fill=BOTH, expand=True, padx=10, pady=10)

        content = """ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ ПРОГРАММЫ

1. Ввод задачи:
   - Введите целевую функцию в поле "Целевая функция"
     Пример: x[0]**2 + x[1]**2
   - Выберите тип оптимизации (минимизация/максимизация)
   - Укажите начальное приближение (через запятую)
     Пример: 1, 1
   - Введите ограничения (по одному в строке)
     Примеры:
     x[0] + x[1] <= 5
     x[0]**2 + x[1]**2 >= 1
     2*x[0] - x[1] = 3

2. Решение задачи:
   - Нажмите кнопку "Решить"
   - Результаты появятся в правой панели
   - Для визуализации нажмите "Построить график"
     (доступно только для 2D задач)

3. Работа с решениями:
   - Все решения сохраняются в истории
   - Можно сохранить решение в файл (меню Файл)
   - Можно загрузить ранее сохраненное решение

4. Примеры задач:
   - Минимизация: x[0]**2 + x[1]**2
     Ограничения: x[0] + x[1] >= 1
   - Максимизация: -x[0]**2 - x[1]**2 + 4*x[0] + 6*x[1]
     Ограничения: x[0] + x[1] <= 5
"""
        text.insert(END, content)
        text.config(state=DISABLED)

        ttk.Button(info_window, text="Закрыть", command=info_window.destroy).pack(pady=10)

    def show_method_info(self):
        info_window = Toplevel(self.root)
        info_window.title("О методе нелинейного программирования")
        info_window.geometry("700x500")

        text = ScrolledText(info_window, wrap=WORD, font=('Arial', 11))
        text.pack(fill=BOTH, expand=True, padx=10, pady=10)

        content = """МЕТОДЫ НЕЛИНЕЙНОЙ ОПТИМИЗАЦИИ

1. Постановка задачи:
   Найти экстремум функции:
   f(x₁, x₂, ..., xₙ) → min/max

   При ограничениях:
   gᵢ(x₁, x₂, ..., xₙ) ≤ 0, i = 1..m
   hⱼ(x₁, x₂, ..., xₙ) = 0, j = 1..k

2. Используемые методы:
   Программа использует алгоритм SLSQP (Sequential Least Squares Programming):
   - Комбинация метода последовательного квадратичного программирования
   - Метода наименьших квадратов
   - Эффективен для задач с ограничениями

3. Особенности нелинейной оптимизации:
   - Могут существовать локальные и глобальные экстремумы
   - Решение может зависеть от начального приближения
   - Ограничения могут быть нелинейными
   - Возможны случаи отсутствия решения

4. Применение:
   - Экономическое моделирование
   - Оптимальное управление
   - Машинное обучение
   - Инженерное проектирование
   - Финансовая математика
"""
        text.insert(END, content)
        text.config(state=DISABLED)

        ttk.Button(info_window, text="Закрыть", command=info_window.destroy).pack(pady=10)

    def show_history_info(self):
        info_window = Toplevel(self.root)
        info_window.title("История и основатели")
        info_window.geometry("700x500")

        text = ScrolledText(info_window, wrap=WORD, font=('Arial', 11))
        text.pack(fill=BOTH, expand=True, padx=10, pady=10)

        content = """ИСТОРИЯ РАЗВИТИЯ МЕТОДОВ ОПТИМИЗАЦИИ

1. Основные этапы развития:
   - 17 век: Первые методы оптимизации (Ферма, Ньютон)
   - 18 век: Метод множителей Лагранжа
   - 1940-е: Развитие линейного программирования (Данциг)
   - 1950-е: Методы нелинейной оптимизации
   - 1960-е: Развитие численных методов
   - 1980-е: Методы внутренней точки

2. Ключевые личности:

Джозеф-Луи Лагранж (1736-1813):
   - Разработал метод множителей Лагранжа
   - Основоположник вариационного исчисления

Джордж Данциг (1914-2005):
   - Разработал симплекс-метод
   - Отец линейного программирования

Леонид Канторович (1912-1986):
   - Развил теорию оптимального планирования
   - Нобелевская премия по экономике (1975)

Ричард Беллман (1920-1984):
   - Разработал динамическое программирование
   - Принцип оптимальности Беллмана

3. Современное состояние:
   - Широкое использование в ИИ и машинном обучении
   - Развитие методов выпуклой оптимизации
   - Применение в больших данных
   - Развитие стохастических методов оптимизации
"""
        text.insert(END, content)
        text.config(state=DISABLED)

        ttk.Button(info_window, text="Закрыть", command=info_window.destroy).pack(pady=10)

    def parse_constraints(self, constraints_text):
        constraints = []
        for line in constraints_text.split('\n'):
            line = line.strip()
            if line:
                try:
                    if '<=' in line:
                        expr = line.split('<=')[0].strip()
                        val = float(line.split('<=')[1].strip())
                        constraints.append({'type': 'ineq', 'fun': lambda x, e=expr: eval(e, {'x': x}) - val})
                    elif '>=' in line:
                        expr = line.split('>=')[0].strip()
                        val = float(line.split('>=')[1].strip())
                        constraints.append({'type': 'ineq', 'fun': lambda x, e=expr: val - eval(e, {'x': x})})
                    elif '=' in line:
                        expr = line.split('=')[0].strip()
                        val = float(line.split('=')[1].strip())
                        constraints.append({'type': 'eq', 'fun': lambda x, e=expr: eval(e, {'x': x}) - val})
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Ошибка в ограничении: {line}\n{str(e)}")
                    return None
        return constraints

    def solve_problem(self):
        try:
            objective = self.objective_entry.get()
            if not objective:
                messagebox.showerror("Ошибка", "Введите целевую функцию")
                return

            constraints_text = self.constraints_text.get("1.0", END)
            initial_guess_str = self.initial_guess_entry.get()

            if not initial_guess_str:
                messagebox.showerror("Ошибка", "Введите начальное приближение")
                return

            try:
                initial_guess = [float(x.strip()) for x in initial_guess_str.split(',')]
            except ValueError:
                messagebox.showerror("Ошибка", "Некорректное начальное приближение")
                return

            constraints = self.parse_constraints(constraints_text)
            if constraints is None:
                return

            try:
                if self.problem_type.get() == "max":
                    objective_func = lambda x: -eval(objective, {'x': x})
                else:
                    objective_func = lambda x: eval(objective, {'x': x})
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка в целевой функции: {str(e)}")
                return

            try:
                result = minimize(objective_func, initial_guess,
                                  constraints=constraints,
                                  method='SLSQP')
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при решении: {str(e)}")
                return

            solution = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'objective': objective,
                'constraints': constraints_text,
                'initial_guess': initial_guess,
                'solution': [float(x) for x in result.x],
                'optimal_value': float(result.fun),
                'success': bool(result.success),
                'message': str(result.message)
            }

            self.current_solution = solution
            self.solutions_history.append(solution)
            self.update_history_list()
            self.display_results(result)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Неожиданная ошибка: {str(e)}")

    def display_results(self, result):
        self.result_text.config(state=NORMAL)
        self.result_text.delete(1.0, END)

        if result.success:
            opt_value = -result.fun if self.problem_type.get() == "max" else result.fun
            opt_value_frac = self.float_to_fraction(opt_value)

            self.result_text.insert(END, f"Решение найдено успешно!\n\n")
            self.result_text.insert(END, f"Оптимальное значение: {opt_value:.6f} ({opt_value_frac})\n\n")
            self.result_text.insert(END, "Оптимальные параметры:\n")

            for i, val in enumerate(result.x):
                val_frac = self.float_to_fraction(val)
                self.result_text.insert(END, f"x{i + 1} = {val:.6f} ({val_frac})\n")
        else:
            self.result_text.insert(END, f"Решение не найдено: {result.message}")

        self.result_text.config(state=DISABLED)

    def plot_solution(self):
        if not self.current_solution:
            messagebox.showwarning("Предупреждение", "Сначала решите задачу")
            return

        try:
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            fig, ax = plt.subplots(figsize=(8, 6))

            if len(self.current_solution['solution']) == 2:
                x = np.linspace(-10, 10, 100)
                y = np.linspace(-10, 10, 100)
                X, Y = np.meshgrid(x, y)

                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            Z[i, j] = eval(self.current_solution['objective'],
                                           {'x': np.array([X[i, j], Y[i, j]])})
                        except:
                            Z[i, j] = np.nan

                cs = ax.contour(X, Y, Z, levels=20)
                ax.clabel(cs, inline=True, fontsize=10)

                sol = self.current_solution['solution']
                ax.plot(sol[0], sol[1], 'ro', markersize=10)

                # Добавляем дробные значения в подпись
                x1_frac = self.float_to_fraction(sol[0])
                x2_frac = self.float_to_fraction(sol[1])
                ax.annotate(f'Решение:\nx1={x1_frac}\nx2={x2_frac}',
                            xy=(sol[0], sol[1]),
                            xytext=(sol[0] + 1, sol[1] + 1),
                            arrowprops=dict(facecolor='black', shrink=0.05))

                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_title('Графическое решение задачи')
                ax.grid(True)

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)

            self.plot_figure = fig

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении графика: {str(e)}")

    def update_history_list(self):
        self.history_listbox.delete(0, END)
        for i, sol in enumerate(self.solutions_history):
            opt_value = -sol['optimal_value'] if self.problem_type.get() == "max" else sol['optimal_value']
            opt_value_frac = self.float_to_fraction(opt_value)
            self.history_listbox.insert(END,
                                        f"{i + 1}. {sol['timestamp']} - f(x)={opt_value:.2f} ({opt_value_frac})")

    def show_selected_solution(self, event):
        selection = self.history_listbox.curselection()
        if selection:
            self.current_solution = self.solutions_history[selection[0]]
            self.display_current_solution()

    def display_current_solution(self):
        if not self.current_solution:
            return

        self.result_text.config(state=NORMAL)
        self.result_text.delete(1.0, END)

        self.result_text.insert(END, f"Решение от {self.current_solution['timestamp']}\n\n")
        self.result_text.insert(END, f"Целевая функция: {self.current_solution['objective']}\n\n")
        self.result_text.insert(END, "Ограничения:\n")
        self.result_text.insert(END, self.current_solution['constraints'])
        self.result_text.insert(END, "\n\nРезультат:\n")

        if self.current_solution['success']:
            opt_value = -self.current_solution['optimal_value'] if self.problem_type.get() == "max" else \
            self.current_solution['optimal_value']
            opt_value_frac = self.float_to_fraction(opt_value)
            self.result_text.insert(END, f"Оптимальное значение: {opt_value:.6f} ({opt_value_frac})\n\n")
            self.result_text.insert(END, "Оптимальные параметры:\n")
            for i, val in enumerate(self.current_solution['solution']):
                val_frac = self.float_to_fraction(val)
                self.result_text.insert(END, f"x{i + 1} = {val:.6f} ({val_frac})\n")
        else:
            self.result_text.insert(END, f"Решение не найдено: {self.current_solution['message']}")

        self.result_text.config(state=DISABLED)

    def save_solution(self):
        if not self.current_solution:
            messagebox.showwarning("Предупреждение", "Нет решения для сохранения")
            return

        try:
            solution_to_save = {
                'timestamp': str(self.current_solution['timestamp']),
                'objective': str(self.current_solution['objective']),
                'constraints': str(self.current_solution['constraints']),
                'initial_guess': [float(x) for x in self.current_solution['initial_guess']],
                'solution': [float(x) for x in self.current_solution['solution']],
                'optimal_value': float(self.current_solution['optimal_value']),
                'success': bool(self.current_solution['success']),
                'message': str(self.current_solution['message'])
            }

            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Сохранить решение"
            )

            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(solution_to_save, f, indent=4, ensure_ascii=False)
                messagebox.showinfo("Успех", "Решение успешно сохранено")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении: {str(e)}")

    def load_solution(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Загрузить решение"
        )

        if filepath:
            try:
                with open(filepath, 'r') as f:
                    loaded_data = json.load(f)

                if not all(key in loaded_data for key in ['objective', 'solution', 'optimal_value']):
                    raise ValueError("Некорректный формат файла")

                solution = {
                    'timestamp': loaded_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    'objective': str(loaded_data['objective']),
                    'constraints': str(loaded_data.get('constraints', '')),
                    'initial_guess': [float(x) for x in loaded_data.get('initial_guess', [])],
                    'solution': [float(x) for x in loaded_data['solution']],
                    'optimal_value': float(loaded_data['optimal_value']),
                    'success': bool(loaded_data.get('success', False)),
                    'message': str(loaded_data.get('message', ''))
                }

                self.current_solution = solution
                self.solutions_history.append(solution)
                self.update_history_list()
                self.display_current_solution()
                messagebox.showinfo("Успех", "Решение успешно загружено")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при загрузке: {str(e)}")

    def clear_input(self):
        self.objective_entry.delete(0, END)
        self.constraints_text.delete("1.0", END)
        self.initial_guess_entry.delete(0, END)
        self.problem_type.set("min")

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        self.result_text.config(state=NORMAL)
        self.result_text.delete(1.0, END)
        self.result_text.config(state=DISABLED)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    try:
        import numpy as np
        from scipy.optimize import minimize
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Необходимо установить зависимости: {e}")
        print("Попробуйте: pip install numpy scipy matplotlib")
        exit(1)

    app = NonlinearSolver()
    app.run()