"""
Точка входа в приложение для решения систем линейных уравнений (СЛАУ).

Создает экземпляр главного окна Tkinter, инициализирует объект
пользовательского интерфейса App и запускает цикл обработки событий mainloop.
"""

import tkinter as tk

from app import App


def main() -> None:
    """
    Создает корневое окно Tkinter, инициализирует интерфейс и запускает приложение.
    """
    root: tk.Tk = tk.Tk()
    _app: App = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
