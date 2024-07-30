import tkinter as tk
from tkinter import ttk, messagebox

class ConfigApp:
    def __init__(self, root):
        self.root = root
        root.title("Configuration Settings")

        # Setting up the frame layout
        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Variables
        self.regionalization = tk.BooleanVar()
        self.use_ga = tk.BooleanVar()
        self.enable_individual = tk.BooleanVar()
        self.number_of_agents = tk.IntVar()
        self.max_battery = tk.IntVar()

        # Widgets
        ttk.Checkbutton(mainframe, text="Regionalization", variable=self.regionalization).grid(column=1, row=1, sticky=tk.W)
        ttk.Checkbutton(mainframe, text="Use Genetic Algorithm", variable=self.use_ga).grid(column=1, row=2, sticky=tk.W)
        ttk.Checkbutton(mainframe, text="Enable Individual", variable=self.enable_individual).grid(column=1, row=3, sticky=tk.W)
        ttk.Entry(mainframe, textvariable=self.number_of_agents).grid(column=2, row=4, sticky=(tk.W, tk.E))
        ttk.Entry(mainframe, textvariable=self.max_battery).grid(column=2, row=5, sticky=(tk.W, tk.E))

        ttk.Label(mainframe, text="Number of Agents").grid(column=1, row=4, sticky=tk.W)
        ttk.Label(mainframe, text="Max Battery").grid(column=1, row=5, sticky=tk.W)
        
        ttk.Button(mainframe, text="Apply", command=self.apply_settings).grid(column=2, row=6, sticky=tk.W)

        # Padding for all children controls
        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def apply_settings(self):
        # This function can be used to process the settings or print them
        settings = {
            "Regionalization": self.regionalization.get(),
            "Use Genetic Algorithm": self.use_ga.get(),
            "Enable Individual": self.enable_individual.get(),
            "Number of Agents": self.number_of_agents.get(),
            "Max Battery": self.max_battery.get(),
        }
        messagebox.showinfo("Settings Applied", "\n".join(f"{k}: {v}" for k, v in settings.items()))

