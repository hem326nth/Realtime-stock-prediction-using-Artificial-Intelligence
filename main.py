import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import pickle
from datetime import date
import webbrowser

class PredictionApp(tk.Tk):
    def __init__(self, pickle_files):
        super().__init__()
        self.title("Stock Prediction App")
        self.pickle_files = pickle_files
        self.create_widgets()

    def load_pickle_file(self):
        selected_company = self.company_dropdown.get()
        return self.pickle_files.get(selected_company)

    def display_predicted_data(self):
        pickle_file = self.load_pickle_file()
        if pickle_file:
            try:
                date = self.date_picker.get_date()
                with open(pickle_file, 'rb') as file:
                    model, scaler = pickle.load(file)  # Load both model and scaler
                date_ordinal = date.toordinal()
                data = [[date_ordinal]]
                data_scaled = scaler.transform(data)  # Scale the input data
                prediction = model.predict(data_scaled)
                headings = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                result_text = "Predicted data:\n"
                for i, heading in enumerate(headings):
                    result_text += f"{heading}: {prediction[0][i]:.2f}\n"
                self.result_label.config(text=result_text, foreground="white", background="blue")
                # Determine whether to buy or sell stocks
                open_price, close_price = prediction[0][0], prediction[0][3]
                recommendation = "Buy" if open_price < close_price else "Sell"
                self.action_label.config(text=f"Recommended action: {recommendation}", foreground="white", background="green" if recommendation == "Buy" else "red")
                self.action_label.bind("<Button-1>", lambda event: self.open_webpage())
            except FileNotFoundError:
                self.result_label.config(text="Pickle file not found.", foreground="white", background="red")
        else:
            self.result_label.config(text="Select a company first.", foreground="white", background="red")

    def create_widgets(self):
        pickle_label = ttk.Label(self, text="Select Company:")
        pickle_label.grid(row=0, column=0, padx=10, pady=10)

        self.selected_company = tk.StringVar()
        self.company_dropdown = ttk.Combobox(self, textvariable=self.selected_company)
        self.company_dropdown['values'] = list(self.pickle_files.keys())
        self.company_dropdown.grid(row=0, column=1, padx=10, pady=10)

        date_label = ttk.Label(self, text="Select Date:")
        date_label.grid(row=1, column=0, padx=10, pady=10)

        # Set the minimum date to January 1st, 2002
        min_date = date(2002, 1, 1)
        self.date_picker = DateEntry(self, width=12, background='yellow',
                                      foreground='white', borderwidth=2, min_date=min_date)
        self.date_picker.grid(row=1, column=1, padx=10, pady=10)

        predict_button = tk.Button(self, text="Predict", command=self.display_predicted_data)
        predict_button.grid(row=1, column=2, padx=10, pady=10)

        self.result_label = ttk.Label(self, text="")
        self.result_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        self.action_label = ttk.Label(self, text="")
        self.action_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

    def open_webpage(self):
        url = "https://brokeriqoption.com/lp/mobile/en/?aff=97642&afftrack=GAD_PMax_ASIA_IN_EN_20715484768__&utm_source=google&utm_medium=cpc&utm_campaign=PMax_ASIA_IN_EN&utm_adset=&gad_source=1&gclid=Cj0KCQjwudexBhDKARIsAI-GWYWARcvPIGEjYxX5FXbuyKCa_icJQ4TRRIFmEjCfH6Q4uL40ebq4bjsaAnnaEALw_wcB"
        webbrowser.open(url)

if __name__ == "__main__":
    pickle_files = {
        "NETFLIX": r"C:\Users\heman\OneDrive\Documents\company_maang[1]\company_maang\NETFLIX_daily_model.pkl",
        "META": r"C:\Users\heman\OneDrive\Documents\company_maang[1]\company_maang\META_daily_model.pkl",
        "GOOGLE": r"C:\Users\heman\OneDrive\Documents\company_maang[1]\company_maang\GOOGLE_daily_model.pkl",
        "APPLE": r"C:\Users\heman\OneDrive\Documents\company_maang[1]\company_maang\APPLE_daily_model.pkl",
        "AMAZON": r"C:\Users\heman\OneDrive\Documents\company_maang[1]\company_maang\AMAZON_daily_model.pkl"
    }
    app = PredictionApp(pickle_files)
    app.mainloop()
