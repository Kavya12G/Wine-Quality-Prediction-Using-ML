# import tkinter as tk
# from tkinter import ttk
# import joblib
# import numpy as np
# try:
#     # Load pre-trained model
#     model = joblib.load(r"C:\Users\kavya\OneDrive\Documents\wine_model.pkl")
# except Exception as e:
#     print("Error loading the model:", e)


# def predict_quality():
#     try:
#         # Get input values from user
#         fixed_acidity = float(fixed_acidity_entry.get())
#         volatile_acidity = float(volatile_acidity_entry.get())
#         citric_acid = float(citric_acid_entry.get())
#         residual_sugar = float(residual_sugar_entry.get())
#         chlorides = float(chlorides_entry.get())
#         free_sulfur_dioxide = float(free_sulfur_dioxide_entry.get())
#         total_sulfur_dioxide = float(total_sulfur_dioxide_entry.get())
#         density = float(density_entry.get())
#         pH = float(pH_entry.get())
#         sulphates = float(sulphates_entry.get())
#         alcohol = float(alcohol_entry.get())

#         # Create input array for prediction
#         input_data = np.array([
#             [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
#              free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
#         ])

#         # Make prediction
#         prediction = model.predict(input_data)
        
#         # Display prediction
#         prediction_label.config(text=f'Predicted Quality: {prediction[0]}', font=('Helvetica', 20), foreground='blue')
#     except ValueError:
#         prediction_label.config(text='Please enter valid input for all fields.', font=('Helvetica', 20), foreground='red')

# # Create main windows
# root = tk.Tk()
# root.title('Wine Quality Prediction')

# # Load background image
# bg_image = tk.PhotoImage(file=r"c:\Users\kavya\OneDrive\Pictures\background_image.png")

# # Create a label to hold the background image
# background_label = tk.Label(root, image=bg_image)
# background_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# # Create labels and entry fields for input features
# label_font = ('Helvetica', 20)
# entry_font = ('Helvetica', 20)
# entry_width = 20


# # Create labels and entry fields for input features
# fixed_acidity_label = ttk.Label(root, text='Fixed Acidity:', font=label_font)
# fixed_acidity_label.grid(row=0, column=0, padx=20, pady=10)
# fixed_acidity_entry = ttk.Entry(root)
# fixed_acidity_entry.grid(row=0, column=1, padx=20, pady=10)

# volatile_acidity_label = ttk.Label(root, text='Volatile Acidity:', font=label_font)
# volatile_acidity_label.grid(row=1, column=0, padx=10, pady=5)
# volatile_acidity_entry = ttk.Entry(root)
# volatile_acidity_entry.grid(row=1, column=1, padx=10, pady=5)

# citric_acid_label = ttk.Label(root, text='Citric Acid:', font=label_font)
# citric_acid_label.grid(row=2, column=0, padx=10, pady=5)
# citric_acid_entry = ttk.Entry(root)
# citric_acid_entry.grid(row=2, column=1, padx=10, pady=5)

# residual_sugar_label = ttk.Label(root, text='Residual Sugar:', font=label_font)
# residual_sugar_label.grid(row=3, column=0, padx=10, pady=5)
# residual_sugar_entry = ttk.Entry(root)
# residual_sugar_entry.grid(row=3, column=1, padx=10, pady=5)

# chlorides_label = ttk.Label(root, text='Chlorides:', font=label_font)
# chlorides_label.grid(row=4, column=0, padx=10, pady=5)
# chlorides_entry = ttk.Entry(root)
# chlorides_entry.grid(row=4, column=1, padx=10, pady=5)

# free_sulfur_dioxide_label = ttk.Label(root, text='Free Sulfur Dioxide:', font=label_font)
# free_sulfur_dioxide_label.grid(row=5, column=0, padx=10, pady=5)
# free_sulfur_dioxide_entry = ttk.Entry(root)
# free_sulfur_dioxide_entry.grid(row=5, column=1, padx=10, pady=5)

# total_sulfur_dioxide_label = ttk.Label(root, text='Total Sulfur Dioxide:', font=label_font)
# total_sulfur_dioxide_label.grid(row=6, column=0, padx=10, pady=5)
# total_sulfur_dioxide_entry = ttk.Entry(root)
# total_sulfur_dioxide_entry.grid(row=6, column=1, padx=10, pady=5)

# density_label = ttk.Label(root, text='Density:', font=label_font)
# density_label.grid(row=7, column=0, padx=10, pady=5)
# density_entry = ttk.Entry(root)
# density_entry.grid(row=7, column=1, padx=10, pady=5)

# pH_label = ttk.Label(root, text='pH:', font=label_font)
# pH_label.grid(row=8, column=0, padx=10, pady=5)
# pH_entry = ttk.Entry(root)
# pH_entry.grid(row=8, column=1, padx=10, pady=5)

# sulphates_label = ttk.Label(root, text='Sulphates:', font=label_font)
# sulphates_label.grid(row=9, column=0, padx=10, pady=5)
# sulphates_entry = ttk.Entry(root)
# sulphates_entry.grid(row=9, column=1, padx=10, pady=5)

# alcohol_label = ttk.Label(root, text='Alcohol:', font=label_font)
# alcohol_label.grid(row=10, column=0, padx=10, pady=5)
# alcohol_entry = ttk.Entry(root)
# alcohol_entry.grid(row=10, column=1, padx=10, pady=5)

# # Create button to predict wine quality
# predict_button = ttk.Button(root, text='Predict Quality', command=predict_quality)
# predict_button.grid(row=11, column=0, columnspan=2, padx=10, pady=10)
# predict_button_style = ttk.Style()
# predict_button_style.configure('Predict.TButton', font=('Helvetica', 20))
# predict_button['style'] = 'Predict.TButton'

# # Create label to display prediction
# prediction_label = ttk.Label(root, text='', font=('Helvetica', 20))
# prediction_label.grid(row=12, column=0, columnspan=2, padx=10, pady=5)

# root.mainloop()


import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np

# Load pre-trained models
models = {
    'Logistic Regression': r"C:\Users\kavya\OneDrive\Documents\wine_model.pkl",
    'XGBoost Classifier': r"C:\Users\kavya\OneDrive\Documents\wine_model.pkl",
    'SVM': r"C:\Users\kavya\OneDrive\Documents\wine_model.pkl"
}

current_model = None

def load_model(model_name):
    global current_model
    try:
        current_model = joblib.load(models[model_name])
        prediction_label.config(text=f'{model_name} model loaded successfully.', font=('Helvetica', 20), foreground='green')
        show_feature_fields()
    except Exception as e:
        prediction_label.config(text=f'Error loading the {model_name} model: {e}', font=('Helvetica', 20), foreground='red')

def show_feature_fields():
    for widget in feature_frame.winfo_children():
        widget.grid()
        
def hide_feature_fields():
    for widget in feature_frame.winfo_children():
        widget.grid_remove()

def predict_quality():
    try:
        # Get input values from user
        fixed_acidity = float(fixed_acidity_entry.get())
        volatile_acidity = float(volatile_acidity_entry.get())
        citric_acid = float(citric_acid_entry.get())
        residual_sugar = float(residual_sugar_entry.get())
        chlorides = float(chlorides_entry.get())
        free_sulfur_dioxide = float(free_sulfur_dioxide_entry.get())
        total_sulfur_dioxide = float(total_sulfur_dioxide_entry.get())
        density = float(density_entry.get())
        pH = float(pH_entry.get())
        sulphates = float(sulphates_entry.get())
        alcohol = float(alcohol_entry.get())

        # Create input array for prediction
        input_data = np.array([
            [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
             free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
        ])

        # Make prediction
        prediction = current_model.predict(input_data)
        
        # Display prediction
        prediction_label.config(text=f'Predicted Quality: {prediction[0]}', font=('Helvetica', 20), foreground='blue')
    except ValueError:
        prediction_label.config(text='Please enter valid input for all fields.', font=('Helvetica', 20), foreground='red')

# Create main windows
root = tk.Tk()
root.title('Wine Quality Prediction')

# Load background image
bg_image = tk.PhotoImage(file=r"C:\Users\kavya\OneDrive\Pictures\background_image.png")

# Create a label to hold the background image
background_label = tk.Label(root, image=bg_image)
background_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Create a frame for feature input fields
feature_frame = ttk.Frame(root)
feature_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Create labels and entry fields for input features
label_font = ('Helvetica', 20)
entry_font = ('Helvetica', 20)
entry_width = 20

fixed_acidity_label = ttk.Label(feature_frame, text='Fixed Acidity:', font=label_font)
fixed_acidity_label.grid(row=0, column=0, padx=20, pady=10)
fixed_acidity_entry = ttk.Entry(feature_frame)
fixed_acidity_entry.grid(row=0, column=1, padx=20, pady=10)

volatile_acidity_label = ttk.Label(feature_frame, text='Volatile Acidity:', font=label_font)
volatile_acidity_label.grid(row=1, column=0, padx=10, pady=5)
volatile_acidity_entry = ttk.Entry(feature_frame)
volatile_acidity_entry.grid(row=1, column=1, padx=10, pady=5)

citric_acid_label = ttk.Label(feature_frame, text='Citric Acid:', font=label_font)
citric_acid_label.grid(row=2, column=0, padx=10, pady=5)
citric_acid_entry = ttk.Entry(feature_frame)
citric_acid_entry.grid(row=2, column=1, padx=10, pady=5)

residual_sugar_label = ttk.Label(feature_frame, text='Residual Sugar:', font=label_font)
residual_sugar_label.grid(row=3, column=0, padx=10, pady=5)
residual_sugar_entry = ttk.Entry(feature_frame)
residual_sugar_entry.grid(row=3, column=1, padx=10, pady=5)

chlorides_label = ttk.Label(feature_frame, text='Chlorides:', font=label_font)
chlorides_label.grid(row=4, column=0, padx=10, pady=5)
chlorides_entry = ttk.Entry(feature_frame)
chlorides_entry.grid(row=4, column=1, padx=10, pady=5)

free_sulfur_dioxide_label = ttk.Label(feature_frame, text='Free Sulfur Dioxide:', font=label_font)
free_sulfur_dioxide_label.grid(row=5, column=0, padx=10, pady=5)
free_sulfur_dioxide_entry = ttk.Entry(feature_frame)
free_sulfur_dioxide_entry.grid(row=5, column=1, padx=10, pady=5)

total_sulfur_dioxide_label = ttk.Label(feature_frame, text='Total Sulfur Dioxide:', font=label_font)
total_sulfur_dioxide_label.grid(row=6, column=0, padx=10, pady=5)
total_sulfur_dioxide_entry = ttk.Entry(feature_frame)
total_sulfur_dioxide_entry.grid(row=6, column=1, padx=10, pady=5)

density_label = ttk.Label(feature_frame, text='Density:', font=label_font)
density_label.grid(row=7, column=0, padx=10, pady=5)
density_entry = ttk.Entry(feature_frame)
density_entry.grid(row=7, column=1, padx=10, pady=5)

pH_label = ttk.Label(feature_frame, text='pH:', font=label_font)
pH_label.grid(row=8, column=0, padx=10, pady=5)
pH_entry = ttk.Entry(feature_frame)
pH_entry.grid(row=8, column=1, padx=10, pady=5)

sulphates_label = ttk.Label(feature_frame, text='Sulphates:', font=label_font)
sulphates_label.grid(row=9, column=0, padx=10, pady=5)
sulphates_entry = ttk.Entry(feature_frame)
sulphates_entry.grid(row=9, column=1, padx=10, pady=5)

alcohol_label = ttk.Label(feature_frame, text='Alcohol:', font=label_font)
alcohol_label.grid(row=10, column=0, padx=10, pady=5)
alcohol_entry = ttk.Entry(feature_frame)
alcohol_entry.grid(row=10, column=1, padx=10, pady=5)

# Initially hide feature fields
hide_feature_fields()

# Create buttons to select the algorithm
logistic_button = ttk.Button(root, text='Logistic Regression', command=lambda: load_model('Logistic Regression'))
logistic_button.grid(row=0, column=0, padx=10, pady=10)

xgboost_button = ttk.Button(root, text='XGBoost Classifier', command=lambda: load_model('XGBoost Classifier'))
xgboost_button.grid(row=0, column=1, padx=10, pady=10)

svm_button = ttk.Button(root, text='SVM', command=lambda: load_model('SVM'))
svm_button.grid(row=1, column=0, padx=10, pady=10)

# Create button to predict wine quality
predict_button = ttk.Button(root, text='Predict Quality', command=predict_quality)
predict_button.grid(row=1, column=1, padx=10, pady=10)
predict_button_style = ttk.Style()
predict_button_style.configure('Predict.TButton', font=('Helvetica', 20))
predict_button['style'] = 'Predict.TButton'

# Create label to display prediction
prediction_label = ttk.Label(root, text='', font=('Helvetica', 20))
prediction_label.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

root.mainloop()
