import joblib

loaded_model = joblib.load("./models/iris_log_reg.pkl")

class model:
    @staticmethod
    def classement(data):
        proba = loaded_model.predict_proba(data)
        prediction = loaded_model.predict(data)
        
        class_names = loaded_model.classes_  # Obtenez les noms de classe dans le bon ordre
        class_probabilities = list(proba[0])  # Convertissez le tableau NumPy en une liste
        
        max_probability_index = class_probabilities.index(max(class_probabilities))
        result = "C'est un iris " + class_names[max_probability_index]
        
        probabilities = {class_names[i]: class_probabilities[i] for i in range(len(class_names))}
        
        return result, probabilities


