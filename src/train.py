import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

class PokemonModelTrainer:
    def __init__(self):
        self.model = None
        
    def load_data(self):
        """Charge les données d'entraînement et de test"""
        X_train = np.load('src/data/X_train.npy')
        X_test = np.load('src/data/X_test.npy')
        y_train = np.load('src/data/y_train.npy')
        y_test = np.load('src/data/y_test.npy')
        return X_train, X_test, y_train, y_test
    
    def load_best_params(self):
        """Charge les meilleurs hyperparamètres"""
        return joblib.load('src/model/best_params.pkl')
    
    def train_model(self, X_train, y_train):
        """Entraîne le modèle avec les meilleurs hyperparamètres"""
        best_params = self.load_best_params()
        self.model = RandomForestClassifier(**best_params, random_state=42)
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        """Évalue les performances du modèle"""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, report
    
    def save_model(self):
        """Sauvegarde le modèle entraîné"""
        if self.model is not None:
            joblib.dump(self.model, 'src/model/pokemon_model.pkl')

def main():
    trainer = PokemonModelTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    
    print("Entraînement du modèle...")
    trainer.train_model(X_train, y_train)
    
    print("Évaluation du modèle...")
    accuracy, report = trainer.evaluate_model(X_test, y_test)
    
    print(f"\nPrécision du modèle: {accuracy:.4f}")
    print("\nRapport de classification:")
    print(report)
    
    trainer.save_model()

if __name__ == "__main__":
    main()