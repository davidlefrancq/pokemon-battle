import os
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import joblib

class PokemonHyperparameterOptimizer:
    def __init__(self):
        self.best_params = None
        self.best_score = None
        
    def load_data(self):
        """Charge les données d'entraînement"""
        X_train = np.load('src/data/X_train.npy')
        y_train = np.load('src/data/y_train.npy')
        return X_train, y_train
    
    def optimize_parameters(self, X_train, y_train):
        """Trouve les meilleurs hyperparamètres"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],
            'bootstrap': [True, False]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        return self.best_params, self.best_score
    
    def save_best_params(self):
        Path('src/model').mkdir(exist_ok=True)
        if self.best_params is not None:
            joblib.dump(self.best_params, 'src/model/best_params.pkl')
        else:
            print("Aucun hyperparamètre à sauvegarder")

def main():
    dateStart = datetime.datetime.now()
    optimizer = PokemonHyperparameterOptimizer()
    X_train, y_train = optimizer.load_data()
    best_params, best_score = optimizer.optimize_parameters(X_train, y_train)
    dateEnd = datetime.datetime.now()
    print(f"\nTemps d'exécution: {dateEnd - dateStart}")
    
    print("Meilleurs paramètres trouvés:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"\nMeilleur score: {best_score:.4f}")
    
    optimizer.save_best_params()

if __name__ == "__main__":
    main()