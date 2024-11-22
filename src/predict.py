import joblib
import pandas as pd
import numpy as np
from typing import Dict, Union, Tuple
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

class PokemonBattlePredictor:
    def __init__(self):
        """Initialise le prédicteur avec le modèle entraîné"""
        model_path = Path('src/model/pokemon_model.pkl')
        if not model_path.exists():
            raise FileNotFoundError("Le modèle n'a pas été trouvé. Exécutez d'abord train.py")
            
        self.pokedex = pd.read_csv('src/data/pokedex.csv', sep=';')
        self.model = joblib.load(model_path)
        self.type_encoder = LabelEncoder()
        self._prepare_pokedex()
        
    def _prepare_pokedex(self):
        """Prépare le Pokédex comme pendant l'entraînement"""
        self.pokedex['TYPE_1_encoded'] = self.type_encoder.fit_transform(self.pokedex['TYPE_1'])
        mask = self.pokedex['TYPE_2'].notna()
        self.pokedex.loc[mask, 'TYPE_2_encoded'] = self.type_encoder.transform(self.pokedex.loc[mask, 'TYPE_2'])
        self.pokedex['TYPE_2_encoded'] = self.pokedex['TYPE_2_encoded'].fillna(-1)
        self.pokedex['LEGENDAIRE'] = (self.pokedex['LEGENDAIRE'] == 'VRAI').astype(int)
        
    def get_pokemon_data(self, pokemon_number: int) -> pd.Series:
        """Récupère les données d'un Pokémon par son numéro"""
        return self.pokedex[self.pokedex['NUMERO'] == pokemon_number].iloc[0]
    
    def predict_battle(self, pokemon1_number: int, pokemon2_number: int) -> Dict[str, Union[float, str]]:
        """Prédit le résultat d'un combat entre deux Pokémon"""
        pokemon1 = self.get_pokemon_data(pokemon1_number)
        pokemon2 = self.get_pokemon_data(pokemon2_number)
        
        battle_features = np.array([
            pokemon1['POINTS_DE_VIE'], pokemon1['POINTS_ATTAQUE'], pokemon1['POINTS_DEFFENCE'],
            pokemon1['POINTS_ATTAQUE_SPECIALE'], pokemon1['POINT_DEFENSE_SPECIALE'], 
            pokemon1['POINTS_VITESSE'], pokemon1['TYPE_1_encoded'], pokemon1['TYPE_2_encoded'],
            pokemon1['LEGENDAIRE'],
            pokemon2['POINTS_DE_VIE'], pokemon2['POINTS_ATTAQUE'], pokemon2['POINTS_DEFFENCE'],
            pokemon2['POINTS_ATTAQUE_SPECIALE'], pokemon2['POINT_DEFENSE_SPECIALE'],
            pokemon2['POINTS_VITESSE'], pokemon2['TYPE_1_encoded'], pokemon2['TYPE_2_encoded'],
            pokemon2['LEGENDAIRE']
        ]).reshape(1, -1)
        
        prediction_proba = self.model.predict_proba(battle_features)[0]
        print('\n-----------------------')
        print('Predicition response')
        print('-----------------------')
        print(prediction_proba)
        print('-----------------------')
        
        return {
            'pokemon1_name': pokemon1['NOM'],
            'pokemon2_name': pokemon2['NOM'],
            'pokemon1_win_probability': prediction_proba[1],
            'pokemon2_win_probability': prediction_proba[0],
            'predicted_winner': pokemon1['NOM'] if prediction_proba[1] > prediction_proba[0] else pokemon2['NOM']
        }

def display_pokemon_info(pokemon_data: pd.Series) -> None:
    """Affiche les informations d'un Pokémon"""
    print(f"\n{pokemon_data['NOM']}:")
    print(f"Types: {pokemon_data['TYPE_1']}", end='')
    if pd.notna(pokemon_data['TYPE_2']):
        print(f"/{pokemon_data['TYPE_2']}", end='')
    print(f"\nPV: {pokemon_data['POINTS_DE_VIE']}")
    print(f"Attaque: {pokemon_data['POINTS_ATTAQUE']}")
    print(f"Defense: {pokemon_data['POINTS_DEFFENCE']}")
    print(f"Attaque Speciale: {pokemon_data['POINTS_ATTAQUE_SPECIALE']}")
    print(f"Defense Speciale: {pokemon_data['POINT_DEFENSE_SPECIALE']}")
    print(f"Vitesse: {pokemon_data['POINTS_VITESSE']}")
    if pokemon_data['LEGENDAIRE']:
        print("* Pokemon Legendaire *")

def main():
    try:
        predictor = PokemonBattlePredictor()
        
        # Exemple d'utilisation
        pokemon1_number = 1  # Bulbizarre
        pokemon2_number = 4  # Salamèche
        
        # Affiche les infos des Pokémon
        pokemon1_data = predictor.get_pokemon_data(pokemon1_number)
        pokemon2_data = predictor.get_pokemon_data(pokemon2_number)
        
        display_pokemon_info(pokemon1_data)
        display_pokemon_info(pokemon2_data)
        
        # Prédiction
        result = predictor.predict_battle(pokemon1_number, pokemon2_number)
        
        print("\nPrediction du combat:")
        print(f"{result['pokemon1_name']} VS {result['pokemon2_name']}")
        print(f"\nProbabilite de victoire:")
        print(f"  {result['pokemon1_name']}: {result['pokemon1_win_probability']:.1%}")
        print(f"  {result['pokemon2_name']}: {result['pokemon2_win_probability']:.1%}")
        print(f"\nVainqueur predit: {result['predicted_winner']}")
        
    except FileNotFoundError as e:
        print(f"Erreur: {str(e)}")
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")

if __name__ == "__main__":
    main()