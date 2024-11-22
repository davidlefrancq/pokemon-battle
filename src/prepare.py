import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

class PokemonDataPreparator:
    """Prépare les données des combats Pokémon pour prédire les probabilités de victoire."""
    
    def __init__(self):
        """Initialise les encodeurs et DataFrames."""
        self.type_encoder = LabelEncoder()
        self.pokedex = None
        self.combats = None
        
    def load_data(self, pokedex_path: str, combats_path: str) -> None:
        """Charge les données et garde les colonnes utiles pour la prédiction."""
        # Ne garde que les colonnes nécessaires pour la prédiction
        columns_to_keep = ['NUMERO', 'NOM', 'TYPE_1', 'TYPE_2', 'POINTS_DE_VIE', 
                          'POINTS_ATTAQUE', 'POINTS_DEFFENCE', 'POINTS_ATTAQUE_SPECIALE', 
                          'POINT_DEFENSE_SPECIALE', 'POINTS_VITESSE', 'LEGENDAIRE']
        self.pokedex = pd.read_csv(pokedex_path, sep=';', usecols=columns_to_keep)
        self.combats = pd.read_csv(combats_path)
    
    def prepare_features(self) -> pd.DataFrame:
        """Prépare le dataset d'entraînement à partir des combats."""
        # Encode les types
        self.pokedex['TYPE_1_encoded'] = self.type_encoder.fit_transform(self.pokedex['TYPE_1'])
        mask = self.pokedex['TYPE_2'].notna()
        self.pokedex.loc[mask, 'TYPE_2_encoded'] = self.type_encoder.transform(self.pokedex.loc[mask, 'TYPE_2'])
        self.pokedex['TYPE_2_encoded'] = self.pokedex['TYPE_2_encoded'].fillna(-1)
        
        # Encode le statut légendaire
        self.pokedex['LEGENDAIRE'] = (self.pokedex['LEGENDAIRE'] == 'VRAI').astype(int)
        
        # Prépare les features des combats
        data = []
        for _, row in self.combats.iterrows():
            pokemon1 = self.pokedex[self.pokedex['NUMERO'] == row['Premier_Pokemon']].iloc[0]
            pokemon2 = self.pokedex[self.pokedex['NUMERO'] == row['Second_Pokemon']].iloc[0]
            
            # Features du combat
            features = [
                pokemon1['POINTS_DE_VIE'], pokemon1['POINTS_ATTAQUE'], pokemon1['POINTS_DEFFENCE'],
                pokemon1['POINTS_ATTAQUE_SPECIALE'], pokemon1['POINT_DEFENSE_SPECIALE'], 
                pokemon1['POINTS_VITESSE'], pokemon1['TYPE_1_encoded'], pokemon1['TYPE_2_encoded'],
                pokemon1['LEGENDAIRE'],
                pokemon2['POINTS_DE_VIE'], pokemon2['POINTS_ATTAQUE'], pokemon2['POINTS_DEFFENCE'],
                pokemon2['POINTS_ATTAQUE_SPECIALE'], pokemon2['POINT_DEFENSE_SPECIALE'],
                pokemon2['POINTS_VITESSE'], pokemon2['TYPE_1_encoded'], pokemon2['TYPE_2_encoded'],
                pokemon2['LEGENDAIRE'],
                1 if row['Pokemon_Gagnant'] == row['Premier_Pokemon'] else 0
            ]
            data.append(features)
        
        # Noms des colonnes
        columns = [
            'p1_hp', 'p1_attack', 'p1_defense', 'p1_sp_attack', 'p1_sp_defense', 'p1_speed',
            'p1_type1', 'p1_type2', 'p1_legendary',
            'p2_hp', 'p2_attack', 'p2_defense', 'p2_sp_attack', 'p2_sp_defense', 'p2_speed',
            'p2_type1', 'p2_type2', 'p2_legendary',
            'winner'
        ]
        
        return pd.DataFrame(data, columns=columns)
    
    def get_pokemon_data(self, numero: int) -> pd.Series:
        """Récupère les données d'un Pokémon par son numéro."""
        return self.pokedex[self.pokedex['NUMERO'] == numero].iloc[0]

    def split_data(self, features_df: pd.DataFrame) -> tuple:
        """Divise les données en ensembles d'entraînement et de test."""
        X = features_df.drop('winner', axis=1)
        y = features_df['winner']
        return train_test_split(X, y, test_size=0.2, random_state=42)


def main():
    # Crée le dossier data s'il n'existe pas
    Path('src/data').mkdir(exist_ok=True)
    Path('src/model').mkdir(exist_ok=True)
    
    # Prépare les données
    preparator = PokemonDataPreparator()
    preparator.load_data('src/data/pokedex.csv', 'src/data/combats.csv')
    
    # Sauvegarde le préparateur pour pouvoir accéder aux données des Pokémon plus tard
    pd.to_pickle(preparator, 'src/model/preparator.pkl')
    
    # Prépare et sauvegarde les données d'entraînement
    features_df = preparator.prepare_features()
    X_train, X_test, y_train, y_test = preparator.split_data(features_df)
    
    np.save('src/data/X_train.npy', X_train)
    np.save('src/data/X_test.npy', X_test)
    np.save('src/data/y_train.npy', y_train)
    np.save('src/data/y_test.npy', y_test)


if __name__ == "__main__":
    main()