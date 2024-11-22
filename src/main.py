import pandas as pd
from predict import PokemonBattlePredictor, display_pokemon_info
from typing import List, Tuple

class PokemonBattleSimulator:
    def __init__(self):
        """Initialise le simulateur avec le Pokédex et le prédicteur"""
        self.pokedex = pd.read_csv('src/data/pokedex.csv', sep=';')
        self.predictor = PokemonBattlePredictor()
        
    def search_pokemon(self, query: str) -> List[Tuple[int, str]]:
        """
        Recherche un Pokémon par son nom ou numéro.
        Retourne une liste de tuples (numéro, nom)
        """
        # Si la recherche est un numéro
        if query.isdigit():
            pokemon = self.pokedex[self.pokedex['NUMERO'] == int(query)]
            if not pokemon.empty:
                return [(pokemon.iloc[0]['NUMERO'], pokemon.iloc[0]['NOM'])]
            
        # Sinon recherche par nom
        query = query.lower()
        matches = self.pokedex[self.pokedex['NOM'].str.lower().str.contains(query, regex=False)]
        return list(zip(matches['NUMERO'], matches['NOM']))
    
    def get_pokemon_choice(self) -> int:
        """Interface utilisateur pour choisir un Pokémon"""
        while True:
            search = input("\nRechercher un Pokemon (nom ou numero): ").strip()
            if not search:
                continue
                
            results = self.search_pokemon(search)
            
            if not results:
                print("Aucun Pokemon trouve. Essayez a nouveau.")
                continue
                
            # Si un seul résultat, on le retourne directement
            if len(results) == 1:
                numero, nom = results[0]
                print(f"Pokemon selectionne: {nom} (#{numero})")
                return numero
                
            # Sinon on affiche la liste des résultats
            print("\nPokemons trouves:")
            for i, (numero, nom) in enumerate(results, 1):
                print(f"{i}. {nom} (#{numero})")
                
            # Demande à l'utilisateur de choisir
            while True:
                try:
                    choice = input("\nChoisissez un numero (ou 'q' pour rechercher a nouveau): ")
                    if choice.lower() == 'q':
                        break
                    index = int(choice) - 1
                    if 0 <= index < len(results):
                        return results[index][0]
                    print("Choix invalide")
                except ValueError:
                    print("Veuillez entrer un numero valide")
    
    def simulate_battle(self):
        """Lance une simulation de combat entre deux Pokémon"""
        print("\n=== SIMULATION DE COMBAT POKEMON ===")
        
        print("Choisissez le premier Pokemon:")
        pokemon1_number = self.get_pokemon_choice()
        
        print("\nChoisissez le deuxieme Pokemon:")
        pokemon2_number = self.get_pokemon_choice()
        
        # Affiche les informations des Pokémon et la prédiction
        pokemon1_data = self.predictor.get_pokemon_data(pokemon1_number)
        pokemon2_data = self.predictor.get_pokemon_data(pokemon2_number)
        
        # Affiche les infos en utilisant la fonction importée
        print("\n=== INFORMATIONS DES COMBATTANTS ===")
        display_pokemon_info(pokemon1_data)
        display_pokemon_info(pokemon2_data)
        
        # Fait la prédiction
        result = self.predictor.predict_battle(pokemon1_number, pokemon2_number)
        
        # Affiche les résultats
        print("\n=== PREDICTION DU COMBAT ===")
        print(f"{result['pokemon1_name']} VS {result['pokemon2_name']}")
        print(f"\nProbabilite de victoire:")
        print(f"  {result['pokemon1_name']}: {result['pokemon1_win_probability']:.1%}")
        print(f"  {result['pokemon2_name']}: {result['pokemon2_win_probability']:.1%}")
        print(f"\nVainqueur predit: {result['predicted_winner']}")

def main():
    try:
        simulator = PokemonBattleSimulator()
        
        while True:
            simulator.simulate_battle()
            
            # Demande si l'utilisateur veut faire un autre combat
            again = input("\nVoulez-vous simuler un autre combat? (o/n): ").lower()
            if again != 'o':
                break
                
        print("\nMerci d'avoir utilise le simulateur de combat Pokemon!")
        
    except Exception as e:
        print(f"\nUne erreur est survenue: {str(e)}")

if __name__ == "__main__":
    main()