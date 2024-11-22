import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_prepared_data():
    """Charge les données préparées."""
    X_train = np.load('src/data/X_train.npy')
    y_train = np.load('src/data/y_train.npy')
    
    # Reconstruction du DataFrame complet
    columns = [
        'p1_hp', 'p1_attack', 'p1_defense', 'p1_sp_attack', 'p1_sp_defense', 'p1_speed',
        'p1_type1', 'p1_type2', 'p1_legendary',
        'p2_hp', 'p2_attack', 'p2_defense', 'p2_sp_attack', 'p2_sp_defense', 'p2_speed',
        'p2_type1', 'p2_type2', 'p2_legendary'
    ]
    data = pd.DataFrame(X_train, columns=columns)
    data['winner'] = y_train
    return data

def plot_stat_distributions(data):
    """Visualise la distribution des stats pour les gagnants et perdants."""
    plt.figure(figsize=(15, 10))
    
    stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    
    for i, stat in enumerate(stats, 1):
        plt.subplot(2, 3, i)
        
        # Stats du P1 quand il gagne vs perd
        sns.kdeplot(data=data[data['winner'] == 1][f'p1_{stat}'], 
                   label='Winner', color='green')
        sns.kdeplot(data=data[data['winner'] == 0][f'p1_{stat}'], 
                   label='Loser', color='red')
        
        plt.title(f'Distribution of {stat}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('src/visualize/stat_distributions.png')
    plt.close()

def plot_correlation_matrix(data):
    """Visualise la matrice de corrélation entre les stats."""
    # Sélectionne uniquement les colonnes numériques (exclut type1, type2)
    numeric_cols = [col for col in data.columns if not any(x in col for x in ['type1', 'type2'])]
    corr_matrix = data[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Pokemon Stats')
    plt.tight_layout()
    plt.savefig('src/visualize/correlation_matrix.png')
    plt.close()

def plot_type_win_rates(data):
    """Visualise les taux de victoire par type."""
    plt.figure(figsize=(12, 6))
    
    # Compte les victoires par type
    type_wins = data[data['winner'] == 1]['p1_type1'].value_counts()
    type_total = data['p1_type1'].value_counts()
    win_rates = (type_wins / type_total * 100).sort_values(ascending=True)
    
    # Crée le graphique
    win_rates.plot(kind='barh')
    plt.title('Win Rate by Pokemon Type')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Type')
    plt.tight_layout()
    plt.savefig('src/visualize/type_win_rates.png')
    plt.close()

def plot_legendary_impact(data):
    """Visualise l'impact du statut légendaire sur les victoires."""
    plt.figure(figsize=(8, 6))
    
    # Calcule les taux de victoire
    legendary_stats = {
        'Normal vs Normal': data[(data['p1_legendary'] == 0) & (data['p2_legendary'] == 0)]['winner'].mean(),
        'Legendary vs Normal': data[(data['p1_legendary'] == 1) & (data['p2_legendary'] == 0)]['winner'].mean(),
        'Normal vs Legendary': data[(data['p1_legendary'] == 0) & (data['p2_legendary'] == 1)]['winner'].mean(),
        'Legendary vs Legendary': data[(data['p1_legendary'] == 1) & (data['p2_legendary'] == 1)]['winner'].mean()
    }
    
    plt.bar(legendary_stats.keys(), legendary_stats.values())
    plt.title('Win Rates in Different Legendary Matchups')
    plt.xticks(rotation=45)
    plt.ylabel('Win Rate')
    plt.tight_layout()
    plt.savefig('src/visualize/legendary_impact.png')
    plt.close()

def main():
    
    # Configure le style de seaborn
    sns.set_style("whitegrid")
    
    # Crée le dossier pour les visualisations
    Path('src/visualize').mkdir(exist_ok=True)
    
    # Charge les données
    data = load_prepared_data()
    
    # Génère les visualisations
    plot_stat_distributions(data)
    plot_correlation_matrix(data)
    plot_type_win_rates(data)
    plot_legendary_impact(data)
    
    print("Visualisations générées dans le dossier src/data/")

if __name__ == "__main__":
    main()