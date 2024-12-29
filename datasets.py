import os
import requests
import pandas as pd
from urllib.parse import urlencode
from bs4 import BeautifulSoup

def download_fred_data(start_date, end_date, filename="data/fred_data.csv", indicator_id="UNRATE", overwrite=False):
    """
    Télécharge les données de la FED à partir de l'URL FRED et les enregistre dans un fichier CSV.

    :param start_date: Date de début au format AAAA-MM-JJ (e.g., "2010-01-01").
    :param end_date: Date de fin au format AAAA-MM-JJ (e.g., "2024-10-01").
    :param filename: Chemin où le fichier CSV sera sauvegardé (par défaut : "data/fred_data.csv").
    :param indicator_id: ID de l'indicateur à télécharger (par défaut : "UNRATE").
    :param overwrite: Si True, remplace le fichier existant. Sinon, ne télécharge pas si le fichier existe.
    :return: Un DataFrame pandas contenant les données téléchargées.
    """
    # Créez le dossier "data" s'il n'existe pas
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Vérifiez si le fichier existe
    if os.path.exists(filename) and not overwrite:
        print(f"Le fichier existe déjà : {filename}. Utilisez overwrite=True pour le remplacer.")
    else:
        # Base de l'URL
        base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

        # Paramètres de l'URL
        params = {
            "bgcolor": "%23e1e9f0",
            "chart_type": "line",
            "drp": "0",
            "fo": "open sans",
            "graph_bgcolor": "%23ffffff",
            "height": "450",
            "mode": "fred",
            "recession_bars": "on",
            "txtcolor": "%23444444",
            "ts": "12",
            "tts": "12",
            "width": "1320",
            "nt": "0",
            "thu": "0",
            "trc": "0",
            "show_legend": "yes",
            "show_axis_titles": "yes",
            "show_tooltip": "yes",
            "id": indicator_id,
            "scale": "left",
            "cosd": start_date,
            "coed": end_date,
            "line_color": "%234572a7",
            "link_values": "false",
            "line_style": "solid",
            "mark_type": "none",
            "mw": "3",
            "lw": "3",
            "ost": "-99999",
            "oet": "99999",
            "mma": "0",
            "fml": "a",
            "fq": "Monthly",
            "fam": "avg",
            "fgst": "lin",
            "fgsnd": "2020-02-01",
            "line_index": "1",
            "transformation": "lin",
            "vintage_date": "2024-12-06",
            "revision_date": "2024-12-06",
            "nd": "1948-01-01",
        }

        # Construisez l'URL complète
        url = f"{base_url}?{urlencode(params)}"

        # Téléchargez le fichier
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Fichier téléchargé et sauvegardé dans : {filename}")
        else:
            print(f"Erreur lors du téléchargement : {response.status_code} - {response.text}")
            return None

    # Charger le fichier CSV
    try:
        df = pd.read_csv(filename)
        print("DataFrame créé avec succès.")
        # Convert observation_date column to datetime
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        # Set observation_date as index
        df.set_index('observation_date', inplace=True)
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du CSV : {e}")
        return None

#<a href='https://www.macrotrends.net/global-metrics/countries/usa/united-states/gdp-growth-rate'>U.S. GDP Growth Rate 1961-2024</a>

