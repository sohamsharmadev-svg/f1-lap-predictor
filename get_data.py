import requests
import pandas as pd

all_laps = []

# Fetching lap data from 2018 to 2023 seasons
for year in range(2018, 2024):
    print(f"Fetching {year}...")
    
    # Get all races in that season
    url = f"http://ergast.com/api/f1/{year}.json"
    res = requests.get(url).json()
    races = res['MRData']['RaceTable']['Races']
    
    for race in races:
        round_num = race['round']
        circuit = race['Circuit']['circuitId']
        
        # Get lap times for each race (page through results)
        for page in range(1, 6):
            lap_url = f"http://ergast.com/api/f1/{year}/{round_num}/laps.json?limit=100&offset={(page-1)*100}"
            lap_res = requests.get(lap_url).json()
            laps = lap_res['MRData']['RaceTable']['Races']
            
            if not laps:
                break
                
            for lap in laps[0]['Laps']:
                lap_num = lap['number']
                for timing in lap['Timings']:
                    all_laps.append({
                        'year': year,
                        'circuit': circuit,
                        'round': round_num,
                        'lap': int(lap_num),
                        'driver': timing['driverId'],
                        'position': int(timing['position']),
                        'time': timing['time']
                    })

df = pd.DataFrame(all_laps)
df.to_csv('f1_laps.csv', index=False)
print(f"Done! {len(df)} rows saved to f1_laps.csv")