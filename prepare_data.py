import pandas as pd

# Load all 4 files
lap_times = pd.read_csv('lap_times.csv')
races = pd.read_csv('races.csv')
drivers = pd.read_csv('drivers.csv')
circuits = pd.read_csv('circuits.csv')

# Merge lap times with race info
df = lap_times.merge(races[['raceId', 'year', 'round', 'circuitId', 'name']], on='raceId')

# Merge with circuit info
df = df.merge(circuits[['circuitId', 'country', 'lat', 'lng', 'alt']], on='circuitId')

# Merge with driver info
df = df.merge(drivers[['driverId', 'driverRef', 'nationality']], on='driverId')

# Convert lap time from milliseconds to seconds
df['lap_time_sec'] = df['milliseconds'] / 1000

# Drop rows with missing values
df = df.dropna()

# Remove outliers (pit stops cause very long laps)
df = df[df['lap_time_sec'] < 200]
df = df[df['lap_time_sec'] > 50]

# Final columns we'll use
df = df[['year', 'round', 'lap', 'position', 'lat', 'lng', 'alt', 'lap_time_sec', 'driverRef', 'country']]

print("Shape:", df.shape)
print("\nSample data:")
print(df.head())
print("\nData types:")
print(df.dtypes)

df.to_csv('f1_cleaned.csv', index=False)
print("\n✅ Saved to f1_cleaned.csv!")