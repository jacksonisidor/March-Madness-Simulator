import duckdb

input = "/Users/jacksonisidor/Documents/March-Madness-Simulator/data/all_matchup_stats.parquet"
output = "/Users/jacksonisidor/Documents/March-Madness-Simulator/data/seed_odds.parquet"

# connect to DuckDB
con = duckdb.connect()

# get all unique seeds from the dataset
con.execute(f"""
    CREATE TABLE unique_seeds AS 
    SELECT DISTINCT seed_1 AS seed FROM read_parquet('{input}')
    UNION
    SELECT DISTINCT seed_2 AS seed FROM read_parquet('{input}');
""")

# generate all possible matchups (cross product)
con.execute(f"""
    CREATE TABLE all_matchups AS
    SELECT a.seed AS seed_a, b.seed AS seed_b
    FROM unique_seeds a, unique_seeds b
    WHERE a.seed <= b.seed;
""")

# compute actual win probabilities from the dataset
con.execute(f"""
    CREATE TABLE computed_odds AS
    SELECT 
        LEAST(seed_1, seed_2) AS seed_a, 
        GREATEST(seed_1, seed_2) AS seed_b,
        SUM(CASE 
            WHEN winner = 1 AND seed_1 < seed_2 THEN 1
            WHEN winner = 0 AND seed_2 < seed_1 THEN 1
            ELSE 0 END) * 1.0 / COUNT(*) AS odds
    FROM read_parquet('{input}')
    WHERE type = 'T'
    GROUP BY seed_a, seed_b;
""")

# merge all possible matchups with computed odds 
## equal seeds and matchups that never occured need to be set now (equal seeds = 0.5, never occurred = 1.0 to the lower seed)
con.execute(f"""
    CREATE TABLE seed_odds AS
    SELECT 
        a.seed_a AS seed_1, 
        a.seed_b AS seed_2,
        CASE 
            WHEN a.seed_a = a.seed_b THEN 0.5 
            WHEN c.odds IS NULL THEN 1.0       
            ELSE c.odds  
            END
        AS odds   
    FROM all_matchups a
    LEFT JOIN computed_odds c
        ON a.seed_a = c.seed_a AND a.seed_b = c.seed_b;
""")

# Save to Parquet
con.execute(f"COPY seed_odds TO '{output}' (FORMAT PARQUET)")

# Close connection
con.close()
