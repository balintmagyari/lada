import numpy as np
import pandas as pd

def parse_frame(lines, start, end):
    for i in range(start, end):
        if lines[i].startswith("ITEM: TIMESTEP"):
            timestep = int(lines[i+1].strip())
        
        if lines[i].startswith("ITEM: BOX BOUNDS"):
            xlo, xhi = map(float, lines[i+1].split())
            ylo, yhi = map(float, lines[i+2].split())
            zlo, zhi = map(float, lines[i+3].split())
            box_bounds = np.array([xlo, ylo, zlo]), np.array([xhi, yhi, zhi])

        if lines[i].startswith("ITEM: ATOMS"):
            cols = lines[i].strip().split()[2:]  # e.g. id type x y z
            atom_start = i + 1
            break
    
    data = []
    for line in lines[atom_start:end]:
        if line.startswith("ITEM:"):
            break
        data.append(line.rstrip())

    df = pd.DataFrame([row.split() for row in data], columns=cols)
    df = df.apply(pd.to_numeric)
    return timestep, box_bounds, df