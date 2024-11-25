from unyt import physical_constants as pc

m_p = pc.mp
G = pc.G
kboltz = pc.kboltz

X_H = 0.76
mu = 1.0 / (2.0 * X_H + 0.75 * (1.0 - X_H))
mue = 1.0 / (X_H + 0.5 * (1.0 - X_H))