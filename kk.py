import finrl
import os

# Ruta base del paquete finrl
base_path = os.path.dirname(finrl.__file__)

# Buscar el archivo env_stocktrading.py recursivamente
for root, dirs, files in os.walk(base_path):
    if 'env_stocktrading.py' in files:
        print("Ruta encontrada:", os.path.join(root, 'env_stocktrading.py'))
        break
else:
    print("env_stocktrading.py no encontrado en el paquete finrl.")