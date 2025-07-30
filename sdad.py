import subprocess
import os

def encontrar_archivos_grandes_git(top_n=20):
    print("🔍 Buscando objetos en el historial de Git...")

    # Paso 1: obtener todos los objetos del historial
    try:
        objetos = subprocess.check_output(
            ["git", "rev-list", "--objects", "--all"],
            text=True
        ).splitlines()
    except subprocess.CalledProcessError as e:
        print("❌ Error al obtener los objetos de Git:", e)
        return

    resultados = []

    for linea in objetos:
        partes = linea.strip().split(" ", 1)
        if len(partes) != 2:
            continue
        sha, path = partes
        try:
            size = int(subprocess.check_output(
                ["git", "cat-file", "-s", sha],
                text=True
            ).strip())
            resultados.append((size, path))
        except subprocess.CalledProcessError:
            continue

    # Ordenar por tamaño descendente
    resultados.sort(reverse=True, key=lambda x: x[0])

    print(f"\n📦 Top {top_n} archivos más grandes encontrados en el historial de Git:\n")
    for i, (size, path) in enumerate(resultados[:top_n], start=1):
        size_mb = size / (1024 * 1024)
        print(f"{i:2d}. {path:60} {size_mb:.2f} MB")

if __name__ == "__main__":
    encontrar_archivos_grandes_git(20)
