import os
import subprocess
import sys
from pathlib import Path

def main():
    env_name = "algo_env"
    req_file = "requirements.txt"

    # -------- Crear entorno virtual --------
    print(f"[+] Creando entorno virtual '{env_name}'...")
    subprocess.run([sys.executable, "-m", "venv", env_name], check=True)

    # -------- Activar pip e instalar dependencias --------
    pip_exe = Path(env_name) / "Scripts" / "pip.exe" if os.name == "nt" else Path(env_name) / "bin" / "pip"

    # Dependencias mínimas
    deps = [
        "matplotlib>=3.8.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0"
    ]
    print(f"[+] Instalando dependencias: {', '.join(deps)}")
    subprocess.run([str(pip_exe), "install", *deps], check=True)

    # -------- Guardar requirements.txt --------
    print(f"[+] Generando {req_file}...")
    with open(req_file, "w", encoding="utf-8") as f:
        for dep in deps:
            f.write(dep + "\n")

    print("\n✅ Entorno creado correctamente.")
    print("Para activarlo usa:\n")
    if os.name == "nt":
        print(f"    {env_name}\\Scripts\\activate")
    else:
        print(f"    source {env_name}/bin/activate")

    print("\nY para ejecutar tu librería:\n")
    print(f"    python main.py --alg pso --obj ackley --dim 2 --bounds \"-5,5\" --iters 200 --pop 50 --seed 7 --animate")

if __name__ == "__main__":
    main()
