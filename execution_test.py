import os
import config

output_json_path = config.output_file_path

def execute_tests(output_json_path):
    """
    Ejecuta las pruebas generadas en Postman utilizando Newman CLI.
    """
    try:
        print(f"Ejecutando pruebas en Postman con Newman para el archivo {output_json_path}...")
        os.system(f'newman run {output_json_path} --reporters cli,html --reporter-html-export results.html')
        print("Pruebas ejecutadas con éxito. Revisa el archivo 'results.html' para ver el reporte.")
    except Exception as e:
        print(f"Error al ejecutar las pruebas en Postman: {e}")
        raise

# Ejecución de las pruebas con Newman.
execute_tests(output_json_path)