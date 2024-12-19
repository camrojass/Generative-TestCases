import re

# Función para extraer parámetros
def extract_parameters_from_text(text):
    """
    Extrae parámetros del texto utilizando NLP y expresiones regulares, clasificándolos como
    pertenecientes al body, header, query, o response.
    """
    parameters = []

    # Expresión regular para identificar parámetros con tipos
    param_pattern = re.compile(r"\b([a-zA-Z0-9_]+):\s*(string|int|float|boolean|bool)\b", re.IGNORECASE)

    # Palabras clave para identificar si el parámetro es requerido u opcional
    required_keywords = ["required", "mandatory", "must be provided"]
    optional_keywords = ["optional", "default", "nullable"]

    # Palabras clave para identificar la ubicación del parámetro
    location_keywords = {
        "query": ["query parameter", "query string"],
        "header": ["header", "http header"],
        "body": ["body", "request body"],
        "response": ["response", "response body", "output"]
    }

    # Dividir el texto en líneas
    lines = text.split("\n")
    for line in lines:
        match = param_pattern.search(line)
        if match:
            param_name = match.group(1)
            param_type = match.group(2)

            # Determinar si el parámetro es requerido u opcional
            is_required = any(keyword in line.lower() for keyword in required_keywords)
            is_optional = any(keyword in line.lower() for keyword in optional_keywords)

            # Buscar valores predeterminados
            default_value = None
            if "default" in line.lower():
                default_match = re.search(r"default\s*[:=]\s*([\w\"']+)", line, re.IGNORECASE)
                if default_match:
                    default_value = default_match.group(1)

            # Determinar la ubicación del parámetro
            source = "unknown"
            for loc, keywords in location_keywords.items():
                if any(keyword in line.lower() for keyword in keywords):
                    source = loc
                    break

            # Agregar el parámetro con su clasificación
            parameters.append({
                "name": param_name,
                "type": param_type,
                "required": is_required and not is_optional,
                "default": default_value,
                "source": source
            })

    return parameters

#Función para Extraer Códigos HTTP
def extract_http_statuses(text):
    """
    Extrae solo los códigos de estado HTTP desde el texto.
    """
    http_statuses = []
    # Expresión regular para detectar códigos HTTP (100-599)
    status_pattern = re.compile(r"\b(1\d{2}|2\d{2}|3\d{2}|4\d{2}|5\d{2})\b(?:\s*[:\-]?\s*(.*))?")

    # Limpiar y dividir el texto en líneas
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines:
        match = status_pattern.search(line)
        if match:
            status_code = match.group(1)  # Captura solo el código
            http_statuses.append(status_code)
    return http_statuses


# Funcion para extraer métodos
def extract_method_from_text(line):
    """
    Extrae el método HTTP de una línea de texto.
    """
    match = re.search(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|TRACE|CONNECT)\b", line, re.IGNORECASE)
    if match:
        return match.group(1).upper()  # Devolver el método en mayúsculas
    return "GET"  # Valor predeterminado


# Función para extraer endpoints y métodos usando heurísticas
def extract_endpoints_and_methods(docs):
    """
    Extrae endpoints, métodos y parámetros desde los documentos procesados utilizando heurísticas.
    :param docs:
    :return: extracted_cases
    """
    extracted_cases = []
    for doc_index, doc in enumerate(docs):
        text = doc.page_content
        lines = text.split("\n")
        for line_index, line in enumerate(lines):
            if "/api/" in line:  # Detectar líneas que contengan endpoints
                method = extract_method_from_text(line)  # Usar la nueva lógica de extracción
                endpoint = line.split()[0]
                surrounding_text = " ".join(lines[max(0, line_index - 2):line_index + 2])

                # Extraer parámetros y códigos de estado HTTP
                parameters = extract_parameters_from_text(surrounding_text)
                http_statuses = extract_http_statuses(surrounding_text)
                case_id = f"doc_{doc_index}_line_{line_index}"
                extracted_cases.append({
                    "id": case_id,
                    "endpoint": endpoint,
                    "method": method,
                    "parameters": parameters,
                    "http_statuses": http_statuses
                })
    return extracted_cases
