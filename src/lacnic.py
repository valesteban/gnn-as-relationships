
URL = "https://ftp.lacnic.net/pub/stats/lacnic/delegated-lacnic-extended-latest"
REQUEST_TIMEOUT = 30
from pathlib import Path
import urllib.request
from typing import Dict, Set, Tuple
    
LACNIC_EXTENDED_URL = "https://ftp.lacnic.net/pub/stats/lacnic/delegated-lacnic-extended-latest"


def ensure_lacnic_file(local_path: str) -> str:
    path = Path(local_path)
    if path.exists():
        print(f"Archivo LACNIC encontrado: {path}") 
        return str(path)
    
    print("No se encontró archivo LACNIC local.\nDescargando ... ")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LACNIC_EXTENDED_URL, path)
    print(f"[OK] Archivo descargado: {path}")    
    return str(path)

def load_country_asns_from_lacnic(delegated_file: str, country_code: str) -> set[str]:
    """
    Carga ASNs de un país a partir de delegated-lacnic-extended-latest.

    Formato esperado:
    registry|cc|type|start|value|date|status|...

    Para type=asn:
    - start = primer ASN
    - value = cantidad de ASNs consecutivos
    """

    delegated_file = ensure_lacnic_file(delegated_file)
    country_code = country_code.upper() #CL
    country_asns: Set[str] = set()

    with open(delegated_file,"r",encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) < 7:
                continue

            registry, cc, resource_type, start, value, date, status = parts[:7]

            if registry.lower() != "lacnic":
                continue
            if cc.upper() != country_code:
                continue
            if resource_type.lower() != "asn":
                continue
            if status.lower() not in {"allocated", "assigned"}:
                continue

            if not start.isdigit() or not value.isdigit():
                continue

            start_asn = int(start)
            count = int(value)

            for asn in range(start_asn, start_asn + count):
                country_asns.add(str(asn))
    
    print(f"[OK] ASNs {country_code} obtenidos desde LACNIC: {len(country_asns)}")
    return country_asns